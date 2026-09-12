"""
mean_working 구조를 이용한 계층적 잔차 보정
===========================================

목적
----
현재 유효한 mean_working별 잔차 중앙값 보정을 유지하면서 다음 두 구조를
train-only nested CV로 검증한다.

1. 가까운 근로시간끼리 잔차 정보를 공유하는 local weighted median
2. mean_working 보정 뒤 남은 잔차를 base 예측 분위와 교차 그룹으로 소량 보정

과적합 방지
-----------
- test.csv는 읽지 않는다.
- correction 방식과 alpha는 meta outer fold의 train 부분에서 inner OOF로 고른다.
- 그룹 중앙값, 분위 경계, local median은 모두 해당 fit 부분에서만 계산한다.
- 희소한 예측구간·교차 그룹 효과는 empirical-Bayes shrinkage를 적용한다.
- 5개 meta seed와 5개 linkage seed를 모두 확인한다.

실행
----
    .venv/bin/python experiment/experiment_v26_structured_work_residual.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
OUTER_SPLITS = 10
INNER_SPLITS = 5
META_SEEDS = (11, 101, 1001, 2026, 31415)
CORRECTION_SEEDS = (11, 101, 1001)
DEPLOYMENT_SPLITS = 100
ALPHAS = (0.75, 0.85, 0.93, 1.00)
TARGET_OOF_GAIN = 0.0022

CONFIGS = (
    {"name": "exact_control", "mode": "exact"},
    {"name": "smooth_bw075", "mode": "smooth", "bandwidth": 0.75},
    {"name": "smooth_bw150", "mode": "smooth", "bandwidth": 1.50},
    {"name": "smooth_bw250", "mode": "smooth", "bandwidth": 2.50},
    {
        "name": "exact_plus_predbin",
        "mode": "exact",
        "pred_bins": 5,
        "bin_shrink": 30.0,
    },
    {
        "name": "smooth150_plus_predbin",
        "mode": "smooth",
        "bandwidth": 1.50,
        "pred_bins": 5,
        "bin_shrink": 30.0,
    },
    {
        "name": "exact_predbin_interaction",
        "mode": "exact",
        "pred_bins": 5,
        "bin_shrink": 30.0,
        "interaction_shrink": 30.0,
    },
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / 0.01) * 0.01, 0.0, 1.0)


def work_keys(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("__MISSING__")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """양의 가중치에 대한 중앙값을 계산한다."""
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return float("nan")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def exact_work_effect(
    train: pd.DataFrame,
    residual: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
) -> np.ndarray:
    """현재 방식과 같은 근로시간별 잔차 중앙값을 fit 부분에서만 만든다."""
    fit_key = work_keys(train.iloc[fit_idx]["mean_working"])
    query_key = work_keys(train.iloc[query_idx]["mean_working"])
    table = pd.DataFrame(
        {"key": fit_key.to_numpy(), "residual": residual[fit_idx]}
    ).groupby("key")["residual"].median()
    fallback = float(np.median(residual[fit_idx]))
    return query_key.map(table).fillna(fallback).to_numpy(float)


def smooth_work_effect(
    train: pd.DataFrame,
    residual: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    비결측 근로시간에는 시간 차이에 따른 Gaussian 가중 중앙값을 적용한다.

    mean_working 결측은 값 자체가 의미 있는 큰 그룹이므로 별도 중앙값을 쓴다.
    """
    fit_work = train.iloc[fit_idx]["mean_working"].to_numpy(float)
    query_work = train.iloc[query_idx]["mean_working"].to_numpy(float)
    fit_residual = residual[fit_idx]
    fallback = float(np.median(fit_residual))
    missing_fit = np.isnan(fit_work)
    missing_effect = (
        float(np.median(fit_residual[missing_fit]))
        if np.any(missing_fit)
        else fallback
    )
    nonmissing = ~missing_fit
    result = np.full(len(query_idx), fallback, dtype=float)

    for value in np.unique(query_work[np.isfinite(query_work)]):
        weights = np.exp(
            -0.5 * ((fit_work[nonmissing] - value) / bandwidth) ** 2
        )
        effect = weighted_median(fit_residual[nonmissing], weights)
        result[query_work == value] = effect if np.isfinite(effect) else fallback
    result[np.isnan(query_work)] = missing_effect
    return result


def prediction_bins(
    fit_prediction: np.ndarray,
    query_prediction: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """fit prediction만으로 분위 경계를 만든 뒤 fit/query에 동일 적용한다."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(fit_prediction, quantiles))
    return (
        np.searchsorted(edges, fit_prediction, side="right"),
        np.searchsorted(edges, query_prediction, side="right"),
    )


def shrunken_group_effect(
    fit_keys: np.ndarray,
    query_keys: np.ndarray,
    fit_residual: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """그룹 중앙값을 표본 수 n/(n+k)만큼만 적용한다."""
    table = pd.DataFrame(
        {"key": fit_keys, "residual": fit_residual}
    ).groupby("key")["residual"].agg(["median", "size"])
    table["effect"] = table["median"] * (
        table["size"] / (table["size"] + shrinkage)
    )
    mapping = table["effect"]
    return pd.Series(query_keys).map(mapping).fillna(0.0).to_numpy(float)


def correction_for_config(
    train: pd.DataFrame,
    residual: np.ndarray,
    base: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    config: dict,
) -> np.ndarray:
    """한 config의 계층적 보정값을 fit에서 학습해 query에 적용한다."""
    if config["mode"] == "exact":
        fit_work = exact_work_effect(train, residual, fit_idx, fit_idx)
        query_work = exact_work_effect(train, residual, fit_idx, query_idx)
    else:
        bandwidth = float(config["bandwidth"])
        fit_work = smooth_work_effect(
            train, residual, fit_idx, fit_idx, bandwidth
        )
        query_work = smooth_work_effect(
            train, residual, fit_idx, query_idx, bandwidth
        )

    if "pred_bins" not in config:
        return query_work

    fit_bins, query_bins = prediction_bins(
        base[fit_idx],
        base[query_idx],
        int(config["pred_bins"]),
    )
    after_work = residual[fit_idx] - fit_work
    fit_bin_effect = shrunken_group_effect(
        fit_bins,
        fit_bins,
        after_work,
        float(config["bin_shrink"]),
    )
    query_bin_effect = shrunken_group_effect(
        fit_bins,
        query_bins,
        after_work,
        float(config["bin_shrink"]),
    )
    query_effect = query_work + query_bin_effect

    if "interaction_shrink" not in config:
        return query_effect

    fit_work_key = work_keys(train.iloc[fit_idx]["mean_working"]).to_numpy()
    query_work_key = work_keys(train.iloc[query_idx]["mean_working"]).to_numpy()
    fit_interaction_key = np.asarray(
        [f"{work}|{bin_id}" for work, bin_id in zip(fit_work_key, fit_bins)]
    )
    query_interaction_key = np.asarray(
        [f"{work}|{bin_id}" for work, bin_id in zip(query_work_key, query_bins)]
    )
    after_main = after_work - fit_bin_effect
    interaction = shrunken_group_effect(
        fit_interaction_key,
        query_interaction_key,
        after_main,
        float(config["interaction_shrink"]),
    )
    return query_effect + interaction


def inner_select(
    train: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    usable_outer: np.ndarray,
    seed: int,
) -> dict:
    """outer-train 내부 OOF만으로 correction config와 alpha를 선택한다."""
    residual = y - base
    corrections = {
        config["name"]: np.zeros(len(usable_outer), dtype=float)
        for config in CONFIGS
    }
    splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    for fit_pos, valid_pos in splitter.split(usable_outer):
        fit_idx = usable_outer[fit_pos]
        valid_idx = usable_outer[valid_pos]
        for config in CONFIGS:
            corrections[config["name"]][valid_pos] = correction_for_config(
                train,
                residual,
                base,
                fit_idx,
                valid_idx,
                config,
            )

    ranking = []
    for config_order, config in enumerate(CONFIGS):
        correction = corrections[config["name"]]
        for alpha_order, alpha in enumerate(ALPHAS):
            prediction = snap(base[usable_outer] + alpha * correction)
            score = mean_absolute_error(y[usable_outer], prediction)
            ranking.append(
                {
                    "config": config["name"],
                    "config_order": config_order,
                    "alpha": float(alpha),
                    "alpha_order": alpha_order,
                    "inner_oof_mae": float(score),
                }
            )
    # 동률이면 먼저 정의한 단순 config와 낮은 alpha를 선택한다.
    return min(
        ranking,
        key=lambda row: (
            row["inner_oof_mae"],
            row["config_order"],
            row["alpha_order"],
        ),
    )


def nested_structured_prediction(
    train: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    context: dict,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """선택과 평가를 분리한 outer OOF 최종 예측을 만든다."""
    current = context["current"]
    linked = context["mask"]
    residual = y - base
    candidate = current.copy()
    choices = []
    splitter = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        usable_outer = train_idx[~linked[train_idx]]
        valid_unlinked = valid_idx[~linked[valid_idx]]
        choice = inner_select(
            train,
            y,
            base,
            usable_outer,
            seed + fold,
        )
        config = next(
            item for item in CONFIGS if item["name"] == choice["config"]
        )
        correction = correction_for_config(
            train,
            residual,
            base,
            usable_outer,
            valid_unlinked,
            config,
        )
        candidate[valid_unlinked] = snap(
            base[valid_unlinked] + choice["alpha"] * correction
        )
        choice.update(
            {
                "fold": fold,
                "outer_train_unlinked": int(len(usable_outer)),
                "outer_valid_unlinked": int(len(valid_unlinked)),
            }
        )
        choices.append(choice)

    return candidate, choices


def deployment_style_correction(
    train: pd.DataFrame,
    y: np.ndarray,
    base: np.ndarray,
    linked: np.ndarray,
    config: dict,
) -> np.ndarray:
    """현재 파이프라인과 동일한 100-fold × 3-seed 평균 보정을 만든다."""
    residual = y - base
    seed_corrections = []
    for seed in CORRECTION_SEEDS:
        correction = np.zeros(len(train), dtype=float)
        splitter = KFold(
            n_splits=DEPLOYMENT_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for train_idx, valid_idx in splitter.split(train):
            usable = train_idx[~linked[train_idx]]
            correction[valid_idx] = correction_for_config(
                train,
                residual,
                base,
                usable,
                valid_idx,
                config,
            )
        seed_corrections.append(correction)
    return np.mean(np.column_stack(seed_corrections), axis=1)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = load_module(
        "experiment_v24",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    base = stored["base_oof"].to_numpy(float)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    fixed_rows = []
    for config in CONFIGS:
        for context in contexts:
            correction = deployment_style_correction(
                train,
                y,
                base,
                context["mask"],
                config,
            )
            current_mae = mean_absolute_error(y, context["current"])
            for alpha in ALPHAS:
                candidate = snap(base + alpha * correction)
                candidate[context["mask"]] = context["label"][context["mask"]]
                candidate_mae = mean_absolute_error(y, candidate)
                fixed_rows.append(
                    {
                        "config": config["name"],
                        "alpha": float(alpha),
                        "link_seed": context["seed"],
                        "candidate_oof_mae": float(candidate_mae),
                        "gain_vs_current": float(current_mae - candidate_mae),
                    }
                )

    fixed_summary = []
    for config in CONFIGS:
        for alpha in ALPHAS:
            subset = [
                row
                for row in fixed_rows
                if row["config"] == config["name"] and row["alpha"] == alpha
            ]
            variant_gains = np.asarray(
                [row["gain_vs_current"] for row in subset]
            )
            variant_scores = np.asarray(
                [row["candidate_oof_mae"] for row in subset]
            )
            fixed_summary.append(
                {
                    "config": config["name"],
                    "alpha": float(alpha),
                    "oof_mae_mean": float(variant_scores.mean()),
                    "gain_mean_vs_current": float(variant_gains.mean()),
                    "wins": int((variant_gains > 0).sum()),
                    "ties": int((variant_gains == 0).sum()),
                    "losses": int((variant_gains < 0).sum()),
                }
            )
    fixed_summary.sort(
        key=lambda row: row["gain_mean_vs_current"], reverse=True
    )

    rows = []
    for context in contexts:
        for meta_seed in META_SEEDS:
            candidate, choices = nested_structured_prediction(
                train,
                y,
                base,
                context,
                int(meta_seed),
            )
            current_mae = mean_absolute_error(y, context["current"])
            candidate_mae = mean_absolute_error(y, candidate)
            rows.append(
                {
                    "link_seed": context["seed"],
                    "meta_seed": int(meta_seed),
                    "current_oof_mae": float(current_mae),
                    "candidate_oof_mae": float(candidate_mae),
                    "gain_vs_current": float(current_mae - candidate_mae),
                    "changed_rows": int(
                        np.sum(candidate != context["current"])
                    ),
                    "fold_choices": choices,
                }
            )
            print(
                f"link={context['seed']:5d} meta={meta_seed:5d} "
                f"OOF={candidate_mae:.9f} "
                f"gain={current_mae-candidate_mae:+.9f}",
                flush=True,
            )

    gains = np.asarray([row["gain_vs_current"] for row in rows])
    scores = np.asarray([row["candidate_oof_mae"] for row in rows])
    choice_counts: dict[str, int] = {}
    alpha_counts: dict[str, int] = {}
    for row in rows:
        for choice in row["fold_choices"]:
            choice_counts[choice["config"]] = (
                choice_counts.get(choice["config"], 0) + 1
            )
            alpha_key = f"{choice['alpha']:.2f}"
            alpha_counts[alpha_key] = alpha_counts.get(alpha_key, 0) + 1

    mean_gain = float(gains.mean())
    all_improve = bool(np.all(gains > 0))
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and all_improve)
    metrics = {
        "protocol": {
            "test_used": False,
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "meta_seeds": META_SEEDS,
            "configs": CONFIGS,
            "alphas": ALPHAS,
            "target_oof_gain": TARGET_OOF_GAIN,
            "deployment_splits": DEPLOYMENT_SPLITS,
            "correction_seeds": CORRECTION_SEEDS,
        },
        "deployment_style_fixed_results": fixed_rows,
        "deployment_style_fixed_ranking": fixed_summary,
        "results": rows,
        "selection_counts": {
            "config": choice_counts,
            "alpha": alpha_counts,
        },
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "all_repeats_improve": all_improve,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
        },
    }
    output_path = ROOT / "outputs/experiment_v26_structured_work_residual_metrics.json"
    output_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Structured mean_working residual =====")
    print("100-fold 고정 비교 상위 5개:")
    for row in fixed_summary[:5]:
        print(
            f"  {row['config']:30s} alpha={row['alpha']:.2f} "
            f"OOF={row['oof_mae_mean']:.9f} "
            f"gain={row['gain_mean_vs_current']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']}"
        )
    print(
        f"OOF={scores.mean():.9f} gain={mean_gain:+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/{(gains < 0).sum()}"
    )
    print(f"config 선택 횟수={choice_counts}")
    print(f"alpha 선택 횟수={alpha_counts}")
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
