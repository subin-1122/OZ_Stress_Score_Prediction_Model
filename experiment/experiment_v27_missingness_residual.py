"""
결측 signature 기반 계층적 잔차 보정
====================================

목적
----
train에서 결측은 family_medical_history, medical_history, mean_working,
edu_level 네 열에만 나타나며 16개 signature를 만든다. 현재 최종 파이프라인
뒤에 남은 잔차가 이 signature와 관련 있는지 train-only CV로 검증한다.

검증
----
1. 현재 고신뢰 연결 + mean_working + snapping OOF를 기준으로 사용한다.
2. 결측 개수 또는 4-bit signature별 잔차 중앙값을 계산한다.
3. 작은 그룹은 n/(n+k)로 0 방향 shrinkage를 적용한다.
4. 고정 100-fold 비교와 방식 선택까지 분리한 nested CV를 모두 수행한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 각 검증 행의 정답은 해당 행의 correction 계산에 들어가지 않는다.
- config와 alpha 선택은 outer fold의 inner OOF에서만 수행한다.

실행
----
    .venv/bin/python experiment/experiment_v27_missingness_residual.py
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
MISSING_COLUMNS = (
    "family_medical_history",
    "medical_history",
    "mean_working",
    "edu_level",
)

DEPLOYMENT_SPLITS = 100
CORRECTION_SEEDS = (11, 101, 1001)
OUTER_SPLITS = 10
INNER_SPLITS = 5
META_SEEDS = (11, 101, 1001, 2026, 31415)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
TARGET_OOF_GAIN = 0.0022

CONFIGS = (
    {"name": "missing_count_k10", "mode": "count", "shrink": 10.0},
    {"name": "missing_count_k30", "mode": "count", "shrink": 30.0},
    {"name": "signature_k10", "mode": "signature", "shrink": 10.0},
    {"name": "signature_k30", "mode": "signature", "shrink": 30.0},
    {"name": "signature_k60", "mode": "signature", "shrink": 60.0},
    {
        "name": "count_then_signature_k30",
        "mode": "hierarchical",
        "count_shrink": 20.0,
        "signature_shrink": 30.0,
    },
    {
        "name": "count_then_signature_k60",
        "mode": "hierarchical",
        "count_shrink": 20.0,
        "signature_shrink": 60.0,
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


def missing_keys(train: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = train.loc[:, MISSING_COLUMNS].isna().to_numpy(np.int8)
    signature = np.asarray(
        ["".join(map(str, row.tolist())) for row in missing],
        dtype=object,
    )
    count = missing.sum(axis=1).astype(np.int8)
    return {"signature": signature, "count": count}


def shrunken_effect(
    fit_key: np.ndarray,
    query_key: np.ndarray,
    fit_residual: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """그룹 중앙값에 n/(n+k)를 곱해 query와 fit 효과를 함께 반환한다."""
    table = pd.DataFrame(
        {"key": fit_key, "residual": fit_residual}
    ).groupby("key")["residual"].agg(["median", "size"])
    table["effect"] = table["median"] * (
        table["size"] / (table["size"] + shrinkage)
    )
    mapping = table["effect"]
    fit_effect = pd.Series(fit_key).map(mapping).fillna(0.0).to_numpy(float)
    query_effect = pd.Series(query_key).map(mapping).fillna(0.0).to_numpy(float)
    return fit_effect, query_effect


def correction_for_config(
    keys: dict[str, np.ndarray],
    residual: np.ndarray,
    fit_idx: np.ndarray,
    query_idx: np.ndarray,
    config: dict,
) -> np.ndarray:
    """fit 행만으로 지정한 결측 패턴 보정을 계산해 query에 적용한다."""
    if config["mode"] == "count":
        _, query_effect = shrunken_effect(
            keys["count"][fit_idx],
            keys["count"][query_idx],
            residual[fit_idx],
            float(config["shrink"]),
        )
        return query_effect

    if config["mode"] == "signature":
        _, query_effect = shrunken_effect(
            keys["signature"][fit_idx],
            keys["signature"][query_idx],
            residual[fit_idx],
            float(config["shrink"]),
        )
        return query_effect

    fit_count_effect, query_count_effect = shrunken_effect(
        keys["count"][fit_idx],
        keys["count"][query_idx],
        residual[fit_idx],
        float(config["count_shrink"]),
    )
    remaining = residual[fit_idx] - fit_count_effect
    _, query_signature_effect = shrunken_effect(
        keys["signature"][fit_idx],
        keys["signature"][query_idx],
        remaining,
        float(config["signature_shrink"]),
    )
    return query_count_effect + query_signature_effect


def deployment_style_correction(
    keys: dict[str, np.ndarray],
    residual: np.ndarray,
    linked: np.ndarray,
    config: dict,
) -> np.ndarray:
    """현재 파이프라인과 같은 100-fold × 3-seed OOF correction을 만든다."""
    seed_corrections = []
    n_rows = len(residual)
    for seed in CORRECTION_SEEDS:
        correction = np.zeros(n_rows, dtype=float)
        splitter = KFold(
            n_splits=DEPLOYMENT_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for train_idx, valid_idx in splitter.split(residual):
            usable = train_idx[~linked[train_idx]]
            correction[valid_idx] = correction_for_config(
                keys,
                residual,
                usable,
                valid_idx,
                config,
            )
        seed_corrections.append(correction)
    return np.mean(np.column_stack(seed_corrections), axis=1)


def inner_select(
    keys: dict[str, np.ndarray],
    residual: np.ndarray,
    current: np.ndarray,
    y: np.ndarray,
    usable_outer: np.ndarray,
    seed: int,
) -> dict:
    """outer-train 내부 OOF에서 no-op을 포함해 config와 alpha를 고른다."""
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
                keys,
                residual,
                fit_idx,
                valid_idx,
                config,
            )

    baseline = mean_absolute_error(y[usable_outer], current[usable_outer])
    ranking = [
        {
            "config": "no_op",
            "config_order": -1,
            "alpha": 0.0,
            "alpha_order": -1,
            "inner_oof_mae": float(baseline),
        }
    ]
    for config_order, config in enumerate(CONFIGS):
        correction = corrections[config["name"]]
        for alpha_order, alpha in enumerate(ALPHAS):
            prediction = snap(current[usable_outer] + alpha * correction)
            ranking.append(
                {
                    "config": config["name"],
                    "config_order": config_order,
                    "alpha": float(alpha),
                    "alpha_order": alpha_order,
                    "inner_oof_mae": float(
                        mean_absolute_error(y[usable_outer], prediction)
                    ),
                }
            )
    # 완전히 같으면 보정을 하지 않는 no-op을 우선한다.
    return min(
        ranking,
        key=lambda row: (
            row["inner_oof_mae"],
            0 if row["config"] == "no_op" else 1,
            row["config_order"],
            row["alpha_order"],
        ),
    )


def nested_prediction(
    train: pd.DataFrame,
    keys: dict[str, np.ndarray],
    y: np.ndarray,
    context: dict,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """보정 선택과 outer 평가를 분리한 nested OOF 예측을 만든다."""
    current = context["current"]
    linked = context["mask"]
    residual = y - current
    candidate = current.copy()
    choices = []
    splitter = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        usable_outer = train_idx[~linked[train_idx]]
        valid_unlinked = valid_idx[~linked[valid_idx]]
        choice = inner_select(
            keys,
            residual,
            current,
            y,
            usable_outer,
            seed + fold,
        )
        changed = 0
        if choice["config"] != "no_op":
            config = next(
                item for item in CONFIGS if item["name"] == choice["config"]
            )
            correction = correction_for_config(
                keys,
                residual,
                usable_outer,
                valid_unlinked,
                config,
            )
            updated = snap(
                current[valid_unlinked] + choice["alpha"] * correction
            )
            candidate[valid_unlinked] = updated
            changed = int(np.sum(updated != current[valid_unlinked]))
        choice.update(
            {
                "fold": fold,
                "outer_train_unlinked": int(len(usable_outer)),
                "outer_valid_unlinked": int(len(valid_unlinked)),
                "outer_changed_rows": changed,
            }
        )
        choices.append(choice)
    return candidate, choices


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
    keys = missing_keys(train)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    fixed_rows = []
    for context in contexts:
        current = context["current"]
        residual = y - current
        current_mae = mean_absolute_error(y, current)
        for config in CONFIGS:
            correction = deployment_style_correction(
                keys,
                residual,
                context["mask"],
                config,
            )
            for alpha in ALPHAS:
                candidate = current.copy()
                unlinked = ~context["mask"]
                candidate[unlinked] = snap(
                    current[unlinked] + alpha * correction[unlinked]
                )
                score = mean_absolute_error(y, candidate)
                fixed_rows.append(
                    {
                        "link_seed": context["seed"],
                        "config": config["name"],
                        "alpha": float(alpha),
                        "candidate_oof_mae": float(score),
                        "gain_vs_current": float(current_mae - score),
                        "changed_rows": int(np.sum(candidate != current)),
                    }
                )

    fixed_ranking = []
    for config in CONFIGS:
        for alpha in ALPHAS:
            subset = [
                row
                for row in fixed_rows
                if row["config"] == config["name"] and row["alpha"] == alpha
            ]
            gains = np.asarray([row["gain_vs_current"] for row in subset])
            scores = np.asarray([row["candidate_oof_mae"] for row in subset])
            fixed_ranking.append(
                {
                    "config": config["name"],
                    "alpha": float(alpha),
                    "oof_mae_mean": float(scores.mean()),
                    "gain_mean_vs_current": float(gains.mean()),
                    "wins": int((gains > 0).sum()),
                    "ties": int((gains == 0).sum()),
                    "losses": int((gains < 0).sum()),
                }
            )
    fixed_ranking.sort(
        key=lambda row: row["gain_mean_vs_current"],
        reverse=True,
    )

    nested_rows = []
    for context in contexts:
        for meta_seed in META_SEEDS:
            candidate, choices = nested_prediction(
                train,
                keys,
                y,
                context,
                int(meta_seed),
            )
            current_mae = mean_absolute_error(y, context["current"])
            candidate_mae = mean_absolute_error(y, candidate)
            nested_rows.append(
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

    nested_gains = np.asarray(
        [row["gain_vs_current"] for row in nested_rows]
    )
    nested_scores = np.asarray(
        [row["candidate_oof_mae"] for row in nested_rows]
    )
    choice_counts: dict[str, int] = {}
    for row in nested_rows:
        for choice in row["fold_choices"]:
            name = choice["config"]
            choice_counts[name] = choice_counts.get(name, 0) + 1

    all_improve = bool(np.all(nested_gains > 0))
    mean_gain = float(nested_gains.mean())
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and all_improve)
    metrics = {
        "protocol": {
            "test_used": False,
            "missing_columns": MISSING_COLUMNS,
            "unique_signatures": int(len(np.unique(keys["signature"]))),
            "deployment_splits": DEPLOYMENT_SPLITS,
            "correction_seeds": CORRECTION_SEEDS,
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "meta_seeds": META_SEEDS,
            "configs": CONFIGS,
            "alphas": ALPHAS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "deployment_style_fixed_results": fixed_rows,
        "deployment_style_fixed_ranking": fixed_ranking,
        "nested_results": nested_rows,
        "nested_choice_counts": choice_counts,
        "summary": {
            "candidate_oof_mae_mean": float(nested_scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "gain_min": float(nested_gains.min()),
            "gain_max": float(nested_gains.max()),
            "wins": int((nested_gains > 0).sum()),
            "ties": int((nested_gains == 0).sum()),
            "losses": int((nested_gains < 0).sum()),
            "all_repeats_improve": all_improve,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
        },
    }
    output_path = ROOT / "outputs/experiment_v27_missingness_residual_metrics.json"
    output_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Missingness residual =====")
    print("100-fold 고정 비교 상위 5개:")
    for row in fixed_ranking[:5]:
        print(
            f"  {row['config']:30s} alpha={row['alpha']:.2f} "
            f"OOF={row['oof_mae_mean']:.9f} "
            f"gain={row['gain_mean_vs_current']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']}"
        )
    print(
        f"nested OOF={nested_scores.mean():.9f} "
        f"gain={mean_gain:+.9f} "
        f"승/무/패={(nested_gains > 0).sum()}/"
        f"{(nested_gains == 0).sum()}/{(nested_gains < 0).sum()}"
    )
    print(f"nested 선택 횟수={choice_counts}")
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
