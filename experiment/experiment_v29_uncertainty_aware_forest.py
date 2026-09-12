"""
ExtraTrees 트리 분포를 이용한 uncertainty-aware 보정
=====================================================

목적
----
현재 강한 3-seed/100-fold ExtraTrees 예측은 그대로 유지한다. 별도의 5-seed
screen forest에서 트리별 예측 분포를 만들고, 분포가 비대칭이거나 불확실한
행에서만 현재 예측을 트리 중앙값 방향으로 소량 이동하는 규칙을 검증한다.

기존 v21과의 차이
-----------------
- v21은 한 forest seed의 중앙값 예측을 전체 행에 고정 비율로 섞었다.
- v29는 5개 독립 forest seed를 사용한다.
- 현재 예측 수준은 유지하고 median - trimmed_mean이라는 분포 모양만 사용한다.
- uncertainty 상·하위 행에만 적용할지 outer-train에서 선택한다.
- no-op을 포함한 규칙 선택은 meta outer fold 밖에서 이루어진다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 전처리기와 ExtraTrees는 각 forest fold의 train 부분으로만 fit한다.
- meta rule은 outer-train 정답으로만 선택하고 outer-valid에서 평가한다.

실행
----
    .venv/bin/python experiment/experiment_v29_uncertainty_aware_forest.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"

FOREST_SEEDS = (73, 307, 911, 2027, 7919)
FOREST_SPLITS = 5
FOREST_TREES = 400
TRIM_RATIO = 0.10
META_SEEDS = (11, 101, 1001, 2026, 31415)
META_SPLITS = 10
TARGET_OOF_GAIN = 0.0022

RULES = (
    {"name": "median_all_a025", "point": "median", "alpha": 0.25, "scope": "all"},
    {"name": "median_all_a050", "point": "median", "alpha": 0.50, "scope": "all"},
    {"name": "median_all_a100", "point": "median", "alpha": 1.00, "scope": "all"},
    {"name": "median_high50_a050", "point": "median", "alpha": 0.50, "scope": "high50"},
    {"name": "median_high50_a100", "point": "median", "alpha": 1.00, "scope": "high50"},
    {"name": "median_high25_a050", "point": "median", "alpha": 0.50, "scope": "high25"},
    {"name": "median_high25_a100", "point": "median", "alpha": 1.00, "scope": "high25"},
    {"name": "median_low50_a050", "point": "median", "alpha": 0.50, "scope": "low50"},
    {"name": "asym_all_a025", "point": "asymmetric", "alpha": 0.25, "scope": "all"},
    {"name": "asym_high50_a025", "point": "asymmetric", "alpha": 0.25, "scope": "high50"},
    {"name": "asym_high25_a025", "point": "asymmetric", "alpha": 0.25, "scope": "high25"},
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


def trimmed_mean(prediction: np.ndarray) -> np.ndarray:
    ordered = np.sort(prediction, axis=1)
    cut = int(ordered.shape[1] * TRIM_RATIO)
    return ordered[:, cut:-cut].mean(axis=1)


def make_distribution_oof(v3, train: pd.DataFrame, y: np.ndarray) -> dict:
    """독립 forest seed별 트리 예측 분포 통계를 OOF로 만든다."""
    features = v3.make_model_features(train.drop(columns=[TARGET]))
    outputs = {}
    started = time.perf_counter()

    for seed in FOREST_SEEDS:
        stats = {
            "trimmed_mean": np.zeros(len(train)),
            "median": np.zeros(len(train)),
            "q40": np.zeros(len(train)),
            "q60": np.zeros(len(train)),
            "std": np.zeros(len(train)),
            "iqr": np.zeros(len(train)),
            "mad": np.zeros(len(train)),
        }
        splitter = KFold(
            n_splits=FOREST_SPLITS,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(features),
            start=1,
        ):
            preprocessor = v3.build_preprocessor(features.iloc[train_idx])
            x_train = preprocessor.fit_transform(features.iloc[train_idx])
            x_valid = preprocessor.transform(features.iloc[valid_idx])
            model = ExtraTreesRegressor(
                n_estimators=FOREST_TREES,
                criterion="squared_error",
                max_features=1,
                min_samples_leaf=1,
                bootstrap=False,
                random_state=seed * 100 + fold,
                n_jobs=-1,
            )
            model.fit(x_train, y[train_idx])
            tree_prediction = np.column_stack(
                [tree.predict(x_valid) for tree in model.estimators_]
            )
            median = np.median(tree_prediction, axis=1)
            stats["trimmed_mean"][valid_idx] = trimmed_mean(tree_prediction)
            stats["median"][valid_idx] = median
            stats["q40"][valid_idx] = np.quantile(
                tree_prediction, 0.40, axis=1
            )
            stats["q60"][valid_idx] = np.quantile(
                tree_prediction, 0.60, axis=1
            )
            stats["std"][valid_idx] = np.std(tree_prediction, axis=1)
            stats["iqr"][valid_idx] = (
                np.quantile(tree_prediction, 0.75, axis=1)
                - np.quantile(tree_prediction, 0.25, axis=1)
            )
            stats["mad"][valid_idx] = np.median(
                np.abs(tree_prediction - median[:, None]),
                axis=1,
            )
        outputs[int(seed)] = stats
        print(
            f"[forest] seed={seed} 완료 "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return outputs


def rule_shift(stats: dict[str, np.ndarray], rule: dict) -> np.ndarray:
    """screen forest의 수준이 아니라 분포 모양에 해당하는 이동량만 반환한다."""
    center = stats["trimmed_mean"]
    if rule["point"] == "median":
        target = stats["median"]
    else:
        target = np.where(
            stats["median"] <= center,
            stats["q40"],
            stats["q60"],
        )
    return float(rule["alpha"]) * (target - center)


def scope_mask(
    uncertainty: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    scope: str,
) -> np.ndarray:
    """outer-train uncertainty 분포로만 query 적용 범위를 정한다."""
    if scope == "all":
        return np.ones(len(query_indices), dtype=bool)
    if scope == "high50":
        threshold = np.quantile(uncertainty[train_indices], 0.50)
        return uncertainty[query_indices] >= threshold
    if scope == "high25":
        threshold = np.quantile(uncertainty[train_indices], 0.75)
        return uncertainty[query_indices] >= threshold
    if scope == "low50":
        threshold = np.quantile(uncertainty[train_indices], 0.50)
        return uncertainty[query_indices] <= threshold
    raise ValueError(f"알 수 없는 scope: {scope}")


def apply_rule(
    current: np.ndarray,
    stats: dict[str, np.ndarray],
    rule: dict,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    linked: np.ndarray,
) -> np.ndarray:
    result = current[query_indices].copy()
    uncertainty = stats["iqr"]
    use = scope_mask(
        uncertainty,
        fit_indices,
        query_indices,
        rule["scope"],
    ) & (~linked[query_indices])
    shift = rule_shift(stats, rule)
    result[use] = snap(current[query_indices][use] + shift[query_indices][use])
    return result


def nested_rule_prediction(
    train: pd.DataFrame,
    y: np.ndarray,
    current: np.ndarray,
    linked: np.ndarray,
    stats: dict[str, np.ndarray],
    meta_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """각 outer-train에서 no-op을 포함한 rule을 고르고 validation에 적용한다."""
    candidate = current.copy()
    choices = []
    splitter = KFold(
        n_splits=META_SPLITS,
        shuffle=True,
        random_state=meta_seed,
    )
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(train),
        start=1,
    ):
        baseline = mean_absolute_error(y[train_idx], current[train_idx])
        ranking = [
            {
                "rule": "no_op",
                "train_mae": float(baseline),
                "train_gain": 0.0,
                "rule_order": -1,
            }
        ]
        for rule_order, rule in enumerate(RULES):
            train_prediction = apply_rule(
                current,
                stats,
                rule,
                train_idx,
                train_idx,
                linked,
            )
            score = mean_absolute_error(y[train_idx], train_prediction)
            ranking.append(
                {
                    "rule": rule["name"],
                    "train_mae": float(score),
                    "train_gain": float(baseline - score),
                    "rule_order": rule_order,
                }
            )
        # 동률이면 no-op, 그 다음 사전에 정의한 단순한 규칙을 우선한다.
        choice = min(
            ranking,
            key=lambda row: (
                row["train_mae"],
                0 if row["rule"] == "no_op" else 1,
                row["rule_order"],
            ),
        )
        changed = 0
        if choice["rule"] != "no_op":
            rule = next(item for item in RULES if item["name"] == choice["rule"])
            updated = apply_rule(
                current,
                stats,
                rule,
                train_idx,
                valid_idx,
                linked,
            )
            candidate[valid_idx] = updated
            changed = int(np.sum(updated != current[valid_idx]))
        choice.update(
            {
                "fold": fold,
                "valid_changed_rows": changed,
            }
        )
        choices.append(choice)
    return candidate, choices


def fixed_rule_ranking(
    y: np.ndarray,
    contexts: list[dict],
    distributions: dict,
) -> list[dict]:
    """5 forest seed × 5 linkage seed에서 각 고정 rule의 방향을 확인한다."""
    rows = []
    all_indices = np.arange(len(y))
    for forest_seed, stats in distributions.items():
        for context in contexts:
            current = context["current"]
            current_mae = mean_absolute_error(y, current)
            for rule in RULES:
                candidate = apply_rule(
                    current,
                    stats,
                    rule,
                    all_indices,
                    all_indices,
                    context["mask"],
                )
                score = mean_absolute_error(y, candidate)
                rows.append(
                    {
                        "forest_seed": int(forest_seed),
                        "link_seed": context["seed"],
                        "rule": rule["name"],
                        "candidate_oof_mae": float(score),
                        "gain_vs_current": float(current_mae - score),
                        "changed_rows": int(np.sum(candidate != current)),
                    }
                )

    ranking = []
    for rule in RULES:
        subset = [row for row in rows if row["rule"] == rule["name"]]
        gains = np.asarray([row["gain_vs_current"] for row in subset])
        scores = np.asarray([row["candidate_oof_mae"] for row in subset])
        ranking.append(
            {
                "rule": rule["name"],
                "oof_mae_mean": float(scores.mean()),
                "gain_mean_vs_current": float(gains.mean()),
                "wins": int((gains > 0).sum()),
                "ties": int((gains == 0).sum()),
                "losses": int((gains < 0).sum()),
                "details": subset,
            }
        )
    ranking.sort(key=lambda row: row["gain_mean_vs_current"], reverse=True)
    return ranking


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
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
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
    contexts = v24.make_link_contexts(v4, train, y, stored)

    distributions = make_distribution_oof(v3, train, y)
    cache_path = ROOT / "outputs/experiment_v29_forest_distribution_oof.npz"
    np.savez_compressed(
        cache_path,
        **{
            f"seed_{seed}_{name}": values
            for seed, stats in distributions.items()
            for name, values in stats.items()
        },
    )

    fixed_ranking = fixed_rule_ranking(y, contexts, distributions)
    nested_rows = []
    for forest_seed, stats in distributions.items():
        for context in contexts:
            for meta_seed in META_SEEDS:
                candidate, choices = nested_rule_prediction(
                    train,
                    y,
                    context["current"],
                    context["mask"],
                    stats,
                    int(meta_seed),
                )
                current_mae = mean_absolute_error(y, context["current"])
                candidate_mae = mean_absolute_error(y, candidate)
                nested_rows.append(
                    {
                        "forest_seed": int(forest_seed),
                        "link_seed": context["seed"],
                        "meta_seed": int(meta_seed),
                        "candidate_oof_mae": float(candidate_mae),
                        "gain_vs_current": float(current_mae - candidate_mae),
                        "changed_rows": int(
                            np.sum(candidate != context["current"])
                        ),
                        "fold_choices": choices,
                    }
                )

    gains = np.asarray([row["gain_vs_current"] for row in nested_rows])
    scores = np.asarray([row["candidate_oof_mae"] for row in nested_rows])
    changes = np.asarray([row["changed_rows"] for row in nested_rows])
    choice_counts: dict[str, int] = {}
    for row in nested_rows:
        for choice in row["fold_choices"]:
            name = choice["rule"]
            choice_counts[name] = choice_counts.get(name, 0) + 1

    all_improve = bool(np.all(gains > 0))
    mean_gain = float(gains.mean())
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and all_improve)
    metrics = {
        "protocol": {
            "test_used": False,
            "forest_seeds": FOREST_SEEDS,
            "forest_splits": FOREST_SPLITS,
            "forest_trees": FOREST_TREES,
            "meta_seeds": META_SEEDS,
            "meta_splits": META_SPLITS,
            "rules": RULES,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "fixed_rule_ranking": fixed_ranking,
        "nested_results": nested_rows,
        "nested_choice_counts": choice_counts,
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "changed_rows_mean": float(changes.mean()),
            "all_repeats_improve": all_improve,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
        },
    }
    metrics_path = (
        ROOT / "outputs/experiment_v29_uncertainty_aware_forest_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Uncertainty-aware forest =====")
    print("고정 rule 상위 5개:")
    for row in fixed_ranking[:5]:
        print(
            f"  {row['rule']:28s} OOF={row['oof_mae_mean']:.9f} "
            f"gain={row['gain_mean_vs_current']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']}"
        )
    print(
        f"nested OOF={scores.mean():.9f} gain={mean_gain:+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/"
        f"{(gains < 0).sum()} changed 평균={changes.mean():.1f}"
    )
    print(f"nested 선택 횟수={choice_counts}")
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"분포 OOF={cache_path}")
    print(f"결과 요약={metrics_path}")


if __name__ == "__main__":
    main()
