"""
Train 연결 그룹 중복 완화 실험
================================

같은 fold-train 안에서 deterministic feature 규칙을 만족하고 target도 같은 행을
연결 그룹으로 만든 뒤, 중복 그룹이 ExtraTrees 학습을 과도하게 지배하는지 본다.

세 모델을 동일한 10-fold x 5-seed에서 비교한다.
1. baseline: 모든 fold-train 행의 가중치가 같다.
2. inverse_weight: 각 행에 1 / 연결 그룹 크기 가중치를 준다.
3. collapsed: 연결 그룹을 숫자 중앙값·범주 최빈값 대표행 하나로 축약한다.

누수 방지
---------
- 그룹은 매 fold의 train_idx 안에서만 만든다.
- validation 행의 feature와 target은 그룹 생성과 전처리 fit에 사용하지 않는다.
- test 데이터는 이 스크리닝에 사용하지 않는다.
- 세 후보는 실행 전에 고정하며 결과를 보고 후보를 추가하지 않는다.
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
SEEDS = (11, 101, 1001, 2026, 31415)
N_SPLITS = 10
N_ESTIMATORS = 250
TRIM_RATIO = 0.10
MIN_SUBMISSION_GAIN = 0.0003


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def union_find_components(
    train_idx: np.ndarray,
    rule_pairs: np.ndarray,
    y: np.ndarray,
) -> list[np.ndarray]:
    """fold-train 내부의 same-target 규칙 쌍만 연결 요소로 묶는다."""
    train_set = np.zeros(len(y), dtype=bool)
    train_set[train_idx] = True
    pairs = rule_pairs[
        train_set[rule_pairs[:, 0]]
        & train_set[rule_pairs[:, 1]]
        & (y[rule_pairs[:, 0]] == y[rule_pairs[:, 1]])
    ]

    parent = {int(index): int(index) for index in train_idx}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in pairs:
        union(int(left), int(right))

    groups: dict[int, list[int]] = {}
    for index in train_idx:
        groups.setdefault(find(int(index)), []).append(int(index))
    return [np.asarray(members, dtype=int) for members in groups.values()]


def inverse_group_weights(
    train_idx: np.ndarray,
    components: list[np.ndarray],
) -> np.ndarray:
    """각 연결 그룹의 총 학습 가중치가 1이 되도록 행 가중치를 만든다."""
    global_weight: dict[int, float] = {}
    for members in components:
        weight = 1.0 / len(members)
        for index in members:
            global_weight[int(index)] = weight
    return np.asarray([global_weight[int(index)] for index in train_idx])


def collapse_components(
    model_features: pd.DataFrame,
    y: np.ndarray,
    components: list[np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    """숫자는 중앙값, 범주는 최빈값으로 연결 그룹 대표행을 만든다."""
    numeric = model_features.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in model_features.columns if c not in numeric]
    rows: list[dict] = []
    targets: list[float] = []

    for members in components:
        group = model_features.iloc[members]
        row: dict = {}
        for column in numeric:
            row[column] = group[column].median(skipna=True)
        for column in categorical:
            values = group[column].astype("string").fillna("__MISSING__")
            modes = values.mode(dropna=False)
            selected = modes.iloc[0] if len(modes) else "__MISSING__"
            row[column] = np.nan if selected == "__MISSING__" else selected
        rows.append(row)
        # component는 같은 target을 가진 쌍만 합쳤으므로 첫 값을 사용해도 같다.
        targets.append(float(y[members[0]]))
    return pd.DataFrame(rows, columns=model_features.columns), np.asarray(targets)


def fit_predict(
    v3,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    preprocessor = v3.build_preprocessor(x_train)
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_valid = preprocessor.transform(x_valid)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        criterion="squared_error",
        max_features=1,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(transformed_train, y_train, sample_weight=sample_weight)
    return v3.trimmed_tree_prediction(model, transformed_valid, TRIM_RATIO)


def main() -> None:
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    y = train[TARGET].to_numpy(float)
    raw_features = train.drop(columns=[ID_COLUMN, TARGET])
    model_features = v3.make_model_features(train.drop(columns=[TARGET]))
    numeric = raw_features.select_dtypes(include=np.number).columns.tolist()
    categorical = raw_features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(raw_features, numeric, categorical)

    seed_results: list[dict] = []
    for seed in SEEDS:
        predictions = {
            "baseline": np.zeros(len(train)),
            "inverse_weight": np.zeros(len(train)),
            "collapsed": np.zeros(len(train)),
        }
        group_stats: list[dict] = []
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        started = time.perf_counter()

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), 1):
            components = union_find_components(train_idx, rule_pairs, y)
            weights = inverse_group_weights(train_idx, components)
            collapsed_x, collapsed_y = collapse_components(
                model_features, y, components
            )
            random_state = seed * 1000 + fold
            x_train = model_features.iloc[train_idx]
            x_valid = model_features.iloc[valid_idx]

            predictions["baseline"][valid_idx] = fit_predict(
                v3, x_train, y[train_idx], x_valid, random_state
            )
            predictions["inverse_weight"][valid_idx] = fit_predict(
                v3,
                x_train,
                y[train_idx],
                x_valid,
                random_state,
                sample_weight=weights,
            )
            predictions["collapsed"][valid_idx] = fit_predict(
                v3, collapsed_x, collapsed_y, x_valid, random_state
            )

            sizes = np.asarray([len(component) for component in components])
            group_stats.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "component_rows": int(len(components)),
                    "rows_in_non_singleton_components": int(sizes[sizes > 1].sum()),
                    "non_singleton_components": int((sizes > 1).sum()),
                    "largest_component": int(sizes.max()),
                }
            )
            print(
                f"[component] seed={seed} fold={fold:02d}/{N_SPLITS} "
                f"components={len(components):,} max={sizes.max()} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

        scores = {
            name: float(mean_absolute_error(y, prediction))
            for name, prediction in predictions.items()
        }
        seed_results.append(
            {
                "seed": int(seed),
                "scores": scores,
                "inverse_weight_gain": scores["baseline"] - scores["inverse_weight"],
                "collapsed_gain": scores["baseline"] - scores["collapsed"],
                "group_stats": group_stats,
            }
        )

    baseline_scores = np.asarray(
        [row["scores"]["baseline"] for row in seed_results]
    )
    inverse_scores = np.asarray(
        [row["scores"]["inverse_weight"] for row in seed_results]
    )
    collapsed_scores = np.asarray(
        [row["scores"]["collapsed"] for row in seed_results]
    )

    variants = {}
    for name, scores in (
        ("inverse_weight", inverse_scores),
        ("collapsed", collapsed_scores),
    ):
        gains = baseline_scores - scores
        variants[name] = {
            "oof_mae_mean": float(scores.mean()),
            "gain_mean": float(gains.mean()),
            "seed_wins": int((gains > 0).sum()),
            "seed_ties": int((gains == 0).sum()),
            "seed_losses": int((gains < 0).sum()),
            "passes_screen": bool(
                gains.mean() >= MIN_SUBMISSION_GAIN and (gains > 0).all()
            ),
        }

    metrics = {
        "protocol": {
            "seeds": list(SEEDS),
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "minimum_submission_gain": MIN_SUBMISSION_GAIN,
            "test_used": False,
        },
        "rule_pair_count": int(len(rule_pairs)),
        "baseline_oof_mae_mean": float(baseline_scores.mean()),
        "variants": variants,
        "seed_results": seed_results,
    }
    output_path = ROOT / "outputs/experiment_v14_component_dedup_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 연결 그룹 중복 완화 결과 =====")
    print(f"baseline 평균       : {baseline_scores.mean():.9f}")
    for name, result in variants.items():
        print(
            f"{name:14s}: {result['oof_mae_mean']:.9f} "
            f"gain={result['gain_mean']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"통과={result['passes_screen']}"
        )
    print(f"결과 요약           : {output_path}")


if __name__ == "__main__":
    main()
