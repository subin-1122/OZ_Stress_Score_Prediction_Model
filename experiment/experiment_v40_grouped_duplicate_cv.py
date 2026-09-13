"""Tier 3-1: feature 중복 그룹 전체를 제외하는 GroupKFold 진단."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SEED = 5050
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10


def make_components(n_rows, pairs):
    """target을 보지 않고 feature 규칙 pair의 connected component를 만든다."""
    parent = np.arange(n_rows)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in pairs:
        root_left = find(int(left))
        root_right = find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    roots = np.asarray([find(index) for index in range(n_rows)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def fit_oof(v3, features, y, splits, name):
    prediction = np.zeros(len(features))
    rows = []
    started = time.perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        preprocessor = v3.build_preprocessor(features.iloc[train_idx])
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        model = ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            criterion="squared_error",
            max_features=1,
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=-1,
            random_state=SEED * 100 + fold,
        )
        model.fit(x_train, y[train_idx])
        prediction[valid_idx] = v3.trimmed_tree_prediction(
            model, x_valid, TRIM_RATIO
        )
        rows.append(
            {"fold": fold, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
        )
        print(
            f"[{name}] fold={fold}/{N_SPLITS} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return prediction, rows


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v40",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v40",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    y = train[TARGET].to_numpy(float)
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = raw.select_dtypes(include=np.number).columns.tolist()
    categorical = raw.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(raw, numeric, categorical)
    groups = make_components(len(train), pairs)
    sizes = pd.Series(groups).value_counts().to_numpy()
    features = v3.make_model_features(train.drop(columns=[TARGET]))

    ordinary = list(
        KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(train)
    )
    grouped = list(
        GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            train, y, groups
        )
    )
    ordinary_oof, ordinary_rows = fit_oof(v3, features, y, ordinary, "ordinary")
    grouped_oof, grouped_rows = fit_oof(v3, features, y, grouped, "grouped")
    ordinary_mae = mean_absolute_error(y, ordinary_oof)
    grouped_mae = mean_absolute_error(y, grouped_oof)
    metrics = {
        "protocol": {
            "test_used": False,
            "seed": SEED,
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "purpose": "diagnostic only",
        },
        "rule_pairs": int(len(pairs)),
        "groups": int(len(np.unique(groups))),
        "non_singleton_groups": int(np.sum(sizes > 1)),
        "rows_in_non_singleton_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "ordinary_oof_mae": float(ordinary_mae),
        "grouped_oof_mae": float(grouped_mae),
        "grouped_minus_ordinary": float(grouped_mae - ordinary_mae),
        "ordinary_folds": ordinary_rows,
        "grouped_folds": grouped_rows,
        "submission_created": False,
    }
    path = ROOT / "outputs/experiment_v40_grouped_duplicate_cv_metrics.json"
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        ROOT / "outputs/experiment_v40_grouped_duplicate_cv_oof.npz",
        ordinary=ordinary_oof,
        grouped=grouped_oof,
        groups=groups,
    )
    print("\n===== Grouped duplicate CV diagnostic =====")
    print(
        f"groups={metrics['groups']} non-singleton={metrics['non_singleton_groups']} "
        f"rows={metrics['rows_in_non_singleton_groups']} max={metrics['largest_group']}"
    )
    print(
        f"ordinary={ordinary_mae:.9f} grouped={grouped_mae:.9f} "
        f"차이={grouped_mae-ordinary_mae:+.9f}"
    )
    print("test를 읽지 않았으며 진단 전용이라 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
