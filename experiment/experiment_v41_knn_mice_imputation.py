"""Tier 3-2: fold-safe median/KNN/MICE 결측치 처리 5-fold 스크리닝."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 5151
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 1.00)
TARGET_OOF_GAIN = 0.0022
IMPUTERS = ("median", "knn5", "knn10", "mice")


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


def make_numeric_imputer(name):
    if name == "median":
        return SimpleImputer(strategy="median", add_indicator=True)
    if name == "knn5":
        return KNNImputer(n_neighbors=5, weights="distance", add_indicator=True)
    if name == "knn10":
        return KNNImputer(n_neighbors=10, weights="distance", add_indicator=True)
    if name == "mice":
        return IterativeImputer(
            max_iter=15,
            initial_strategy="median",
            add_indicator=True,
            random_state=SCREEN_SEED,
            skip_complete=True,
        )
    raise ValueError(name)


def make_preprocessor(numeric, categorical, imputer_name):
    """모든 변환은 호출한 fold-train에서만 fit된다."""
    numeric_pipeline = Pipeline(
        [
            # StandardScaler는 NaN을 보존하므로 거리 기반 imputation 전에
            # 변수 단위만 맞출 수 있다.
            ("scaler", StandardScaler()),
            ("imputer", make_numeric_imputer(imputer_name)),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v1 = v31.load_module(
        "experiment_v1_for_v41", ROOT / "experiment/experiment_v1_models.py"
    )
    v3 = v31.load_module(
        "experiment_v3_for_v41",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v41",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v41",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = v1.create_domain_features(
        train.drop(columns=[ID_COLUMN, TARGET])
    ).replace([np.inf, -np.inf], np.nan)
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    contexts = v24.make_link_contexts(v4, train, y, stored)

    predictions = {name: np.zeros(len(train)) for name in IMPUTERS}
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        dimensions = {}
        for imputer_name in IMPUTERS:
            preprocessor = make_preprocessor(numeric, categorical, imputer_name)
            x_train = preprocessor.fit_transform(features.iloc[train_idx])
            x_valid = preprocessor.transform(features.iloc[valid_idx])
            model = ExtraTreesRegressor(
                n_estimators=N_ESTIMATORS,
                criterion="squared_error",
                max_features=1,
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                random_state=SCREEN_SEED * 100 + fold,
            )
            model.fit(x_train, y[train_idx])
            predictions[imputer_name][valid_idx] = v3.trimmed_tree_prediction(
                model, x_valid, TRIM_RATIO
            )
            dimensions[imputer_name] = int(x_train.shape[1])
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "transformed_dimensions": dimensions,
            }
        )
        print(
            f"[imputation] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for imputer_name, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = (1.0 - blend_weight) * baseline + blend_weight * alternative
            rows = []
            for context in contexts:
                candidate = v24.finalize(
                    v4, train, y, raw, context["mask"], context["label"]
                )
                current_mae = mean_absolute_error(y, context["current"])
                score = mean_absolute_error(y, candidate)
                rows.append(
                    {
                        "link_seed": int(context["seed"]),
                        "candidate_oof_mae": float(score),
                        "gain_vs_current": float(current_mae - score),
                    }
                )
            gains = np.asarray([row["gain_vs_current"] for row in rows])
            scores = np.asarray([row["candidate_oof_mae"] for row in rows])
            results.append(
                {
                    "imputer": imputer_name,
                    "blend_weight": blend_weight,
                    "candidate_oof_mae_mean": float(scores.mean()),
                    "gain_mean": float(gains.mean()),
                    "gain_min": float(gains.min()),
                    "gain_max": float(gains.max()),
                    "wins": int((gains > 0).sum()),
                    "ties": int((gains == 0).sum()),
                    "losses": int((gains < 0).sum()),
                    "passes_screen": bool(
                        gains.mean() >= TARGET_OOF_GAIN and np.all(gains > 0)
                    ),
                    "context_results": rows,
                }
            )
    results.sort(key=lambda row: row["gain_mean"], reverse=True)
    metrics = {
        "protocol": {
            "test_used": False,
            "screen_seed": SCREEN_SEED,
            "n_splits": N_SPLITS,
            "imputers": IMPUTERS,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "numeric_missing_counts": {
            column: int(features[column].isna().sum())
            for column in numeric
            if features[column].isna().any()
        },
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v41_knn_mice_imputation_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v41_knn_mice_imputation_oof.npz",
        **predictions,
    )
    print("\n===== KNN/MICE imputation screen =====")
    print(f"numeric missing={metrics['numeric_missing_counts']}")
    for row in results[:8]:
        print(
            f"{row['imputer']:7s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
