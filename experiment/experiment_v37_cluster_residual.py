"""Tier 2-2: 다변량 K-means/GMM cluster residual 5-fold 스크리닝."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 4747
N_SPLITS = 5
SHRINKAGE = 20.0
ALPHAS = (0.25, 0.50, 1.00)
TARGET_OOF_GAIN = 0.0022
CONFIGS = {
    "kmeans_4": ("kmeans", 4),
    "kmeans_8": ("kmeans", 8),
    "kmeans_16": ("kmeans", 16),
    "gmm_4": ("gmm", 4),
    "gmm_8": ("gmm", 8),
}


def snap(values):
    return np.clip(np.rint(values / 0.01) * 0.01, 0.0, 1.0)


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


def fit_cluster(kind, count, x_train, x_valid, seed):
    if kind == "kmeans":
        model = KMeans(n_clusters=count, n_init=20, random_state=seed)
    else:
        model = GaussianMixture(
            n_components=count,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            random_state=seed,
        )
    train_label = model.fit_predict(x_train)
    valid_label = model.predict(x_valid)
    return train_label, valid_label


def cluster_correction(train_label, valid_label, residual):
    """클러스터 잔차 중앙값을 크기에 따라 0 방향으로 수축한다."""
    correction = np.zeros(len(valid_label), dtype=float)
    for label in np.unique(train_label):
        group = residual[train_label == label]
        value = np.median(group) * len(group) / (len(group) + SHRINKAGE)
        correction[valid_label == label] = value
    return correction


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v1 = v31.load_module(
        "experiment_v1_for_v37", ROOT / "experiment/experiment_v1_models.py"
    )
    v4 = v31.load_module(
        "experiment_v4_for_v37",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v37",
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

    candidate = {
        (context_index, config, alpha): context["current"].copy()
        for context_index, context in enumerate(contexts)
        for config in CONFIGS
        for alpha in ALPHAS
    }
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        preprocessor = v1.make_preprocessor(
            numeric, categorical, scale_numeric=True
        )
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        labels = {
            name: fit_cluster(
                kind,
                count,
                x_train,
                x_valid,
                SCREEN_SEED * 100 + fold,
            )
            for name, (kind, count) in CONFIGS.items()
        }
        for context_index, context in enumerate(contexts):
            residual = y[train_idx] - context["current"][train_idx]
            for config_name, (train_label, valid_label) in labels.items():
                correction = cluster_correction(
                    train_label, valid_label, residual
                )
                unlinked = ~context["mask"][valid_idx]
                for alpha in ALPHAS:
                    updated = context["current"][valid_idx].copy()
                    updated[unlinked] = snap(
                        updated[unlinked] + alpha * correction[unlinked]
                    )
                    candidate[(context_index, config_name, alpha)][valid_idx] = updated
        fold_rows.append(
            {"fold": fold, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
        )
        print(
            f"[cluster] fold={fold}/{N_SPLITS} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    results = []
    for config_name in CONFIGS:
        for alpha in ALPHAS:
            rows = []
            for context_index, context in enumerate(contexts):
                prediction = candidate[(context_index, config_name, alpha)]
                current_mae = mean_absolute_error(y, context["current"])
                score = mean_absolute_error(y, prediction)
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
                    "config": config_name,
                    "alpha": alpha,
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
            "configs": CONFIGS,
            "alphas": ALPHAS,
            "shrinkage": SHRINKAGE,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v37_cluster_residual_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n===== Cluster residual screen =====")
    for row in results[:6]:
        print(
            f"{row['config']:10s} alpha={row['alpha']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
