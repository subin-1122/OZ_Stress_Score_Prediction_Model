"""Tier 2-1: RandomForest train-only 5-fold OOF 스크리닝."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 4646
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 1.00)
TARGET_OOF_GAIN = 0.0022
CONFIGS = {
    "maxfeat1_leaf1": {"max_features": 1, "min_samples_leaf": 1},
    "sqrt_leaf1": {"max_features": "sqrt", "min_samples_leaf": 1},
    "half_leaf2": {"max_features": 0.5, "min_samples_leaf": 2},
}


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


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v36",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v36",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v36",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = v3.make_model_features(train.drop(columns=[TARGET]))
    contexts = v24.make_link_contexts(v4, train, y, stored)

    predictions = {name: np.zeros(len(train)) for name in CONFIGS}
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        preprocessor = v3.build_preprocessor(features.iloc[train_idx])
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        for name, config in CONFIGS.items():
            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                criterion="squared_error",
                max_features=config["max_features"],
                min_samples_leaf=config["min_samples_leaf"],
                bootstrap=True,
                n_jobs=-1,
                random_state=SCREEN_SEED * 100 + fold,
            )
            model.fit(x_train, y[train_idx])
            predictions[name][valid_idx] = v3.trimmed_tree_prediction(
                model, x_valid, TRIM_RATIO
            )
        fold_rows.append(
            {"fold": fold, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
        )
        print(
            f"[RandomForest] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for variant, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = (1.0 - blend_weight) * baseline + blend_weight * alternative
            context_rows = []
            for context in contexts:
                candidate = v24.finalize(
                    v4, train, y, raw, context["mask"], context["label"]
                )
                current_mae = mean_absolute_error(y, context["current"])
                score = mean_absolute_error(y, candidate)
                context_rows.append(
                    {
                        "link_seed": int(context["seed"]),
                        "candidate_oof_mae": float(score),
                        "gain_vs_current": float(current_mae - score),
                    }
                )
            gains = np.asarray([row["gain_vs_current"] for row in context_rows])
            scores = np.asarray([row["candidate_oof_mae"] for row in context_rows])
            results.append(
                {
                    "variant": variant,
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
                    "context_results": context_rows,
                }
            )
    results.sort(key=lambda row: row["gain_mean"], reverse=True)
    metrics = {
        "protocol": {
            "test_used": False,
            "screen_seed": SCREEN_SEED,
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "configs": CONFIGS,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v36_random_forest_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v36_random_forest_oof.npz", **predictions
    )
    print("\n===== RandomForest 5-fold screen =====")
    for row in results[:6]:
        print(
            f"{row['variant']:18s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
