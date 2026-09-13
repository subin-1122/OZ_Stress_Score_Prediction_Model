"""Tier 2-4: fold-train 연속형 winsorizing ExtraTrees 5-fold 스크리닝."""

from __future__ import annotations

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
SCREEN_SEED = 4949
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
CLIP_RATIOS = (0.01, 0.02, 0.05)
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)
TARGET_OOF_GAIN = 0.0022


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


def clip_fold(train_frame, valid_frame, numeric, ratio):
    """fold-train의 분위 경계만 사용해 두 frame을 클리핑한다."""
    clipped_train = train_frame.copy()
    clipped_valid = valid_frame.copy()
    bounds = {}
    for column in numeric:
        lower = float(train_frame[column].quantile(ratio))
        upper = float(train_frame[column].quantile(1.0 - ratio))
        clipped_train[column] = train_frame[column].clip(lower, upper)
        clipped_valid[column] = valid_frame[column].clip(lower, upper)
        bounds[column] = {"lower": lower, "upper": upper}
    return clipped_train, clipped_valid, bounds


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v39",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v39",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v39",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    raw = train.drop(columns=[TARGET])
    numeric = raw.drop(columns=[ID_COLUMN]).select_dtypes(include=np.number).columns.tolist()
    contexts = v24.make_link_contexts(v4, train, y, stored)

    predictions = {
        f"clip_{int(ratio * 100)}pct": np.zeros(len(train))
        for ratio in CLIP_RATIOS
    }
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        ratio_bounds = {}
        for ratio in CLIP_RATIOS:
            clipped_train, clipped_valid, bounds = clip_fold(
                raw.iloc[train_idx], raw.iloc[valid_idx], numeric, ratio
            )
            x_train = v3.make_model_features(clipped_train)
            x_valid = v3.make_model_features(clipped_valid)
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
                random_state=SCREEN_SEED * 100 + fold,
            )
            model.fit(transformed_train, y[train_idx])
            predictions[f"clip_{int(ratio * 100)}pct"][valid_idx] = (
                v3.trimmed_tree_prediction(model, transformed_valid, TRIM_RATIO)
            )
            ratio_bounds[str(ratio)] = bounds
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "bounds": ratio_bounds,
            }
        )
        print(
            f"[winsor] fold={fold}/{N_SPLITS} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for variant, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            blended = (1.0 - blend_weight) * baseline + blend_weight * alternative
            rows = []
            for context in contexts:
                candidate = v24.finalize(
                    v4, train, y, blended, context["mask"], context["label"]
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
                    "context_results": rows,
                }
            )
    results.sort(key=lambda row: row["gain_mean"], reverse=True)
    metrics = {
        "protocol": {
            "test_used": False,
            "screen_seed": SCREEN_SEED,
            "n_splits": N_SPLITS,
            "clip_ratios": CLIP_RATIOS,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v39_winsorizing_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v39_winsorizing_oof.npz", **predictions
    )
    print("\n===== Fold-safe winsorizing screen =====")
    for row in results[:6]:
        print(
            f"{row['variant']:10s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
