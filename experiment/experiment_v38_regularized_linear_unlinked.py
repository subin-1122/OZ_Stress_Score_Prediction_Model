"""Tier 2-3: unlinked 전용 정규화 선형모델 5-fold 스크리닝."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 4848
N_SPLITS = 5
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 1.00)
TARGET_OOF_GAIN = 0.0022
MODELS = {
    "bayesian_ridge": lambda: BayesianRidge(),
    "ridge_10": lambda: Ridge(alpha=10.0),
    "ridge_100": lambda: Ridge(alpha=100.0),
    "lasso_0p001": lambda: Lasso(alpha=0.001, max_iter=20_000),
    "lasso_0p01": lambda: Lasso(alpha=0.01, max_iter=20_000),
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
    v1 = v31.load_module(
        "experiment_v1_for_v38", ROOT / "experiment/experiment_v1_models.py"
    )
    v4 = v31.load_module(
        "experiment_v4_for_v38",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v38",
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
    canonical_unlinked = ~contexts[0]["mask"]

    predictions = {name: np.zeros(len(train)) for name in MODELS}
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (outer_train, valid_idx) in enumerate(splitter.split(train), start=1):
        fit_idx = outer_train[canonical_unlinked[outer_train]]
        preprocessor = v1.make_preprocessor(
            numeric, categorical, scale_numeric=True
        )
        x_train = preprocessor.fit_transform(features.iloc[fit_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        for name, factory in MODELS.items():
            model = factory()
            model.fit(x_train, y[fit_idx])
            predictions[name][valid_idx] = np.clip(
                model.predict(x_valid), 0.0, 1.0
            )
        fold_rows.append(
            {"fold": fold, "fit_unlinked_rows": len(fit_idx), "valid_rows": len(valid_idx)}
        )
        print(
            f"[linear] fold={fold}/{N_SPLITS} fit={len(fit_idx)} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for model_name, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = baseline.copy()
            raw[canonical_unlinked] = (
                (1.0 - blend_weight) * baseline[canonical_unlinked]
                + blend_weight * alternative[canonical_unlinked]
            )
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
                    "model": model_name,
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
            "models": list(MODELS),
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "canonical_unlinked_rows": int(canonical_unlinked.sum()),
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v38_regularized_linear_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v38_regularized_linear_oof.npz",
        **predictions,
    )
    print("\n===== Regularized linear unlinked-only screen =====")
    for row in results[:7]:
        print(
            f"{row['model']:15s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
