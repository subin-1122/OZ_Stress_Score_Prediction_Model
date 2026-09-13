"""
Tier 1-3: v2 변수 조합 A/B의 unlinked 전용 5-fold 재평가
=========================================================

v2의 A/B는 전체 train 기준으로 비교했다. 이번에는 현재 연결 규칙에서 미연결인
행만 학습하는 ExtraTrees로 다시 비교하고, 현재 raw 예측에 일부 혼합한 뒤 기존
연결 + mean_working + 0.01 snapping까지 적용한다.

test.csv를 읽지 않으며 전처리와 모델은 fold-train의 미연결 행으로만 학습한다.
"""

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
SCREEN_SEED = 4343
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
TARGET_OOF_GAIN = 0.0022
BLEND_WEIGHTS = (0.25, 0.50, 0.75, 1.00)


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
    v2 = v31.load_module(
        "experiment_v2_for_v33",
        ROOT / "experiment/experiment_v2_extratrees.py",
    )
    v3 = v31.load_module(
        "experiment_v3_for_v33",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v33",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v33",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    contexts = v24.make_link_contexts(v4, train, y, stored)
    canonical_unlinked = ~contexts[0]["mask"]
    base_frame = train.drop(columns=[TARGET])
    features = {
        name: v2.make_variant_features(base_frame, drop_columns)
        for name, drop_columns in v2.VARIANTS.items()
    }
    predictions = {name: np.zeros(len(train)) for name in features}
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    folds = list(splitter.split(train))
    started = time.perf_counter()
    fold_rows = []

    for fold, (outer_train, valid_idx) in enumerate(folds, start=1):
        fit_idx = outer_train[canonical_unlinked[outer_train]]
        for variant_name, frame in features.items():
            preprocessor = v2.build_preprocessor(frame.iloc[fit_idx])
            x_fit = preprocessor.fit_transform(frame.iloc[fit_idx])
            x_valid = preprocessor.transform(frame.iloc[valid_idx])
            model = ExtraTreesRegressor(
                n_estimators=N_ESTIMATORS,
                criterion="squared_error",
                max_features=1,
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                random_state=SCREEN_SEED * 100 + fold,
            )
            model.fit(x_fit, y[fit_idx])
            predictions[variant_name][valid_idx] = v3.trimmed_tree_prediction(
                model,
                x_valid,
                TRIM_RATIO,
            )
        fold_rows.append(
            {
                "fold": fold,
                "fit_unlinked_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
        print(
            f"[v2-unlinked] fold={fold}/{N_SPLITS} fit={len(fit_idx)} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    predictions["A50_B50"] = 0.5 * (
        predictions["A_drop_work_age_sleep"]
        + predictions["B_drop_work_age_smoke_dia"]
    )
    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for variant_name, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = baseline.copy()
            raw[canonical_unlinked] = (
                (1.0 - blend_weight) * baseline[canonical_unlinked]
                + blend_weight * alternative[canonical_unlinked]
            )
            context_rows = []
            for context in contexts:
                candidate = v24.finalize(
                    v4,
                    train,
                    y,
                    raw,
                    context["mask"],
                    context["label"],
                )
                current_mae = mean_absolute_error(y, context["current"])
                candidate_mae = mean_absolute_error(y, candidate)
                context_rows.append(
                    {
                        "link_seed": int(context["seed"]),
                        "candidate_oof_mae": float(candidate_mae),
                        "gain_vs_current": float(current_mae - candidate_mae),
                    }
                )
            gains = np.asarray([row["gain_vs_current"] for row in context_rows])
            scores = np.asarray([row["candidate_oof_mae"] for row in context_rows])
            results.append(
                {
                    "variant": variant_name,
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
            "variants": v2.VARIANTS,
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
    path = ROOT / "outputs/experiment_v33_unlinked_v2_features_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v33_unlinked_v2_features_oof.npz",
        **predictions,
    )

    print("\n===== v2 feature sets unlinked-only screen =====")
    for row in results[:6]:
        print(
            f"{row['variant']:28s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
