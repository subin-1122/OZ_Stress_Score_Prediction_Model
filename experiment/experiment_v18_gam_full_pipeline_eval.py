"""
GAM 20% 블렌딩 100-fold 최종 파이프라인 평가
================================================

독립 검증을 통과한 한 후보만 기존 제출 파이프라인에 결합한다.
- ExtraTrees base 80%
- flexible GAM 20% (spline knots=7, quantile alpha=0.0003)
- 100-fold x 3 seeds
- learned/deterministic linkage + mean_working alpha=0.75 + 0.01 snapping

이 파일은 평가와 예측 cache만 저장하며 제출 CSV를 만들지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (11, 101, 1001)
N_SPLITS = 100
GAM_WEIGHT = 0.20
GAM_N_KNOTS = 7
GAM_ALPHA = 0.0003
WORK_ALPHA = 0.75
GRID_STEP = 0.01
MIN_GAIN = 0.0003


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def make_gam_oof_and_test(v16, train_features, test_features, y):
    oof_by_seed = []
    test_by_seed = []
    for seed in SEEDS:
        oof = np.zeros(len(train_features))
        test_sum = np.zeros(len(test_features))
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        started = time.perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_features), 1):
            preprocessor = v16.build_gam_preprocessor(
                train_features.iloc[train_idx], GAM_N_KNOTS
            )
            x_train = preprocessor.fit_transform(train_features.iloc[train_idx])
            x_valid = preprocessor.transform(train_features.iloc[valid_idx])
            x_test = preprocessor.transform(test_features)
            model = QuantileRegressor(
                quantile=0.5,
                alpha=GAM_ALPHA,
                fit_intercept=True,
                solver="highs",
            )
            model.fit(x_train, y[train_idx])
            oof[valid_idx] = np.clip(model.predict(x_valid), 0.0, 1.0)
            test_sum += np.clip(model.predict(x_test), 0.0, 1.0)
            if fold % 10 == 0 or fold == N_SPLITS:
                print(
                    f"[GAM full] seed={seed} fold={fold:03d}/{N_SPLITS} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        oof_by_seed.append(oof)
        test_by_seed.append(test_sum / N_SPLITS)
    return (
        np.mean(np.column_stack(oof_by_seed), axis=1),
        np.mean(np.column_stack(test_by_seed), axis=1),
    )


def main() -> None:
    v3 = load_module("v3", ROOT / "experiment/experiment_v3_best_record_linkage.py")
    v4 = load_module("v4", ROOT / "experiment/experiment_v4_deterministic_union.py")
    v8 = load_module("v8", ROOT / "experiment/experiment_v8_no_work_correction_diagnostic.py")
    v16 = load_module("v16", ROOT / "experiment/experiment_v16_gam_quantile.py")

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    test = pd.read_csv(ROOT / "open (3)/test.csv")
    stored_oof = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    current_v4 = pd.read_csv(ROOT / "outputs/best_union_submission.csv")
    y = train[TARGET].to_numpy(float)
    base_oof = stored_oof["base_oof"].to_numpy(float)
    learned_label = stored_oof["link_label"].to_numpy(float)
    learned_mask = stored_oof["linked"].astype(bool).to_numpy()
    train_features = train.drop(columns=[ID_COLUMN, TARGET])
    test_features = test.drop(columns=[ID_COLUMN])

    gam_oof, gam_test = make_gam_oof_and_test(
        v16, train_features, test_features, y
    )
    blended_oof = (1.0 - GAM_WEIGHT) * base_oof + GAM_WEIGHT * gam_oof

    numeric = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical = train_features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(train_features, numeric, categorical)
    repeated = []
    for seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(rule_pairs, y, len(train), seed)
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]

        old_correction = v4.crossfit_work_correction(
            train, y, base_oof, union_mask
        )
        old_prediction = snap(
            np.clip(base_oof + WORK_ALPHA * old_correction, 0.0, 1.0)
        )
        old_prediction[union_mask] = union_label[union_mask]

        new_correction = v4.crossfit_work_correction(
            train, y, blended_oof, union_mask
        )
        new_prediction = snap(
            np.clip(blended_oof + WORK_ALPHA * new_correction, 0.0, 1.0)
        )
        new_prediction[union_mask] = union_label[union_mask]

        old_mae = mean_absolute_error(y, old_prediction)
        new_mae = mean_absolute_error(y, new_prediction)
        repeated.append(
            {
                "seed": int(seed),
                "old_oof_mae": float(old_mae),
                "new_oof_mae": float(new_mae),
                "gain": float(old_mae - new_mae),
                "unlinked_old_mae": float(
                    mean_absolute_error(y[~union_mask], old_prediction[~union_mask])
                ),
                "unlinked_new_mae": float(
                    mean_absolute_error(y[~union_mask], new_prediction[~union_mask])
                ),
            }
        )

    # 기존 100-fold ExtraTrees test 예측을 정확히 복원한 뒤 GAM과 혼합한다.
    _, _, test_union_mask = v8.make_test_masks(v3, v4, train, test, y)
    base_test, _ = v8.recover_base_test_prediction(
        v3, train, test, stored_oof, current_v4, test_union_mask
    )
    blended_test = (1.0 - GAM_WEIGHT) * base_test + GAM_WEIGHT * gam_test

    gains = np.asarray([row["gain"] for row in repeated])
    passes = bool(gains.mean() >= MIN_GAIN and (gains > 0).all())
    metrics = {
        "candidate": {
            "gam_weight": GAM_WEIGHT,
            "gam_n_knots": GAM_N_KNOTS,
            "gam_alpha": GAM_ALPHA,
            "work_alpha": WORK_ALPHA,
            "grid_step": GRID_STEP,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "n_splits": N_SPLITS,
            "minimum_gain": MIN_GAIN,
            "test_used_for_model_selection": False,
        },
        "old_oof_mae_mean": float(np.mean([row["old_oof_mae"] for row in repeated])),
        "new_oof_mae_mean": float(np.mean([row["new_oof_mae"] for row in repeated])),
        "gain_mean": float(gains.mean()),
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "passes_submission_gate": passes,
        "repeated": repeated,
    }
    metrics_path = ROOT / "outputs/experiment_v18_gam_full_pipeline_metrics.json"
    cache_path = ROOT / "outputs/experiment_v18_gam_full_predictions.npz"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        cache_path,
        gam_oof=gam_oof,
        gam_test=gam_test,
        blended_oof=blended_oof,
        blended_test=blended_test,
        test_union_mask=test_union_mask,
    )

    print("\n===== 100-fold GAM 최종 파이프라인 OOF =====")
    for row in repeated:
        print(
            f"seed={row['seed']:5d} old={row['old_oof_mae']:.9f} "
            f"new={row['new_oof_mae']:.9f} gain={row['gain']:+.9f}"
        )
    print(
        f"평균 old={metrics['old_oof_mae_mean']:.9f} "
        f"new={metrics['new_oof_mae_mean']:.9f} "
        f"gain={metrics['gain_mean']:+.9f} "
        f"승/무/패={metrics['seed_wins']}/"
        f"{metrics['seed_ties']}/{metrics['seed_losses']} "
        f"통과={passes}"
    )
    print(f"metrics={metrics_path}")
    print(f"cache={cache_path}")


if __name__ == "__main__":
    main()
