"""
Tier 1-2: unlinked 전용 Gaussian Process Regression 5-fold 스크리닝
===================================================================

현재 ExtraTrees와 다른 커널 기반 국소 스무딩이 미연결 행의 오차를 줄이는지
확인한다. 기존 연결행은 최종 단계에서 원래 연결값으로 덮어쓰므로, GPR은 현재
연결 규칙에서 미연결로 분류된 fold-train 행만 학습한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 전처리기와 GPR은 각 fold-train의 미연결 행으로만 fit한다.
- kernel과 blend 비율은 실행 전에 고정했다.
- validation target은 OOF 평가에만 사용한다.
- 5-fold 스크리닝에서 목표 0.0022와 연결 seed 5/5 개선을 만족할 때만
  100-fold 확인 대상으로 간주한다. 이 파일은 제출 파일을 만들지 않는다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 4242
N_SPLITS = 5
TARGET_OOF_GAIN = 0.0022
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 1.00)

KERNELS = {
    "rbf_l2_n005": ConstantKernel(1.0, constant_value_bounds="fixed")
    * RBF(2.0, length_scale_bounds="fixed")
    + WhiteKernel(0.05, noise_level_bounds="fixed"),
    "rbf_l4_n005": ConstantKernel(1.0, constant_value_bounds="fixed")
    * RBF(4.0, length_scale_bounds="fixed")
    + WhiteKernel(0.05, noise_level_bounds="fixed"),
    "matern_l4_n005": ConstantKernel(1.0, constant_value_bounds="fixed")
    * Matern(4.0, length_scale_bounds="fixed", nu=1.5)
    + WhiteKernel(0.05, noise_level_bounds="fixed"),
    "matern_l8_n010": ConstantKernel(1.0, constant_value_bounds="fixed")
    * Matern(8.0, length_scale_bounds="fixed", nu=2.5)
    + WhiteKernel(0.10, noise_level_bounds="fixed"),
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
        "experiment_v1_for_v32",
        ROOT / "experiment/experiment_v1_models.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v32",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v32",
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

    # 첫 연결 seed의 mask를 GPR 학습 대상 정의에만 사용한다. 다른 네 mask에서도
    # 최종 OOF를 별도로 평가해 이 선택에만 맞춘 결과인지 확인한다.
    canonical_unlinked = ~contexts[0]["mask"]
    predictions = {name: np.zeros(len(train)) for name in KERNELS}
    uncertainties = {name: np.zeros(len(train)) for name in KERNELS}
    fold_rows = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()

    for fold, (outer_train, valid_idx) in enumerate(splitter.split(train), start=1):
        fit_idx = outer_train[canonical_unlinked[outer_train]]
        preprocessor = v1.make_preprocessor(
            numeric,
            categorical,
            scale_numeric=True,
        )
        x_fit = preprocessor.fit_transform(features.iloc[fit_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        for name, kernel in KERNELS.items():
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                optimizer=None,
                normalize_y=True,
                random_state=SCREEN_SEED + fold,
                copy_X_train=False,
            )
            model.fit(x_fit, y[fit_idx])
            mean, std = model.predict(x_valid, return_std=True)
            predictions[name][valid_idx] = np.clip(mean, 0.0, 1.0)
            uncertainties[name][valid_idx] = std
        fold_rows.append(
            {
                "fold": fold,
                "fit_unlinked_rows": int(len(fit_idx)),
                "valid_rows": int(len(valid_idx)),
                "transformed_features": int(x_fit.shape[1]),
            }
        )
        print(
            f"[GPR] fold={fold}/{N_SPLITS} fit={len(fit_idx)} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for kernel_name, gpr_prediction in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = baseline.copy()
            raw[canonical_unlinked] = (
                (1.0 - blend_weight) * baseline[canonical_unlinked]
                + blend_weight * gpr_prediction[canonical_unlinked]
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
                    "kernel": kernel_name,
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
    best = results[0]
    metrics = {
        "protocol": {
            "test_used": False,
            "screen_seed": SCREEN_SEED,
            "n_splits": N_SPLITS,
            "kernels": {name: str(kernel) for name, kernel in KERNELS.items()},
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "canonical_unlinked_rows": int(canonical_unlinked.sum()),
        "folds": fold_rows,
        "results": results,
        "best": best,
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v32_gaussian_process_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v32_gaussian_process_oof.npz",
        **{f"prediction_{name}": values for name, values in predictions.items()},
        **{f"uncertainty_{name}": values for name, values in uncertainties.items()},
    )

    print("\n===== Gaussian Process 5-fold screen =====")
    for row in results[:6]:
        print(
            f"{row['kernel']:18s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
