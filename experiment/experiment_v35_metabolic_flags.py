"""
Tier 1-5: 대사 위험 복합 이진 플래그 ExtraTrees 5-fold 스크리닝
================================================================

혈당, 콜레스테롤, 혈압, BMI가 동시에 높은 패턴을 연속 비율이 아닌 이산
상호작용으로 추가한다. 외부 임상 기준을 사용하지 않고 각 fold-train의
60/70/80% 분위수를 임계값으로 사용한다.

test.csv를 읽지 않으며 임계값, 결측치 처리와 모델은 fold-train으로만 학습한다.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SCREEN_SEED = 4545
N_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
RISK_QUANTILES = (0.60, 0.70, 0.80)
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


def risk_values(raw: pd.DataFrame) -> pd.DataFrame:
    """한 행 안의 값만 이용해 네 가지 위험 축을 만든다."""
    height_m = pd.to_numeric(raw["height"], errors="coerce") / 100.0
    bmi = pd.to_numeric(raw["weight"], errors="coerce") / height_m.where(
        height_m > 0
    ).pow(2)
    # 두 혈압 중 어느 하나가 높을 때를 표현하도록 fold-train 안에서 각각
    # 표준화하기 전에는 원본 두 열을 그대로 보존한다.
    return pd.DataFrame(
        {
            "glucose_risk": pd.to_numeric(raw["glucose"], errors="coerce"),
            "cholesterol_risk": pd.to_numeric(
                raw["cholesterol"], errors="coerce"
            ),
            "systolic_risk": pd.to_numeric(
                raw["systolic_blood_pressure"], errors="coerce"
            ),
            "diastolic_risk": pd.to_numeric(
                raw["diastolic_blood_pressure"], errors="coerce"
            ),
            "bmi_risk": bmi,
        },
        index=raw.index,
    )


def make_flags(
    risks: pd.DataFrame,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    quantile: float,
) -> pd.DataFrame:
    """fold-train 분위 임계값을 query에 적용해 복합 플래그를 만든다."""
    fit = risks.iloc[fit_indices]
    query = risks.iloc[query_indices]
    thresholds = fit.quantile(quantile)
    high = pd.DataFrame(index=query.index)
    high["glucose"] = query["glucose_risk"] >= thresholds["glucose_risk"]
    high["cholesterol"] = (
        query["cholesterol_risk"] >= thresholds["cholesterol_risk"]
    )
    high["blood_pressure"] = (
        (query["systolic_risk"] >= thresholds["systolic_risk"])
        | (query["diastolic_risk"] >= thresholds["diastolic_risk"])
    )
    high["bmi"] = query["bmi_risk"] >= thresholds["bmi_risk"]
    high = high.fillna(False).astype(np.int8)

    flags = pd.DataFrame(index=query.index)
    flags["metabolic_risk_count"] = high.sum(axis=1).astype(np.int8)
    flags["metabolic_risk_ge2"] = (flags["metabolic_risk_count"] >= 2).astype(
        np.int8
    )
    flags["metabolic_risk_ge3"] = (flags["metabolic_risk_count"] >= 3).astype(
        np.int8
    )
    flags["metabolic_risk_all4"] = (flags["metabolic_risk_count"] == 4).astype(
        np.int8
    )
    for left, right in combinations(high.columns, 2):
        flags[f"risk_{left}_and_{right}"] = (high[left] & high[right]).astype(
            np.int8
        )
    return flags.reset_index(drop=True)


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v35",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v35",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v35",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    raw = train.drop(columns=[ID_COLUMN, TARGET])
    risks = risk_values(raw)
    base_features = v3.make_model_features(train.drop(columns=[TARGET]))
    contexts = v24.make_link_contexts(v4, train, y, stored)

    predictions = {
        f"quantile_{int(100 * quantile)}": np.zeros(len(train))
        for quantile in RISK_QUANTILES
    }
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        for quantile in RISK_QUANTILES:
            train_flags = make_flags(risks, train_idx, train_idx, quantile)
            valid_flags = make_flags(risks, train_idx, valid_idx, quantile)
            x_train = pd.concat(
                [base_features.iloc[train_idx].reset_index(drop=True), train_flags],
                axis=1,
            )
            x_valid = pd.concat(
                [base_features.iloc[valid_idx].reset_index(drop=True), valid_flags],
                axis=1,
            )
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
            predictions[f"quantile_{int(100 * quantile)}"][valid_idx] = (
                v3.trimmed_tree_prediction(model, transformed_valid, TRIM_RATIO)
            )
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
            }
        )
        print(
            f"[metabolic] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for variant, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            blended = (1.0 - blend_weight) * baseline + blend_weight * alternative
            context_rows = []
            for context in contexts:
                candidate = v24.finalize(
                    v4,
                    train,
                    y,
                    blended,
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
            "risk_quantiles": RISK_QUANTILES,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v35_metabolic_flags_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v35_metabolic_flags_oof.npz",
        **predictions,
    )

    print("\n===== Metabolic flag screen =====")
    for row in results[:6]:
        print(
            f"{row['variant']:12s} blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
