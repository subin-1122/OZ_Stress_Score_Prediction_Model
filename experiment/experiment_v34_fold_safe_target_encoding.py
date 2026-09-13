"""
Tier 1-4: fold-safe categorical target encoding ExtraTrees 5-fold 스크리닝
========================================================================

범주별 잔차를 사후에 더하는 대신, smoothed target encoding을 숫자형 raw 입력으로
ExtraTrees에 추가한다. 학습행 encoding도 inner OOF로 만들어 자기 target이 자기
입력값에 직접 포함되지 않게 한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- outer-valid encoding은 outer-train의 target만 사용한다.
- outer-train encoding은 다시 5-fold로 나눈 inner-train target만 사용한다.
- 결측치 처리와 ExtraTrees 학습도 outer-train으로만 수행한다.
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
SCREEN_SEED = 4444
OUTER_SPLITS = 5
INNER_SPLITS = 5
N_ESTIMATORS = 500
TRIM_RATIO = 0.10
SMOOTHING_VALUES = (5.0, 20.0, 100.0)
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


def normalized_category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__")


def fit_mapping(
    category: pd.Series,
    target: np.ndarray,
    smoothing: float,
) -> tuple[dict[str, float], float]:
    """한 inner-train의 범주별 smoothed target 평균을 학습한다."""
    global_mean = float(np.mean(target))
    frame = pd.DataFrame(
        {
            "category": normalized_category(category).to_numpy(),
            "target": target,
        }
    )
    grouped = frame.groupby("category", dropna=False)["target"].agg(["sum", "count"])
    encoded = (grouped["sum"] + smoothing * global_mean) / (
        grouped["count"] + smoothing
    )
    return {str(key): float(value) for key, value in encoded.items()}, global_mean


def apply_mapping(
    category: pd.Series,
    mapping: dict[str, float],
    fallback: float,
) -> np.ndarray:
    values = normalized_category(category).map(mapping).fillna(fallback)
    return values.to_numpy(float)


def make_target_encoded_frames(
    raw_features: pd.DataFrame,
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_valid: np.ndarray,
    categorical: list[str],
    smoothing: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """outer-train은 inner OOF, outer-valid는 full outer-train mapping을 쓴다."""
    encoded_train = pd.DataFrame(index=outer_train)
    encoded_valid = pd.DataFrame(index=outer_valid)
    inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)

    for column in categorical:
        train_values = np.zeros(len(outer_train), dtype=float)
        for inner_fit_position, inner_valid_position in inner.split(outer_train):
            inner_fit = outer_train[inner_fit_position]
            inner_valid = outer_train[inner_valid_position]
            mapping, fallback = fit_mapping(
                raw_features.iloc[inner_fit][column],
                y[inner_fit],
                smoothing,
            )
            train_values[inner_valid_position] = apply_mapping(
                raw_features.iloc[inner_valid][column],
                mapping,
                fallback,
            )
        full_mapping, full_fallback = fit_mapping(
            raw_features.iloc[outer_train][column],
            y[outer_train],
            smoothing,
        )
        encoded_train[f"te_{column}"] = train_values
        encoded_valid[f"te_{column}"] = apply_mapping(
            raw_features.iloc[outer_valid][column],
            full_mapping,
            full_fallback,
        )
    return encoded_train.reset_index(drop=True), encoded_valid.reset_index(drop=True)


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v34",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v34",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v34",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    raw_features = train.drop(columns=[ID_COLUMN, TARGET])
    model_features = v3.make_model_features(train.drop(columns=[TARGET]))
    categorical = raw_features.select_dtypes(exclude=np.number).columns.tolist()
    contexts = v24.make_link_contexts(v4, train, y, stored)

    predictions = {
        f"smooth_{int(smoothing)}": np.zeros(len(train))
        for smoothing in SMOOTHING_VALUES
    }
    fold_rows = []
    splitter = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    started = time.perf_counter()

    for fold, (outer_train, outer_valid) in enumerate(splitter.split(train), start=1):
        for smoothing in SMOOTHING_VALUES:
            encoded_train, encoded_valid = make_target_encoded_frames(
                raw_features,
                y,
                outer_train,
                outer_valid,
                categorical,
                smoothing,
                SCREEN_SEED * 100 + fold,
            )
            x_train = pd.concat(
                [
                    model_features.iloc[outer_train].reset_index(drop=True),
                    encoded_train,
                ],
                axis=1,
            )
            x_valid = pd.concat(
                [
                    model_features.iloc[outer_valid].reset_index(drop=True),
                    encoded_valid,
                ],
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
            model.fit(transformed_train, y[outer_train])
            predictions[f"smooth_{int(smoothing)}"][outer_valid] = (
                v3.trimmed_tree_prediction(model, transformed_valid, TRIM_RATIO)
            )
        fold_rows.append(
            {
                "fold": fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_valid_rows": int(len(outer_valid)),
                "target_encoded_columns": len(categorical),
            }
        )
        print(
            f"[target-encoding] fold={fold}/{OUTER_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for variant_name, alternative in predictions.items():
        for blend_weight in BLEND_WEIGHTS:
            raw = (1.0 - blend_weight) * baseline + blend_weight * alternative
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
            "outer_splits": OUTER_SPLITS,
            "inner_splits_for_train_encoding": INNER_SPLITS,
            "smoothing_values": SMOOTHING_VALUES,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "categorical_columns": categorical,
        "folds": fold_rows,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v34_target_encoding_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v34_target_encoding_oof.npz",
        **predictions,
    )

    print("\n===== Fold-safe target encoding screen =====")
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
