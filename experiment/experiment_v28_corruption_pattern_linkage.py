"""
중복쌍의 corruption pattern likelihood 연결
============================================

목적
----
고신뢰 train 중복쌍에서 각 열이 그대로 유지되거나 변형되는 패턴을 학습한다.
일반적인 유사도나 같은-target 이진 분류기 대신 다음 생성 확률비를 사용한다.

    log P(pair pattern | duplicate) - log P(pair pattern | hard negative)

열별 상태 확률과 16개 열의 공동 equality-mask 확률을 합쳐 확장 후보를
점수화한다. 이 점수가 충분히 높은 행만 현재 최종 예측에 추가 연결한다.

누수 방지
---------
- test.csv는 읽지 않는다.
- positive seed와 negative는 outer/inner fold의 train-train 쌍으로만 만든다.
- 연결 threshold는 outer-train 내부 calibration 행에서만 선택한다.
- outer validation의 정답은 최종 평가에만 사용한다.
- 기존 최종 파이프라인을 기준선으로 두고 새 연결만 추가한다.

실행
----
    .venv/bin/python experiment/experiment_v28_corruption_pattern_linkage.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"

OUTER_SPLITS = 10
CALIBRATION_SPLITS = 5
META_SEEDS = (11, 101, 1001, 2026, 31415)
MIN_SEED_TOTAL_EQUAL = 10
NEGATIVE_RATIO = 8
JOINT_WEIGHT = 0.35
LAPLACE = 1.0
JOINT_LAPLACE = 0.5
MIN_CALIBRATION_CHANGES = 5
THRESHOLD_QUANTILES = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
TARGET_OOF_GAIN = 0.0022


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_scales(
    features: pd.DataFrame,
    indices: np.ndarray,
    numeric: list[str],
) -> dict[str, float]:
    scales = {}
    fold = features.iloc[indices]
    for column in numeric:
        values = fold[column].to_numpy(float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        scales[column] = float(mad) if np.isfinite(mad) and mad > 0 else 1.0
    return scales


def pair_states(
    features: pd.DataFrame,
    pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    scales: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    각 열의 pair 상태, equality mask, 동일 열 개수를 만든다.

    숫자형 상태:
      0 둘 다 결측, 1 한쪽만 결측, 2 정확히 같음,
      3 매우 가까움, 4 가까움, 5 보통 차이, 6 큰 차이
    범주형 상태:
      0 둘 다 결측, 1 한쪽만 결측, 2 같은 관측값, 3 다른 관측값
    """
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    columns = []
    equal_columns = []
    cardinalities = []

    for column in numeric:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        a_missing = np.isnan(a)
        b_missing = np.isnan(b)
        both_missing = a_missing & b_missing
        one_missing = a_missing ^ b_missing
        observed = ~(a_missing | b_missing)
        equal = observed & (a == b)
        difference = np.zeros(len(a), dtype=float)
        difference[observed] = (
            np.abs(a[observed] - b[observed]) / scales[column]
        )
        state = np.full(len(a), 6, dtype=np.uint8)
        state[both_missing] = 0
        state[one_missing] = 1
        state[equal] = 2
        state[observed & ~equal & (difference <= 0.05)] = 3
        state[observed & (difference > 0.05) & (difference <= 0.25)] = 4
        state[observed & (difference > 0.25) & (difference <= 1.00)] = 5
        columns.append(state)
        equal_columns.append(both_missing | equal)
        cardinalities.append(7)

    for column in categorical:
        a = left[column].astype("string")
        b = right[column].astype("string")
        a_missing = a.isna().to_numpy()
        b_missing = b.isna().to_numpy()
        both_missing = a_missing & b_missing
        one_missing = a_missing ^ b_missing
        a_value = a.fillna("__MISSING__").to_numpy()
        b_value = b.fillna("__MISSING__").to_numpy()
        equal = ~a_missing & ~b_missing & (a_value == b_value)
        state = np.full(len(a), 3, dtype=np.uint8)
        state[both_missing] = 0
        state[one_missing] = 1
        state[equal] = 2
        columns.append(state)
        equal_columns.append(both_missing | equal)
        cardinalities.append(4)

    state_matrix = np.column_stack(columns)
    equal_matrix = np.column_stack(equal_columns)
    return (
        state_matrix,
        np.asarray(cardinalities, dtype=np.int16),
        equal_matrix.sum(axis=1).astype(np.int16),
    )


def equality_code(states: np.ndarray, numeric_count: int) -> np.ndarray:
    """16개 열의 equality 여부를 uint32 bit mask로 바꾼다."""
    equality = np.zeros_like(states, dtype=bool)
    equality[:, :numeric_count] = (
        (states[:, :numeric_count] == 0)
        | (states[:, :numeric_count] == 2)
    )
    equality[:, numeric_count:] = (
        (states[:, numeric_count:] == 0)
        | (states[:, numeric_count:] == 2)
    )
    powers = (1 << np.arange(states.shape[1], dtype=np.uint32))[None, :]
    return np.sum(equality.astype(np.uint32) * powers, axis=1, dtype=np.uint32)


def fit_likelihood(
    features: pd.DataFrame,
    y: np.ndarray,
    pairs: np.ndarray,
    fit_indices: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> dict:
    """fit 내부의 고신뢰 positive seed와 hard negative로 확률비를 학습한다."""
    in_fit = np.zeros(len(features), dtype=bool)
    in_fit[fit_indices] = True
    fit_pairs = pairs[in_fit[pairs[:, 0]] & in_fit[pairs[:, 1]]]
    scales = robust_scales(features, fit_indices, numeric)
    states, cardinalities, total_equal = pair_states(
        features,
        fit_pairs,
        numeric,
        categorical,
        scales,
    )
    same_target = y[fit_pairs[:, 0]] == y[fit_pairs[:, 1]]
    positive = np.flatnonzero(
        same_target & (total_equal >= MIN_SEED_TOTAL_EQUAL)
    )
    negative = np.flatnonzero(~same_target)
    if len(positive) < 20:
        raise RuntimeError("corruption likelihood를 학습할 positive seed가 부족합니다.")

    # 가장 비슷해서 구분하기 어려운 negative를 우선 사용한다.
    negative_order = negative[
        np.argsort(total_equal[negative], kind="stable")[::-1]
    ]
    negative = negative_order[: min(len(negative_order), NEGATIVE_RATIO * len(positive))]

    feature_log_ratio = []
    for column, cardinality in enumerate(cardinalities):
        positive_count = np.bincount(
            states[positive, column],
            minlength=int(cardinality),
        ).astype(float)
        negative_count = np.bincount(
            states[negative, column],
            minlength=int(cardinality),
        ).astype(float)
        positive_probability = (positive_count + LAPLACE) / (
            positive_count.sum() + LAPLACE * cardinality
        )
        negative_probability = (negative_count + LAPLACE) / (
            negative_count.sum() + LAPLACE * cardinality
        )
        feature_log_ratio.append(
            np.log(positive_probability / negative_probability)
        )

    codes = equality_code(states, len(numeric))
    n_codes = 1 << states.shape[1]
    positive_joint = np.bincount(codes[positive], minlength=n_codes).astype(float)
    negative_joint = np.bincount(codes[negative], minlength=n_codes).astype(float)
    positive_joint_probability = (positive_joint + JOINT_LAPLACE) / (
        positive_joint.sum() + JOINT_LAPLACE * n_codes
    )
    negative_joint_probability = (negative_joint + JOINT_LAPLACE) / (
        negative_joint.sum() + JOINT_LAPLACE * n_codes
    )
    return {
        "scales": scales,
        "feature_log_ratio": feature_log_ratio,
        "joint_log_ratio": np.log(
            positive_joint_probability / negative_joint_probability
        ),
        "positive_pairs": int(len(positive)),
        "negative_pairs": int(len(negative)),
    }


def score_pairs(
    model: dict,
    features: pd.DataFrame,
    pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> np.ndarray:
    states, _, _ = pair_states(
        features,
        pairs,
        numeric,
        categorical,
        model["scales"],
    )
    score = np.zeros(len(pairs), dtype=float)
    for column, lookup in enumerate(model["feature_log_ratio"]):
        score += lookup[states[:, column]]
    codes = equality_code(states, len(numeric))
    score += JOINT_WEIGHT * model["joint_log_ratio"][codes]
    return score


def orient_cross_pairs(
    pairs: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    train_mask = np.zeros(n_rows, dtype=bool)
    query_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_indices] = True
    query_mask[query_indices] = True
    cross = (
        (query_mask[pairs[:, 0]] & train_mask[pairs[:, 1]])
        | (query_mask[pairs[:, 1]] & train_mask[pairs[:, 0]])
    )
    result = pairs[cross].copy()
    reverse = train_mask[result[:, 0]]
    result[reverse] = result[reverse][:, ::-1]
    return result


def aggregate_candidates(
    pairs: np.ndarray,
    pair_score: np.ndarray,
    y: np.ndarray,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """후보 target별 최대 likelihood와 support를 합쳐 query별 1위를 고른다."""
    result = {
        "label": np.full(n_rows, np.nan),
        "score": np.full(n_rows, -np.inf),
        "margin": np.zeros(n_rows),
        "support": np.zeros(n_rows, dtype=np.int16),
    }
    if len(pairs) == 0:
        return result

    frame = pd.DataFrame(
        {
            "row": pairs[:, 0],
            "label": y[pairs[:, 1]],
            "pair_score": pair_score,
        }
    )
    grouped = (
        frame.groupby(["row", "label"], sort=False)["pair_score"]
        .agg(["max", "size"])
        .reset_index()
    )
    grouped["group_score"] = grouped["max"] + 0.10 * np.log1p(grouped["size"])
    grouped = grouped.sort_values(
        ["row", "group_score"],
        ascending=[True, False],
        kind="stable",
    )
    for row, candidates in grouped.groupby("row", sort=False):
        first = candidates.iloc[0]
        result["label"][int(row)] = float(first["label"])
        result["score"][int(row)] = float(first["group_score"])
        result["support"][int(row)] = int(first["size"])
        if len(candidates) > 1:
            result["margin"][int(row)] = float(
                first["group_score"] - candidates.iloc[1]["group_score"]
            )
    return result


def fit_and_predict(
    features: pd.DataFrame,
    y: np.ndarray,
    pairs: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> tuple[dict[str, np.ndarray], dict]:
    model = fit_likelihood(
        features,
        y,
        pairs,
        fit_indices,
        numeric,
        categorical,
    )
    query_pairs = orient_cross_pairs(
        pairs,
        fit_indices,
        query_indices,
        len(features),
    )
    scores = score_pairs(
        model,
        features,
        query_pairs,
        numeric,
        categorical,
    )
    return (
        aggregate_candidates(query_pairs, scores, y, len(features)),
        {
            "positive_pairs": model["positive_pairs"],
            "negative_pairs": model["negative_pairs"],
            "query_pairs": int(len(query_pairs)),
        },
    )


def make_fold_cache(
    features: pd.DataFrame,
    y: np.ndarray,
    pairs: np.ndarray,
    seed: int,
) -> list[dict]:
    """
    meta seed 하나의 calibration/outer-validation likelihood 결과를 만든다.

    calibration은 outer train의 20%이며, validation 모델은 outer train 전체로
    다시 학습한다. 두 단계 모두 query target은 likelihood 학습에 들어가지 않는다.
    """
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    cache = []
    outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    for fold, (outer_train, outer_valid) in enumerate(
        outer.split(features), start=1
    ):
        calibration_splitter = KFold(
            n_splits=CALIBRATION_SPLITS,
            shuffle=True,
            random_state=seed + fold,
        )
        inner_fit_pos, calibration_pos = next(
            calibration_splitter.split(outer_train)
        )
        inner_fit = outer_train[inner_fit_pos]
        calibration = outer_train[calibration_pos]
        calibration_result, calibration_stats = fit_and_predict(
            features,
            y,
            pairs,
            inner_fit,
            calibration,
            numeric,
            categorical,
        )
        validation_result, validation_stats = fit_and_predict(
            features,
            y,
            pairs,
            outer_train,
            outer_valid,
            numeric,
            categorical,
        )
        cache.append(
            {
                "fold": fold,
                "calibration_indices": calibration,
                "validation_indices": outer_valid,
                "calibration_result": calibration_result,
                "validation_result": validation_result,
                "calibration_stats": calibration_stats,
                "validation_stats": validation_stats,
            }
        )
    return cache


def select_threshold(
    calibration_indices: np.ndarray,
    result: dict[str, np.ndarray],
    current: np.ndarray,
    y: np.ndarray,
    linked: np.ndarray,
) -> dict:
    """calibration 정답으로 no-op을 포함한 likelihood threshold를 선택한다."""
    eligible = calibration_indices[
        (~linked[calibration_indices])
        & np.isfinite(result["label"][calibration_indices])
        & np.isfinite(result["score"][calibration_indices])
    ]
    baseline = mean_absolute_error(y[calibration_indices], current[calibration_indices])
    best = {
        "threshold": None,
        "calibration_mae": float(baseline),
        "calibration_gain": 0.0,
        "calibration_changed_rows": 0,
    }
    if len(eligible) == 0:
        return best

    thresholds = np.unique(
        np.quantile(result["score"][eligible], THRESHOLD_QUANTILES)
    )
    for threshold in thresholds:
        use = eligible[result["score"][eligible] >= threshold]
        if len(use) < MIN_CALIBRATION_CHANGES:
            continue
        candidate = current[calibration_indices].copy()
        positions = pd.Series(
            np.arange(len(calibration_indices)),
            index=calibration_indices,
        ).loc[use].to_numpy()
        candidate[positions] = result["label"][use]
        score = mean_absolute_error(y[calibration_indices], candidate)
        gain = baseline - score
        old_threshold = (
            best["threshold"] if best["threshold"] is not None else np.inf
        )
        # 동률이면 더 높은 threshold를 택해 연결 수를 줄인다.
        if (gain, threshold) > (best["calibration_gain"], old_threshold):
            best = {
                "threshold": float(threshold),
                "calibration_mae": float(score),
                "calibration_gain": float(gain),
                "calibration_changed_rows": int(len(use)),
            }
    return best


def evaluate_context(
    cache: list[dict],
    context: dict,
    y: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    current = context["current"]
    linked = context["mask"]
    candidate = current.copy()
    choices = []
    for fold_data in cache:
        calibration = fold_data["calibration_indices"]
        valid = fold_data["validation_indices"]
        calibration_result = fold_data["calibration_result"]
        validation_result = fold_data["validation_result"]
        choice = select_threshold(
            calibration,
            calibration_result,
            current,
            y,
            linked,
        )
        changed = np.empty(0, dtype=int)
        if choice["threshold"] is not None and choice["calibration_gain"] > 0:
            changed = valid[
                (~linked[valid])
                & np.isfinite(validation_result["label"][valid])
                & (
                    validation_result["score"][valid]
                    >= choice["threshold"]
                )
            ]
            candidate[changed] = validation_result["label"][changed]
        choice.update(
            {
                "fold": fold_data["fold"],
                "outer_changed_rows": int(len(changed)),
                "calibration_stats": fold_data["calibration_stats"],
                "validation_stats": fold_data["validation_stats"],
            }
        )
        choices.append(choice)
    return candidate, choices


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


def main() -> None:
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = load_module(
        "experiment_v24",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    v25 = load_module(
        "experiment_v25",
        ROOT / "experiment/experiment_v25_mae_aware_linkage.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    features = train.drop(columns=[ID_COLUMN, TARGET])
    y = train[TARGET].to_numpy(float)
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pair_pools = v25.expanded_block_pairs(v3, features, categorical)
    pairs = pair_pools["expanded"]
    contexts = v24.make_link_contexts(v4, train, y, stored)

    rows = []
    for meta_seed in META_SEEDS:
        print(f"[cache] meta seed={meta_seed}", flush=True)
        cache = make_fold_cache(features, y, pairs, int(meta_seed))
        for context in contexts:
            candidate, choices = evaluate_context(cache, context, y)
            current_mae = mean_absolute_error(y, context["current"])
            candidate_mae = mean_absolute_error(y, candidate)
            row = {
                "meta_seed": int(meta_seed),
                "link_seed": context["seed"],
                "current_oof_mae": float(current_mae),
                "candidate_oof_mae": float(candidate_mae),
                "gain_vs_current": float(current_mae - candidate_mae),
                "changed_rows": int(np.sum(candidate != context["current"])),
                "fold_choices": choices,
            }
            rows.append(row)
            print(
                f"  link={context['seed']:5d} OOF={candidate_mae:.9f} "
                f"gain={current_mae-candidate_mae:+.9f} "
                f"changed={row['changed_rows']}",
                flush=True,
            )

    gains = np.asarray([row["gain_vs_current"] for row in rows])
    scores = np.asarray([row["candidate_oof_mae"] for row in rows])
    changes = np.asarray([row["changed_rows"] for row in rows])
    selected_fold_count = sum(
        choice["threshold"] is not None and choice["calibration_gain"] > 0
        for row in rows
        for choice in row["fold_choices"]
    )
    all_improve = bool(np.all(gains > 0))
    mean_gain = float(gains.mean())
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and all_improve)
    metrics = {
        "protocol": {
            "test_used": False,
            "candidate_pairs": int(len(pairs)),
            "original_pairs": int(len(pair_pools["original"])),
            "cat4_pairs": int(len(pair_pools["cat4"])),
            "positive_seed_min_total_equal": MIN_SEED_TOTAL_EQUAL,
            "negative_ratio": NEGATIVE_RATIO,
            "joint_weight": JOINT_WEIGHT,
            "outer_splits": OUTER_SPLITS,
            "calibration_splits": CALIBRATION_SPLITS,
            "meta_seeds": META_SEEDS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "results": rows,
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "changed_rows_mean": float(changes.mean()),
            "selected_outer_folds": int(selected_fold_count),
            "all_repeats_improve": all_improve,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
        },
    }
    output_path = (
        ROOT / "outputs/experiment_v28_corruption_pattern_linkage_metrics.json"
    )
    output_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Corruption-pattern likelihood =====")
    print(
        f"OOF={scores.mean():.9f} gain={mean_gain:+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/"
        f"{(gains < 0).sum()} changed 평균={changes.mean():.1f}"
    )
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
