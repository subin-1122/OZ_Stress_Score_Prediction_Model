"""
MAE-aware 후보 순위 학습과 후보군 oracle 상한선 검사
====================================================

목적
----
기존 연결 분류기는 두 행의 stress_score가 정확히 같은지를 학습한다. 하지만
최종 평가지표는 MAE이므로 이 실험은 후보 점수를 복사했을 때 발생하는
abs(query_score - candidate_score)를 직접 예측한다.

검증 순서
---------
1. feature만 사용해 기존 blocking, 범주형 4개 blocking, Gower 최근접 후보를
   합친다.
2. 각 pair fold에서 validation 행을 완전히 제외하고 pair-risk 모델을 학습한다.
3. validation 후보 중 예상 절대오차가 가장 작은 후보를 고른다.
4. 별도의 nested gate가 현재 예측보다 후보 복사가 유리한 행만 선택한다.
5. 후보군 oracle 개선, 모델 top-1 개선, nested gate 개선을 따로 기록한다.

누수 방지
---------
- test.csv는 읽지 않는다.
- 후보 생성은 feature만 사용한다.
- pair-risk 모델의 target은 해당 fold의 train-train 쌍에서만 만든다.
- gate 분류기와 threshold/혼합 강도는 meta outer-fold 밖에서만 학습·선택한다.
- oracle은 후보군의 이론적 상한선 진단에만 사용하고 실제 선택에는 사용하지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v25_mae_aware_linkage.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"

PAIR_SPLITS = 5
PAIR_SEED = 271828
GOWER_NEIGHBORS = 80
MAX_PAIR_TRAIN_ROWS = 350_000
NEAR_TARGET_LIMIT = 0.05

META_SPLITS = 5
META_INNER_SPLITS = 3
META_SEEDS = (11, 101, 1001, 2026, 31415)
GATE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)

# Public 0.12368 목표를 현재 gap으로 환산한 대략적인 필요 OOF 개선량이다.
TARGET_OOF_GAIN = 0.0022


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / 0.01) * 0.01, 0.0, 1.0)


def robust_scales(
    features: pd.DataFrame,
    indices: np.ndarray,
    numeric: list[str],
) -> dict[str, float]:
    """현재 pair fold의 train 행만으로 숫자형 MAD 크기를 계산한다."""
    scales: dict[str, float] = {}
    for column in numeric:
        values = features.iloc[indices][column].to_numpy(float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        scales[column] = float(mad) if np.isfinite(mad) and mad > 0 else 1.0
    return scales


def unique_pairs(parts: list[np.ndarray]) -> np.ndarray:
    valid = [part.astype(np.int32, copy=False) for part in parts if len(part)]
    if not valid:
        return np.empty((0, 2), dtype=np.int32)
    return np.unique(np.vstack(valid), axis=0)


def expanded_block_pairs(v3, features: pd.DataFrame, categorical: list[str]) -> dict:
    """
    기존 후보와 범주형 7개 중 4개가 같은 후보를 합친다.

    cat4는 과거 binary classifier에서는 실패했지만, 이번에는 MAE-risk라는
    다른 목표로 순위를 매기기 때문에 후보 reservoir로만 다시 사용한다.
    """
    original = v3.train_candidate_pairs(features, v3.linkage_blocks(categorical))
    cat4_blocks = list(combinations(categorical, 4))
    cat4 = v3.train_candidate_pairs(features, cat4_blocks)
    expanded = unique_pairs([original, cat4])
    return {"original": original, "cat4": cat4, "expanded": expanded}


def gower_top_pairs(
    features: pd.DataFrame,
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    scales: dict[str, float],
    n_neighbors: int,
    exclude_self: bool,
) -> np.ndarray:
    """fold-train 기준 Gower형 거리에서 가까운 reference 후보를 만든다."""
    q = features.iloc[query_indices]
    r = features.iloc[reference_indices]
    distance = np.zeros((len(q), len(r)), dtype=np.float32)
    feature_count = 0

    for column in numeric:
        left = q[column].to_numpy(float)
        right = r[column].to_numpy(float)
        left_missing = np.isnan(left)[:, None]
        right_missing = np.isnan(right)[None, :]
        both_missing = left_missing & right_missing
        one_missing = left_missing ^ right_missing
        difference = np.abs(left[:, None] - right[None, :]) / scales[column]
        difference = np.minimum(np.nan_to_num(difference, nan=2.0), 3.0) / 3.0
        difference[both_missing] = 0.0
        difference[one_missing] = 1.0
        distance += difference.astype(np.float32)
        feature_count += 1

    for column in categorical:
        left = q[column].astype("string").fillna("__MISSING__").to_numpy()
        right = r[column].astype("string").fillna("__MISSING__").to_numpy()
        distance += (left[:, None] != right[None, :]).astype(np.float32)
        feature_count += 1

    distance /= max(feature_count, 1)
    if exclude_self:
        reference_position = {int(row): pos for pos, row in enumerate(reference_indices)}
        for q_pos, row in enumerate(query_indices):
            r_pos = reference_position.get(int(row))
            if r_pos is not None:
                distance[q_pos, r_pos] = np.inf

    available = len(reference_indices) - (1 if exclude_self else 0)
    k = min(n_neighbors, max(available, 1))
    local = np.argpartition(distance, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(query_indices, k)
    cols = reference_indices[local.reshape(-1)]
    return np.column_stack([rows, cols]).astype(np.int32)


def orient_cross_pairs(
    pairs: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """정적 pair를 항상 (validation query, fold-train candidate)로 맞춘다."""
    cross = (
        (valid_mask[pairs[:, 0]] & train_mask[pairs[:, 1]])
        | (valid_mask[pairs[:, 1]] & train_mask[pairs[:, 0]])
    )
    result = pairs[cross].copy()
    reverse = train_mask[result[:, 0]]
    result[reverse] = result[reverse][:, ::-1]
    return result


def pair_feature_frame(
    features: pd.DataFrame,
    pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    scales: dict[str, float],
) -> pd.DataFrame:
    """행 쌍의 일치 패턴과 정규화 차이를 pair-risk 입력으로 만든다."""
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    data: dict[str, np.ndarray] = {}
    numeric_equal = []
    numeric_difference = []
    one_missing_rows = []

    for column in numeric:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        a_missing = np.isnan(a)
        b_missing = np.isnan(b)
        equal = (a == b) | (a_missing & b_missing)
        one_missing = a_missing ^ b_missing
        difference = np.minimum(
            np.nan_to_num(np.abs(a - b) / scales[column], nan=3.0),
            5.0,
        )
        difference[a_missing & b_missing] = 0.0
        data[f"{column}_equal"] = equal.astype(np.float32)
        data[f"{column}_difference"] = difference.astype(np.float32)
        data[f"{column}_one_missing"] = one_missing.astype(np.float32)
        numeric_equal.append(equal)
        numeric_difference.append(difference)
        one_missing_rows.append(one_missing)

    categorical_equal = []
    for column in categorical:
        a = left[column].astype("string").fillna("__MISSING__").to_numpy()
        b = right[column].astype("string").fillna("__MISSING__").to_numpy()
        equal = a == b
        data[f"{column}_equal"] = equal.astype(np.float32)
        categorical_equal.append(equal)

    core = ["height", "weight", "cholesterol", "glucose"]
    core_equal = [
        (
            (left[column].to_numpy(float) == right[column].to_numpy(float))
            | (
                np.isnan(left[column].to_numpy(float))
                & np.isnan(right[column].to_numpy(float))
            )
        )
        for column in core
    ]
    numeric_diff_matrix = np.column_stack(numeric_difference)
    data["core_equal_count"] = np.column_stack(core_equal).sum(axis=1)
    data["numeric_equal_count"] = np.column_stack(numeric_equal).sum(axis=1)
    data["categorical_equal_count"] = np.column_stack(categorical_equal).sum(axis=1)
    data["one_missing_count"] = np.column_stack(one_missing_rows).sum(axis=1)
    data["numeric_difference_mean"] = numeric_diff_matrix.mean(axis=1)
    data["numeric_difference_max"] = numeric_diff_matrix.max(axis=1)
    data["numeric_difference_q75"] = np.quantile(
        numeric_diff_matrix, 0.75, axis=1
    )
    total_mismatch = (
        len(numeric)
        - data["numeric_equal_count"]
        + len(categorical)
        - data["categorical_equal_count"]
    )
    data["total_mismatch_count"] = total_mismatch
    return pd.DataFrame(data, dtype=np.float32)


def sample_training_pairs(
    pairs: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    target이 가까운 pair는 모두 살리고 먼 pair를 제한해 학습량과 불균형을 줄인다.

    이 sampling은 현재 outer pair fold의 train target만 사용한다.
    """
    target = np.abs(y[pairs[:, 0]] - y[pairs[:, 1]])
    near = np.flatnonzero(target <= NEAR_TARGET_LIMIT + 1e-12)
    far = np.flatnonzero(target > NEAR_TARGET_LIMIT + 1e-12)
    far_limit = min(len(far), max(2 * len(near), 50_000))
    kept_far = (
        rng.choice(far, size=far_limit, replace=False)
        if far_limit < len(far)
        else far
    )
    keep = np.concatenate([near, kept_far])
    if len(keep) > MAX_PAIR_TRAIN_ROWS:
        near_limit = min(len(near), MAX_PAIR_TRAIN_ROWS // 2)
        kept_near = (
            rng.choice(near, size=near_limit, replace=False)
            if near_limit < len(near)
            else near
        )
        remaining = MAX_PAIR_TRAIN_ROWS - len(kept_near)
        kept_far = (
            rng.choice(far, size=remaining, replace=False)
            if remaining < len(far)
            else far
        )
        keep = np.concatenate([kept_near, kept_far])
    rng.shuffle(keep)
    return pairs[keep], target[keep]


def build_pair_model(seed: int) -> HistGradientBoostingRegressor:
    """후보 복사 시 기대 절대오차를 추정하는 비선형 회귀모델."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=2.0,
        random_state=seed,
    )


def aggregate_query_candidates(
    query_pairs: np.ndarray,
    predicted_risk: np.ndarray,
    pair_features: pd.DataFrame,
    y: np.ndarray,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """query별 최소 예상 MAE 후보와 사후 oracle 상한선을 동시에 집계한다."""
    output = {
        "selected_label": np.full(n_rows, np.nan),
        "predicted_risk": np.full(n_rows, np.nan),
        "risk_margin": np.zeros(n_rows),
        "pair_distance": np.full(n_rows, np.nan),
        "candidate_count": np.zeros(n_rows, dtype=np.int32),
        "label_support": np.zeros(n_rows, dtype=np.int32),
        "oracle_error": np.full(n_rows, np.nan),
        "oracle_exact": np.zeros(n_rows, dtype=bool),
    }
    order = np.argsort(query_pairs[:, 0], kind="stable")
    pairs = query_pairs[order]
    risk = predicted_risk[order]
    distance = pair_features["numeric_difference_mean"].to_numpy()[order]
    rows, starts = np.unique(pairs[:, 0], return_index=True)
    ends = np.r_[starts[1:], len(pairs)]

    for row, start, end in zip(rows, starts, ends):
        candidate_rows = pairs[start:end, 1]
        labels = y[candidate_rows]
        local_risk = risk[start:end]
        local_distance = distance[start:end]
        # 위험 예측이 같으면 feature 거리가 더 작은 후보를 선택한다.
        local_order = np.lexsort((local_distance, local_risk))
        best = int(local_order[0])
        selected = float(labels[best])
        output["selected_label"][row] = selected
        output["predicted_risk"][row] = float(local_risk[best])
        output["pair_distance"][row] = float(local_distance[best])
        output["candidate_count"][row] = int(end - start)
        output["label_support"][row] = int(np.sum(labels == selected))
        if len(local_order) >= 2:
            output["risk_margin"][row] = float(
                local_risk[local_order[1]] - local_risk[best]
            )

        # 아래 두 값은 후보군 상한선 진단에만 사용하며 선택에는 들어가지 않는다.
        oracle_error = float(np.min(np.abs(y[row] - labels)))
        output["oracle_error"][row] = oracle_error
        output["oracle_exact"][row] = oracle_error <= 1e-12
    return output


def make_pair_oof(
    v3,
    train: pd.DataFrame,
    y: np.ndarray,
    expanded_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    """5-fold에서 query를 제외한 pair-risk OOF 후보를 만든다."""
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    n_rows = len(train)
    result = {
        "selected_label": np.full(n_rows, np.nan),
        "predicted_risk": np.full(n_rows, np.nan),
        "risk_margin": np.zeros(n_rows),
        "pair_distance": np.full(n_rows, np.nan),
        "candidate_count": np.zeros(n_rows, dtype=np.int32),
        "label_support": np.zeros(n_rows, dtype=np.int32),
        "oracle_error": np.full(n_rows, np.nan),
        "oracle_exact": np.zeros(n_rows, dtype=bool),
    }
    splitter = KFold(n_splits=PAIR_SPLITS, shuffle=True, random_state=PAIR_SEED)
    rng = np.random.default_rng(PAIR_SEED)
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        train_mask = np.zeros(n_rows, dtype=bool)
        valid_mask = np.zeros(n_rows, dtype=bool)
        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        scales = robust_scales(features, train_idx, numeric)

        static_train = expanded_pairs[
            train_mask[expanded_pairs[:, 0]] & train_mask[expanded_pairs[:, 1]]
        ]
        nearest_train = gower_top_pairs(
            features,
            train_idx,
            train_idx,
            numeric,
            categorical,
            scales,
            GOWER_NEIGHBORS,
            exclude_self=True,
        )
        train_pairs = unique_pairs([static_train, nearest_train])
        sampled_pairs, pair_target = sample_training_pairs(train_pairs, y, rng)
        train_pair_features = pair_feature_frame(
            features,
            sampled_pairs,
            numeric,
            categorical,
            scales,
        )
        model = build_pair_model(PAIR_SEED + fold)
        model.fit(train_pair_features, pair_target)

        static_query = orient_cross_pairs(
            expanded_pairs,
            train_mask,
            valid_mask,
        )
        nearest_query = gower_top_pairs(
            features,
            valid_idx,
            train_idx,
            numeric,
            categorical,
            scales,
            GOWER_NEIGHBORS,
            exclude_self=False,
        )
        query_pairs = unique_pairs([static_query, nearest_query])
        query_features = pair_feature_frame(
            features,
            query_pairs,
            numeric,
            categorical,
            scales,
        )
        predicted_risk = model.predict(query_features)
        fold_output = aggregate_query_candidates(
            query_pairs,
            predicted_risk,
            query_features,
            y,
            n_rows,
        )
        for name in result:
            result[name][valid_idx] = fold_output[name][valid_idx]

        print(
            f"[pair] fold={fold}/{PAIR_SPLITS} "
            f"train_pool={len(train_pairs):,} sampled={len(sampled_pairs):,} "
            f"query_pool={len(query_pairs):,} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    if not np.isfinite(result["selected_label"]).all():
        raise RuntimeError("일부 OOF 행에 MAE-aware 후보가 없습니다.")
    return result


def gate_features(pair_oof: dict[str, np.ndarray], current: np.ndarray) -> np.ndarray:
    """정답 없이 계산 가능한 후보 신뢰도와 현재 예측 차이를 만든다."""
    selected = pair_oof["selected_label"]
    signed_delta = selected - current
    return np.column_stack(
        [
            pair_oof["predicted_risk"],
            pair_oof["risk_margin"],
            pair_oof["pair_distance"],
            np.log1p(pair_oof["candidate_count"]),
            np.log1p(pair_oof["label_support"]),
            np.abs(signed_delta),
            signed_delta,
            current,
            selected,
        ]
    )


def build_gate(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def probability_of_benefit(
    model: Pipeline,
    x: np.ndarray,
) -> np.ndarray:
    classes = model.named_steps["model"].classes_
    probability = model.predict_proba(x)
    positive = np.flatnonzero(classes == 1)
    if len(positive) == 0:
        return np.zeros(len(x), dtype=float)
    return probability[:, positive[0]]


def choose_action(
    probability: np.ndarray,
    current: np.ndarray,
    selected: np.ndarray,
    y: np.ndarray,
) -> dict:
    """inner OOF에서 threshold와 blend alpha를 고르고 무변경과 비교한다."""
    baseline = mean_absolute_error(y, current)
    best = {
        "gain": 0.0,
        "threshold": None,
        "alpha": 0.0,
        "changed_rows": 0,
    }
    for threshold in GATE_THRESHOLDS:
        use = probability >= threshold
        for alpha in BLEND_ALPHAS:
            candidate = current.copy()
            candidate[use] = snap(
                (1.0 - alpha) * current[use] + alpha * selected[use]
            )
            gain = baseline - mean_absolute_error(y, candidate)
            # 동률이면 더 보수적인 높은 threshold, 낮은 alpha를 유지한다.
            key = (gain, threshold, -alpha)
            old_threshold = best["threshold"] if best["threshold"] is not None else 1.0
            old_key = (best["gain"], old_threshold, -best["alpha"])
            if key > old_key:
                best = {
                    "gain": float(gain),
                    "threshold": float(threshold),
                    "alpha": float(alpha),
                    "changed_rows": int(use.sum()),
                }
    return best


def nested_gate(
    x: np.ndarray,
    selected: np.ndarray,
    current: np.ndarray,
    y: np.ndarray,
    usable_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """outer meta-fold 밖에서 gate 학습과 action 선택을 모두 수행한다."""
    prediction = current.copy()
    choices: list[dict] = []
    outer = KFold(n_splits=META_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(outer.split(y), start=1):
        usable_train = train_idx[usable_mask[train_idx]]
        usable_valid = valid_idx[usable_mask[valid_idx]]
        benefit = (
            np.abs(y[usable_train] - selected[usable_train])
            < np.abs(y[usable_train] - current[usable_train])
        ).astype(np.int8)
        if len(np.unique(benefit)) < 2 or len(usable_valid) == 0:
            choices.append(
                {"fold": fold, "threshold": None, "alpha": 0.0, "changed_rows": 0}
            )
            continue

        inner_probability = np.zeros(len(usable_train), dtype=float)
        inner = KFold(
            n_splits=META_INNER_SPLITS,
            shuffle=True,
            random_state=seed + fold,
        )
        for inner_fold, (fit_pos, tune_pos) in enumerate(
            inner.split(usable_train),
            start=1,
        ):
            fit_rows = usable_train[fit_pos]
            tune_rows = usable_train[tune_pos]
            inner_target = (
                np.abs(y[fit_rows] - selected[fit_rows])
                < np.abs(y[fit_rows] - current[fit_rows])
            ).astype(np.int8)
            if len(np.unique(inner_target)) < 2:
                inner_probability[tune_pos] = float(inner_target[0])
                continue
            model = build_gate(seed + 100 * fold + inner_fold)
            model.fit(x[fit_rows], inner_target)
            inner_probability[tune_pos] = probability_of_benefit(
                model,
                x[tune_rows],
            )

        action = choose_action(
            inner_probability,
            current[usable_train],
            selected[usable_train],
            y[usable_train],
        )
        action.update({"fold": fold, "outer_train_rows": int(len(usable_train))})
        if action["threshold"] is None or action["gain"] <= 0:
            action["outer_changed_rows"] = 0
            choices.append(action)
            continue

        model = build_gate(seed + 10_000 + fold)
        model.fit(x[usable_train], benefit)
        valid_probability = probability_of_benefit(model, x[usable_valid])
        use = valid_probability >= action["threshold"]
        changed = usable_valid[use]
        prediction[changed] = snap(
            (1.0 - action["alpha"]) * current[changed]
            + action["alpha"] * selected[changed]
        )
        action["outer_changed_rows"] = int(len(changed))
        choices.append(action)
    return prediction, choices


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
    v24 = load_module(
        "experiment_v24",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()

    print("===== 1. 확장 후보군 생성 =====", flush=True)
    pair_pools = expanded_block_pairs(v3, features, categorical)
    print(
        f"기존={len(pair_pools['original']):,} "
        f"cat4={len(pair_pools['cat4']):,} "
        f"합집합={len(pair_pools['expanded']):,}",
        flush=True,
    )

    print("\n===== 2. MAE-aware pair OOF =====", flush=True)
    pair_oof = make_pair_oof(v3, train, y, pair_pools["expanded"])
    contexts = v24.make_link_contexts(
        load_module(
            "experiment_v4",
            ROOT / "experiment/experiment_v4_deterministic_union.py",
        ),
        train,
        y,
        stored,
    )

    results = []
    oracle_results = []
    direct_results = []
    for context in contexts:
        current = context["current"]
        usable = ~context["mask"]
        current_error = np.abs(y - current)
        oracle_prediction_error = pair_oof["oracle_error"]
        oracle_gain = float(
            np.sum(
                current_error[usable]
                - np.minimum(current_error[usable], oracle_prediction_error[usable])
            )
            / len(y)
        )
        oracle_results.append(
            {
                "link_seed": context["seed"],
                "oracle_gain_with_keep_option": oracle_gain,
                "oracle_exact_candidates_on_unlinked": int(
                    pair_oof["oracle_exact"][usable].sum()
                ),
                "oracle_better_rows": int(
                    (oracle_prediction_error[usable] < current_error[usable]).sum()
                ),
            }
        )

        direct = current.copy()
        direct[usable] = pair_oof["selected_label"][usable]
        direct_score = mean_absolute_error(y, direct)
        direct_results.append(
            {
                "link_seed": context["seed"],
                "oof_mae": float(direct_score),
                "gain_vs_current": float(
                    mean_absolute_error(y, current) - direct_score
                ),
            }
        )

        x_gate = gate_features(pair_oof, current)
        for meta_seed in META_SEEDS:
            candidate, choices = nested_gate(
                x_gate,
                pair_oof["selected_label"],
                current,
                y,
                usable,
                int(meta_seed),
            )
            current_mae = mean_absolute_error(y, current)
            candidate_mae = mean_absolute_error(y, candidate)
            results.append(
                {
                    "link_seed": context["seed"],
                    "meta_seed": int(meta_seed),
                    "current_oof_mae": float(current_mae),
                    "candidate_oof_mae": float(candidate_mae),
                    "gain_vs_current": float(current_mae - candidate_mae),
                    "changed_rows": int(np.sum(candidate != current)),
                    "fold_choices": choices,
                }
            )

    gains = np.asarray([row["gain_vs_current"] for row in results])
    scores = np.asarray([row["candidate_oof_mae"] for row in results])
    mean_gain = float(gains.mean())
    passes_direction = bool(np.all(gains > 0))
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and passes_direction)
    metrics = {
        "protocol": {
            "test_used": False,
            "pair_splits": PAIR_SPLITS,
            "pair_seed": PAIR_SEED,
            "gower_neighbors": GOWER_NEIGHBORS,
            "expanded_blocks": "original + categorical combinations of 4",
            "pair_target": "absolute stress_score difference",
            "meta_splits": META_SPLITS,
            "meta_inner_splits": META_INNER_SPLITS,
            "meta_seeds": META_SEEDS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "candidate_pool": {
            "original_pairs": len(pair_pools["original"]),
            "cat4_pairs": len(pair_pools["cat4"]),
            "expanded_union_pairs": len(pair_pools["expanded"]),
            "mean_candidates_per_row": float(
                np.mean(pair_oof["candidate_count"])
            ),
        },
        "oracle_results": oracle_results,
        "oracle_gain_mean": float(
            np.mean([row["oracle_gain_with_keep_option"] for row in oracle_results])
        ),
        "direct_top1_results": direct_results,
        "nested_gate_results": results,
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "all_repeats_improve": passes_direction,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
        },
    }
    output_path = ROOT / "outputs/experiment_v25_mae_aware_linkage_metrics.json"
    output_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 3. 결과 =====")
    print(f"oracle 평균 개선 상한={metrics['oracle_gain_mean']:+.9f}")
    print(
        "direct top1 평균 개선="
        f"{np.mean([row['gain_vs_current'] for row in direct_results]):+.9f}"
    )
    print(
        f"nested gate OOF={scores.mean():.9f} "
        f"평균 개선={mean_gain:+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/{(gains < 0).sum()}"
    )
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
