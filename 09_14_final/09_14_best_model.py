"""
DACON 스트레스 점수 예측 — 2026-09-14 최고 모델 (Public MAE 0.12404)
=====================================================================

이 파일 하나가 train.csv, test.csv, sample_submission.csv를 읽어 학습부터
최종 제출 파일 생성까지 전부 수행한다. 과거 experiment_v*.py나 중간 예측 CSV는
필요하지 않다.

최종 파이프라인
--------------
1. 3-seed × 100-fold ExtraTrees 절사평균 회귀
2. train에서 학습한 고신뢰 train-test 레코드 연결
3. 설명 가능한 완전일치 결정 규칙 연결
4. 미연결 행의 mean_working 잔차 보정(alpha=1.30)
5. 0.01 단위 snapping
6. target-free 전역 매칭으로 찾은 test-test 774쌍의 예측을 50:50 평균

데이터 누수 방지
---------------
* fold의 결측치 중앙값과 one-hot encoder는 fold-train으로만 fit한다.
* 레코드 연결 분류기, 숫자 거리 크기, mean_working 보정표는 train으로만 fit한다.
* test target, test의 평균/중앙값, test 기반 결측치 통계를 사용하지 않는다.
* test 입력은 예측과 운영진 허용 범위에서 확인한 행 간 구조 매칭에만 사용한다.

실행 예시
---------
프로젝트 루트에서:

    .venv/bin/python 09_14_final/09_14_best_model.py

데이터 폴더를 직접 지정할 수도 있다.

    .venv/bin/python 09_14_final/09_14_best_model.py \
        --data-dir "open (3)" --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# 재현성을 위해 고정한 설정
# -----------------------------------------------------------------------------
TARGET = "stress_score"
ID_COLUMN = "ID"

BASE_SEEDS = (11, 101, 1001)
BASE_N_SPLITS = 100
N_ESTIMATORS_BY_SEED = {11: 500, 101: 300, 1001: 300}
TRIM_RATIO = 0.10

LINKAGE_SEED = 31415
LINKAGE_N_SPLITS = 20
LINK_PROBABILITY_THRESHOLD = 0.97

# Public 0.12404를 기록한 보정 강도다.
WORK_CORRECTION_ALPHA = 1.30

# 전역 매칭 숫자 거리에서 한쪽만 결측일 때 주는 고정 패널티다.
ONE_MISSING_PENALTY = 3.0

CORE_COLUMNS = ["height", "weight", "cholesterol", "glucose"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Public MAE 0.12404 최종 제출 파일을 처음부터 재현합니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="train.csv, test.csv, sample_submission.csv가 있는 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="제출 파일과 실행 기록을 저장할 폴더(기본: 프로젝트/outputs)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="5-fold/1-seed 동작 확인용. quick 결과는 제출하면 안 됩니다.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """VS Code 실행 위치와 무관하게 데이터 폴더가 있는 프로젝트를 찾는다."""
    checked: set[Path] = set()
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / "open (3)" / "train.csv").exists():
                return candidate
            if (candidate / "train.csv").exists():
                return candidate
    raise FileNotFoundError(
        "train.csv를 찾지 못했습니다. --data-dir로 데이터 폴더를 지정해주세요."
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = find_project_root()
    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
    elif (root / "open (3)" / "train.csv").exists():
        data_dir = root / "open (3)"
    else:
        data_dir = root
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "outputs"
    )
    return root, data_dir, output_dir


def load_data(data_dir: Path):
    required = ("train.csv", "test.csv", "sample_submission.csv")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"필요한 파일이 없습니다: {missing}\n폴더: {data_dir}")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if TARGET not in train or TARGET in test:
        raise ValueError("train에는 stress_score가 있고 test에는 없어야 합니다.")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("train에서 target을 뺀 열과 test 열이 일치하지 않습니다.")
    if len(test) != len(sample) or not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("test와 sample_submission의 행 또는 ID 순서가 다릅니다.")
    if train[TARGET].isna().any():
        raise ValueError("train의 stress_score에 결측치가 있습니다.")
    return train, test, sample


# -----------------------------------------------------------------------------
# 1. ExtraTrees 회귀 모델
# -----------------------------------------------------------------------------
def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """한 행 안의 값만 사용해 파생변수를 만든다."""
    x = frame.drop(
        columns=[ID_COLUMN, "mean_working", "age", "sleep_pattern"],
        errors="raise",
    ).copy()
    height_m = pd.to_numeric(x["height"], errors="coerce") / 100.0
    x["bmi"] = pd.to_numeric(x["weight"], errors="coerce") / height_m.where(
        height_m > 0
    ).pow(2)
    x["chol_minus_weight"] = x["cholesterol"] - x["weight"]
    x["chol_plus_glucose"] = x["cholesterol"] + x["glucose"]
    x["height_times_bone"] = x["height"] * x["bone_density"]
    x["systolic_plus_bone"] = x["systolic_blood_pressure"] + x["bone_density"]
    x["chol_div_glucose"] = x["cholesterol"] / x["glucose"]
    x["bone_div_height"] = x["bone_density"] / x["height"]
    x["chol_times_bone"] = x["cholesterol"] * x["bone_density"]
    x["weight_plus_glucose"] = x["weight"] + x["glucose"]
    return x.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(x_train_fold: pd.DataFrame) -> ColumnTransformer:
    """결측치 처리와 인코딩을 현재 fold의 train 부분으로만 학습한다."""
    categorical = x_train_fold.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric = [column for column in x_train_fold.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def trimmed_tree_prediction(model, transformed: np.ndarray) -> np.ndarray:
    """트리 예측의 위·아래 10%를 제외한 평균을 사용한다."""
    prediction = np.column_stack(
        [tree.predict(transformed) for tree in model.estimators_]
    )
    prediction.sort(axis=1)
    cut = int(len(model.estimators_) * TRIM_RATIO)
    return prediction.mean(axis=1) if cut == 0 else prediction[:, cut:-cut].mean(axis=1)


def make_base_oof_and_test(train_x, test_x, y, seeds, n_splits, quick):
    """OOF 예측과 test 예측을 동일한 fold 모델들로 생성한다."""
    oof_by_seed = []
    test_by_seed = []
    for seed in seeds:
        n_estimators = 100 if quick else N_ESTIMATORS_BY_SEED[seed]
        oof = np.zeros(len(train_x), dtype=float)
        test_sum = np.zeros(len(test_x), dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        started = time.perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x), start=1):
            preprocessor = build_preprocessor(train_x.iloc[train_idx])
            x_train = preprocessor.fit_transform(train_x.iloc[train_idx])
            x_valid = preprocessor.transform(train_x.iloc[valid_idx])
            x_test = preprocessor.transform(test_x)
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                criterion="squared_error",
                max_features=1,
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed * 1000 + fold,
            )
            model.fit(x_train, y[train_idx])
            oof[valid_idx] = trimmed_tree_prediction(model, x_valid)
            test_sum += trimmed_tree_prediction(model, x_test)
            report_every = 1 if quick else 10
            if fold % report_every == 0 or fold == n_splits:
                print(
                    f"[base] seed={seed} fold={fold:03d}/{n_splits} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        oof_by_seed.append(oof)
        test_by_seed.append(test_sum / n_splits)
        print(f"[base] seed={seed} OOF MAE={mean_absolute_error(y, oof):.9f}")
    return (
        np.mean(np.column_stack(oof_by_seed), axis=1),
        np.mean(np.column_stack(test_by_seed), axis=1),
    )


# -----------------------------------------------------------------------------
# 2. 학습형 train-test 레코드 연결
# -----------------------------------------------------------------------------
def linkage_blocks(categorical: list[str]):
    core = ["height", "weight", "cholesterol", "glucose"]
    other = [
        "age",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "bone_density",
    ]
    return (
        [(column,) for column in core]
        + list(combinations(categorical, 5))
        + list(combinations(other, 2))
    )


def train_candidate_pairs(features, blocks) -> np.ndarray:
    result: set[tuple[int, int]] = set()
    for block in blocks:
        grouper = block[0] if len(block) == 1 else list(block)
        for group in features.groupby(grouper, dropna=False, sort=False).groups.values():
            members = list(group)
            if len(members) >= 2:
                result.update(combinations(members, 2))
    return (
        np.asarray(sorted(result), dtype=np.int32)
        if result
        else np.empty((0, 2), dtype=np.int32)
    )


def train_test_candidate_pairs(train_features, test_features, blocks) -> np.ndarray:
    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], ignore_index=True)
    result: set[tuple[int, int]] = set()
    for block in blocks:
        grouper = block[0] if len(block) == 1 else list(block)
        for group in combined.groupby(grouper, dropna=False, sort=False).groups.values():
            train_rows = [int(index) for index in group if index < n_train]
            test_rows = [int(index) for index in group if index >= n_train]
            for test_row in test_rows:
                for train_row in train_rows:
                    result.add((test_row, train_row))
    return (
        np.asarray(sorted(result), dtype=np.int32)
        if result
        else np.empty((0, 2), dtype=np.int32)
    )


def robust_scales(features, train_indices, numeric) -> dict[str, float]:
    result = {}
    fold = features.iloc[train_indices]
    for column in numeric:
        values = fold[column].to_numpy(float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        result[column] = float(mad) if np.isfinite(mad) and mad > 0 else 1.0
    return result


def make_pair_features(features, pairs, numeric, categorical, scales):
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    data = {}
    numeric_equal = []
    numeric_difference = []
    for column in numeric:
        a = left[column].to_numpy(float)
        b = right[column].to_numpy(float)
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        difference = np.minimum(
            np.nan_to_num(np.abs(a - b) / scales[column], nan=10.0), 10.0
        )
        data[f"{column}_equal"] = equal.astype(np.float32)
        data[f"{column}_difference"] = difference.astype(np.float32)
        numeric_equal.append(equal)
        numeric_difference.append(difference)
    categorical_equal = []
    for column in categorical:
        a = left[column].astype("string").fillna("MISSING").to_numpy()
        b = right[column].astype("string").fillna("MISSING").to_numpy()
        equal = a == b
        data[f"{column}_equal"] = equal.astype(np.float32)
        categorical_equal.append(equal)
    data["core_equal_count"] = np.column_stack(
        [left[c].to_numpy() == right[c].to_numpy() for c in CORE_COLUMNS]
    ).sum(axis=1).astype(np.float32)
    data["numeric_equal_count"] = np.column_stack(numeric_equal).sum(axis=1)
    data["categorical_equal_count"] = np.column_stack(categorical_equal).sum(axis=1)
    data["numeric_difference_mean"] = np.column_stack(numeric_difference).mean(axis=1)
    data["numeric_difference_max"] = np.column_stack(numeric_difference).max(axis=1)
    return pd.DataFrame(data)


def balanced_pair_sample(pair_target, rng) -> np.ndarray:
    positive = np.flatnonzero(pair_target == 1)
    negative = np.flatnonzero(pair_target == 0)
    if len(positive) == 0:
        raise RuntimeError("레코드 연결 분류기의 positive pair가 없습니다.")
    kept_negative = rng.choice(
        negative, size=min(len(negative), 8 * len(positive)), replace=False
    )
    keep = np.concatenate([positive, kept_negative])
    rng.shuffle(keep)
    return keep


def build_link_classifier(random_state):
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2,
        class_weight="balanced",
        random_state=random_state,
    )


def aggregate_link_candidates(query_pairs, probability, train_target, n_train, n_output):
    label = np.full(n_output, np.nan)
    best_probability = np.zeros(n_output)
    margin = np.zeros(n_output)
    support = np.zeros(n_output, dtype=np.int16)
    if len(query_pairs) == 0:
        return {
            "label": label,
            "probability": best_probability,
            "margin": margin,
            "support": support,
        }
    offset = n_train if query_pairs[:, 0].min() >= n_train else 0
    output_row = query_pairs[:, 0] - offset
    order = np.argsort(output_row, kind="stable")
    output_row = output_row[order]
    query_pairs = query_pairs[order]
    probability = probability[order]
    rows, starts = np.unique(output_row, return_index=True)
    ends = np.r_[starts[1:], len(output_row)]
    for row, start, end in zip(rows, starts, ends):
        candidates = pd.DataFrame(
            {
                "label": train_target[query_pairs[start:end, 1]],
                "probability": probability[start:end],
            }
        )
        grouped = candidates.groupby("label").agg(
            max_probability=("probability", "max"),
            sum_probability=("probability", "sum"),
            count=("probability", "size"),
        )
        grouped["score"] = (
            grouped["max_probability"]
            + 0.15 * np.log1p(grouped["count"])
            + 0.03 * np.maximum(grouped["sum_probability"] - grouped["max_probability"], 0)
        )
        grouped = grouped.sort_values("score", ascending=False)
        label[row] = float(grouped.index[0])
        best_probability[row] = float(grouped.iloc[0]["max_probability"])
        support[row] = int(grouped.iloc[0]["count"])
        first = float(grouped.iloc[0]["score"])
        second = float(grouped.iloc[1]["score"]) if len(grouped) > 1 else 0.0
        margin[row] = first - second
    return {
        "label": label,
        "probability": best_probability,
        "margin": margin,
        "support": support,
    }


def make_oof_link_prediction(train_features, y, all_pairs, numeric, categorical, n_splits):
    n_rows = len(train_features)
    output = {
        "label": np.full(n_rows, np.nan),
        "probability": np.zeros(n_rows),
        "margin": np.zeros(n_rows),
        "support": np.zeros(n_rows, dtype=np.int16),
    }
    rng = np.random.default_rng(LINKAGE_SEED)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=LINKAGE_SEED)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_features), start=1):
        in_train = np.zeros(n_rows, dtype=bool)
        in_valid = np.zeros(n_rows, dtype=bool)
        in_train[train_idx] = True
        in_valid[valid_idx] = True
        train_pairs = all_pairs[in_train[all_pairs[:, 0]] & in_train[all_pairs[:, 1]]]
        cross = (
            (in_valid[all_pairs[:, 0]] & in_train[all_pairs[:, 1]])
            | (in_valid[all_pairs[:, 1]] & in_train[all_pairs[:, 0]])
        )
        query_pairs = all_pairs[cross].copy()
        reverse = in_train[query_pairs[:, 0]]
        query_pairs[reverse] = query_pairs[reverse][:, ::-1]
        pair_target = (y[train_pairs[:, 0]] == y[train_pairs[:, 1]]).astype(np.int8)
        keep = balanced_pair_sample(pair_target, rng)
        scales = robust_scales(train_features, train_idx, numeric)
        classifier = build_link_classifier(LINKAGE_SEED + fold)
        classifier.fit(
            make_pair_features(train_features, train_pairs[keep], numeric, categorical, scales),
            pair_target[keep],
        )
        query_probability = classifier.predict_proba(
            make_pair_features(train_features, query_pairs, numeric, categorical, scales)
        )[:, 1]
        fold_result = aggregate_link_candidates(
            query_pairs, query_probability, y, n_rows, n_rows
        )
        for name in output:
            output[name][valid_idx] = fold_result[name][valid_idx]
        print(f"[link OOF] fold={fold:02d}/{n_splits}", flush=True)
    return output


def make_test_link_prediction(train_features, test_features, y, train_pairs, test_pairs, numeric, categorical):
    rng = np.random.default_rng(LINKAGE_SEED)
    pair_target = (y[train_pairs[:, 0]] == y[train_pairs[:, 1]]).astype(np.int8)
    keep = balanced_pair_sample(pair_target, rng)
    scales = robust_scales(train_features, np.arange(len(train_features)), numeric)
    classifier = build_link_classifier(LINKAGE_SEED)
    classifier.fit(
        make_pair_features(train_features, train_pairs[keep], numeric, categorical, scales),
        pair_target[keep],
    )
    combined = pd.concat([train_features, test_features], ignore_index=True)
    probability = classifier.predict_proba(
        make_pair_features(combined, test_pairs, numeric, categorical, scales)
    )[:, 1]
    return aggregate_link_candidates(
        test_pairs, probability, y, len(train_features), len(test_features)
    )


def high_confidence_mask(result) -> np.ndarray:
    return (
        np.isfinite(result["label"])
        & (result["probability"] >= LINK_PROBABILITY_THRESHOLD)
        & (result["support"] >= 1)
    )


# -----------------------------------------------------------------------------
# 3. mean_working 보정과 결정 규칙 연결
# -----------------------------------------------------------------------------
def work_key(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str)


def fit_final_work_table(train, y, base_oof, learned_linked):
    """train OOF 잔차만으로 test용 근로시간 보정표를 만든다."""
    keys = work_key(train["mean_working"])
    usable = np.flatnonzero(~learned_linked)
    residual = y[usable] - base_oof[usable]
    table = pd.DataFrame(
        {"key": keys.iloc[usable].to_numpy(), "residual": residual}
    ).groupby("key")["residual"].median()
    return table, float(np.median(residual))


def categorical_key(frame, columns) -> pd.Series:
    return frame[columns].astype("string").fillna("__MISSING__").agg("||".join, axis=1)


def make_test_rule_labels(train_features, test_features, y, numeric, categorical):
    """범주형 전체 일치 + 숫자 2개/핵심 1개 일치 + target 만장일치 규칙."""
    labels = np.full(len(test_features), np.nan)
    train_keys = categorical_key(train_features, categorical)
    test_keys = categorical_key(test_features, categorical)
    train_groups = pd.Series(train_features.index, index=train_keys).groupby(level=0)
    test_groups = pd.Series(test_features.index, index=test_keys).groupby(level=0)
    train_lookup = {key: values.to_numpy() for key, values in train_groups}
    core_positions = [numeric.index(column) for column in CORE_COLUMNS]
    for key, test_rows_series in test_groups:
        train_rows = train_lookup.get(key)
        if train_rows is None:
            continue
        test_rows = test_rows_series.to_numpy()
        train_num = train_features.loc[train_rows, numeric].to_numpy()
        test_num = test_features.loc[test_rows, numeric].to_numpy()
        equal = (test_num[:, None, :] == train_num[None, :, :]) | (
            pd.isna(test_num[:, None, :]) & pd.isna(train_num[None, :, :])
        )
        eligible = (equal.sum(axis=2) >= 2) & (
            equal[:, :, core_positions].sum(axis=2) >= 1
        )
        for local_test, test_row in enumerate(test_rows):
            candidates = train_rows[eligible[local_test]]
            if len(candidates):
                proposed = np.unique(y[candidates])
                if len(proposed) == 1:
                    labels[test_row] = proposed[0]
    return labels


def snap_001(values):
    """train target에서 확인한 0.01 격자로 각 행 예측을 맞춘다."""
    return np.clip(np.rint(np.asarray(values) / 0.01) * 0.01, 0.0, 1.0)


# -----------------------------------------------------------------------------
# 4. target-free 전역 완전매칭 및 test-test 예측 평균
# -----------------------------------------------------------------------------
def global_minimum_matching(train, test):
    """범주형 그룹 안에서 train-MAD 숫자 L1 거리가 최소인 완전매칭을 만든다."""
    train_raw = train.drop(columns=[ID_COLUMN, TARGET])
    test_raw = test.drop(columns=[ID_COLUMN])
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    numeric = train_raw.select_dtypes(include=np.number).columns.tolist()
    categorical = train_raw.select_dtypes(exclude=np.number).columns.tolist()
    train_values = train_raw[numeric].to_numpy(float)
    median = np.nanmedian(train_values, axis=0)
    mad = np.nanmedian(np.abs(train_values - median), axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)
    all_values = combined[numeric].to_numpy(float)
    matching = []
    grouper = categorical[0] if len(categorical) == 1 else categorical
    for member_index in combined.groupby(grouper, dropna=False, sort=False).groups.values():
        members = list(map(int, member_index))
        if len(members) % 2:
            raise ValueError("전역 매칭 범주 그룹 크기가 홀수입니다.")
        graph = nx.Graph()
        graph.add_nodes_from(members)
        for left_position, left in enumerate(members):
            a = all_values[left]
            for right in members[left_position + 1 :]:
                b = all_values[right]
                both_missing = np.isnan(a) & np.isnan(b)
                one_missing = np.isnan(a) ^ np.isnan(b)
                valid = ~(both_missing | one_missing)
                distance = np.zeros(len(numeric), dtype=float)
                distance[one_missing] = ONE_MISSING_PENALTY
                distance[valid] = np.abs(a[valid] - b[valid]) / scale[valid]
                # 같은 거리일 때도 항상 같은 결과를 내기 위한 고정 tie-breaker다.
                weight = float(distance.sum()) + 1e-10 * (left * 6001 + right)
                graph.add_edge(left, right, weight=weight)
        matching.extend(
            tuple(sorted(map(int, edge)))
            for edge in nx.min_weight_matching(graph, weight="weight")
        )
    return set(matching)


def apply_test_pair_average(prediction, global_pairs, n_train):
    test_pairs = sorted(
        (left - n_train, right - n_train)
        for left, right in global_pairs
        if left >= n_train
    )
    if len(test_pairs) != 774:
        raise ValueError(f"예상한 test-test 774쌍과 다릅니다: {len(test_pairs)}쌍")
    output = prediction.copy()
    for left, right in test_pairs:
        value = float(output[[left, right]].mean())
        output[[left, right]] = value
    return output, test_pairs


def validate_submission(submission, sample):
    if list(submission.columns) != list(sample.columns):
        raise ValueError("제출 열이 sample_submission과 다릅니다.")
    if len(submission) != len(sample):
        raise ValueError("제출 행 수가 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 ID 또는 순서가 sample_submission과 다릅니다.")
    prediction = submission[TARGET].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("제출값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("제출값이 0~1 범위를 벗어났습니다.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root, data_dir, output_dir = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample = load_data(data_dir)
    y = train[TARGET].to_numpy(float)

    if args.quick:
        base_seeds = (11,)
        base_splits = 5
        link_splits = 5
        suffix = "_quick_NOT_FOR_SUBMISSION"
        print("[주의] quick 결과는 제출용이 아닙니다.")
    else:
        base_seeds = BASE_SEEDS
        base_splits = BASE_N_SPLITS
        link_splits = LINKAGE_N_SPLITS
        suffix = ""

    print(f"data_dir={data_dir}")
    print(f"train={train.shape}, test={test.shape}")
    train_model = make_model_features(train.drop(columns=[TARGET]))
    test_model = make_model_features(test)
    base_oof, base_test = make_base_oof_and_test(
        train_model, test_model, y, base_seeds, base_splits, args.quick
    )

    train_link = train.drop(columns=[ID_COLUMN, TARGET])
    test_link = test.drop(columns=[ID_COLUMN])
    numeric = train_link.select_dtypes(include=np.number).columns.tolist()
    categorical = train_link.select_dtypes(exclude=np.number).columns.tolist()
    blocks = linkage_blocks(categorical)
    train_pairs = train_candidate_pairs(train_link, blocks)
    test_pairs = train_test_candidate_pairs(train_link, test_link, blocks)
    print(
        f"[link] train pairs={len(train_pairs):,}, "
        f"train-test pairs={len(test_pairs):,}",
        flush=True,
    )

    oof_link = make_oof_link_prediction(
        train_link, y, train_pairs, numeric, categorical, link_splits
    )
    test_link = make_test_link_prediction(
        train_link,
        test_link,
        y,
        train_pairs,
        test_pairs,
        numeric,
        categorical,
    )
    oof_learned_mask = high_confidence_mask(oof_link)
    test_learned_mask = high_confidence_mask(test_link)

    work_table, work_fallback = fit_final_work_table(
        train, y, base_oof, oof_learned_mask
    )
    test_work_correction = (
        work_key(test["mean_working"])
        .map(work_table)
        .fillna(work_fallback)
        .to_numpy(float)
    )
    prediction = np.clip(
        base_test + WORK_CORRECTION_ALPHA * test_work_correction, 0.0, 1.0
    )
    prediction[test_learned_mask] = test_link["label"][test_learned_mask]

    # 학습형 연결 위에 설명 가능한 결정 규칙을 합친다. 두 규칙이 겹치면
    # 기존 최고 파일과 동일하게 결정 규칙 값을 마지막에 적용한다.
    deterministic_label = make_test_rule_labels(
        train_link,
        test.drop(columns=[ID_COLUMN]),
        y,
        numeric,
        categorical,
    )
    deterministic_mask = np.isfinite(deterministic_label)
    prediction[deterministic_mask] = deterministic_label[deterministic_mask]
    prediction = snap_001(prediction)

    global_pairs = global_minimum_matching(train, test)
    prediction, test_test_pairs = apply_test_pair_average(
        prediction, global_pairs, len(train)
    )

    submission = sample.copy()
    submission[TARGET] = prediction
    validate_submission(submission, sample)
    output_path = output_dir / f"09_14_best_submission{suffix}.csv"
    metrics_path = output_dir / f"09_14_best_run_metrics{suffix}.json"
    submission.to_csv(output_path, index=False, float_format="%.3f")

    # 개발 환경에 기존 0.12404 파일이 있으면 자동으로 완전 재현 여부를 검사한다.
    reference_path = root / "outputs/pair_average_global_774_alpha130_submission.csv"
    reproduction_max_diff = None
    if not args.quick and reference_path.exists() and reference_path != output_path:
        reference = pd.read_csv(reference_path)
        if reference[ID_COLUMN].equals(submission[ID_COLUMN]):
            reproduction_max_diff = float(
                np.max(np.abs(reference[TARGET].to_numpy(float) - prediction))
            )
            if reproduction_max_diff > 1e-12:
                raise RuntimeError(
                    "기존 Public 0.12404 파일과 예측이 일치하지 않습니다: "
                    f"max diff={reproduction_max_diff}"
                )

    metrics = {
        "public_mae": 0.12404,
        "quick_mode": args.quick,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "base_seeds": list(base_seeds),
        "base_folds": base_splits,
        "trees_by_seed": (
            {str(seed): N_ESTIMATORS_BY_SEED[seed] for seed in base_seeds}
            if not args.quick
            else {"11": 100}
        ),
        "base_oof_mae": float(mean_absolute_error(y, base_oof)),
        "learned_test_links": int(test_learned_mask.sum()),
        "deterministic_test_links": int(deterministic_mask.sum()),
        "union_test_links": int((test_learned_mask | deterministic_mask).sum()),
        "work_correction_alpha": WORK_CORRECTION_ALPHA,
        "global_total_pairs": len(global_pairs),
        "global_test_test_pairs": len(test_test_pairs),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "reference_reproduction_max_abs_diff": reproduction_max_diff,
        "submission_path": str(output_path),
        "submission_sha256": file_sha256(output_path),
        "sklearn_version": sklearn.__version__,
        "test_target_used": False,
        "test_encoder_scaler_imputer_fit": False,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 09_14 최종 모델 완료 =====")
    print(f"학습형 test 연결      : {test_learned_mask.sum():,}")
    print(f"결정 규칙 test 연결   : {deterministic_mask.sum():,}")
    print(f"연결 합집합           : {(test_learned_mask | deterministic_mask).sum():,}")
    print(f"전역 test-test 쌍     : {len(test_test_pairs):,}")
    print(f"기존 파일 재현 최대차: {reproduction_max_diff}")
    print(f"제출 파일             : {output_path}")
    print(f"실행 기록             : {metrics_path}")


if __name__ == "__main__":
    main()
