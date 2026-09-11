"""
현재까지 가장 낮은 OOF 점수를 기록한 제출 파이프라인 | 공식: 0.1258158348점 
====================================================

실행 방법
---------
VS Code에서 이 파일을 열고 오른쪽 위의 "Python 파일 실행" 버튼을 누르면 된다.
터미널에서는 프로젝트 루트에서 아래처럼 실행할 수 있다.

    .venv/bin/python experiment/experiment_v3_best_record_linkage.py

빠른 동작 확인만 하고 싶다면 다음 옵션을 사용한다. 빠른 모드는 모델 수를
줄이므로 실제 제출 파일을 만드는 용도로 사용하면 안 된다.

    .venv/bin/python experiment/experiment_v3_best_record_linkage.py --quick

파이프라인 요약
---------------
1. train 데이터만 사용하여 ExtraTrees의 OOF 예측을 만든다.
2. train 행끼리 비교하여 "같은 스트레스 점수를 가질 가능성"을 학습한다.
3. 검증/test 행과 매우 비슷한 train 행을 찾으면 그 train 행의 점수를 사용한다.
4. 연결되지 않은 행에는 mean_working별 OOF 잔차 중앙값을 작게 보정한다.
5. test는 각 fold에서 transform/predict만 하며 어떤 통계량도 test로 학습하지 않는다.

누수 방지 원칙
--------------
- 숫자 결측치 중앙값과 원-핫 인코더는 매 fold의 train 부분으로만 fit한다.
- 레코드 연결 분류기와 숫자 차이의 크기 기준도 train만으로 fit한다.
- 연결 확률 기준 0.97과 보정 강도 0.75는 train OOF 실험에서 미리 정한 값이다.
- test target, test 통계량, test-test 연결, pseudo label 재학습을 사용하지 않는다.
- 최종 mean_working 보정값은 full-train 모델의 학습 잔차가 아니라 OOF 잔차로 만든다.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from itertools import combinations
from pathlib import Path

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


TARGET = "stress_score"
ID_COLUMN = "ID"

# 가장 안정적이었던 세 개의 ExtraTrees seed를 평균한다.
BASE_SEEDS = (11, 101, 1001)
BASE_N_SPLITS = 100

# 기존 최저 실험에서 seed=11은 500개, 나머지는 300개 트리를 사용했다.
N_ESTIMATORS_BY_SEED = {11: 500, 101: 300, 1001: 300}
TRIM_RATIO = 0.10

# 레코드 연결 분류기의 설정도 OOF 실험에서 고정한 값이다.
LINKAGE_SEED = 31415
LINKAGE_N_SPLITS = 20
LINK_PROBABILITY_THRESHOLD = 0.97

# mean_working 잔차 보정은 nested 검증에서 가장 자주 선택된 강도다.
WORK_CORRECTION_ALPHA = 0.75
CORRECTION_SEEDS = (11, 101, 1001)
CORRECTION_N_SPLITS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="현재 최저 OOF 레코드 연결 모델로 DACON 제출 파일을 만듭니다."
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
        help="결과 저장 폴더. 기본값은 프로젝트의 outputs 폴더",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="코드 점검용 축소 실행. 이 결과는 제출하지 마세요.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """실행 위치와 관계없이 데이터 폴더가 있는 프로젝트 루트를 찾는다."""
    candidates: list[Path] = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend([start, *start.parents])

    for candidate in candidates:
        if (candidate / "open (3)" / "train.csv").exists():
            return candidate
        if (candidate / "train.csv").exists():
            return candidate

    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. --data-dir로 데이터 폴더를 지정해주세요."
    )


def resolve_data_dir(project_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        data_dir = requested.expanduser().resolve()
    elif (project_root / "open (3)" / "train.csv").exists():
        data_dir = project_root / "open (3)"
    else:
        data_dir = project_root

    required = ("train.csv", "test.csv", "sample_submission.csv")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"필요한 파일이 없습니다: {missing}\n확인한 폴더: {data_dir}"
        )
    return data_dir


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CSV를 읽고 train/test/submission 구조가 맞는지 먼저 검사한다."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if TARGET not in train or TARGET in test:
        raise ValueError("train에는 stress_score가 있고 test에는 없어야 합니다.")
    if ID_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError("train과 test에 ID 열이 필요합니다.")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("train에서 target을 뺀 열과 test 열이 일치하지 않습니다.")
    if len(test) != len(sample) or not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("test와 sample_submission의 행 또는 ID 순서가 다릅니다.")
    if train[TARGET].isna().any():
        raise ValueError("train의 stress_score에 결측치가 있습니다.")
    return train, test, sample


def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    ExtraTrees 입력 변수를 만든다.

    각 파생변수는 같은 행에 이미 있는 값끼리만 계산한다. train/test 전체의
    평균이나 중앙값을 여기서 사용하지 않는다. 계산 중 생긴 무한대는 결측으로
    바꾸고, 실제 결측 처리는 이후 fold-train 전처리기가 담당한다.
    """
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
    """
    현재 fold의 학습 행만으로 결측치 처리와 인코딩 규칙을 학습한다.

    숫자형은 fold-train 중앙값, 범주형은 MISSING 문자열을 사용한다.
    validation/test에서 처음 나타난 범주는 모두 0으로 변환한다.
    """
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


def trimmed_tree_prediction(
    model: ExtraTreesRegressor,
    transformed: np.ndarray,
    trim_ratio: float,
) -> np.ndarray:
    """각 트리 예측의 위·아래 10%를 빼고 평균하여 극단적인 트리를 완화한다."""
    predictions = np.column_stack(
        [tree.predict(transformed) for tree in model.estimators_]
    )
    predictions.sort(axis=1)
    cut = int(len(model.estimators_) * trim_ratio)
    if cut == 0:
        return predictions.mean(axis=1)
    return predictions[:, cut:-cut].mean(axis=1)


def make_base_oof_and_test_prediction(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    seeds: tuple[int, ...],
    n_splits: int,
    quick: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    반복 K-Fold OOF 예측과 test 예측을 함께 만든다.

    OOF의 각 행은 그 행을 학습에서 제외한 모델만 예측한다. test 예측은 모든
    fold 모델 예측의 평균이다. 따라서 test를 모델 학습에 넣지 않는다.
    """
    oof_by_seed: list[np.ndarray] = []
    test_by_seed: list[np.ndarray] = []

    for seed in seeds:
        n_estimators = 100 if quick else N_ESTIMATORS_BY_SEED[seed]
        oof = np.zeros(len(train_features), dtype=float)
        test_sum = np.zeros(len(test_features), dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        started = time.perf_counter()

        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(train_features), start=1
        ):
            preprocessor = build_preprocessor(train_features.iloc[train_idx])
            x_train = preprocessor.fit_transform(train_features.iloc[train_idx])
            x_valid = preprocessor.transform(train_features.iloc[valid_idx])
            x_test = preprocessor.transform(test_features)

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
            oof[valid_idx] = trimmed_tree_prediction(model, x_valid, TRIM_RATIO)
            test_sum += trimmed_tree_prediction(model, x_test, TRIM_RATIO)

            report_every = 1 if quick else 10
            if fold % report_every == 0 or fold == n_splits:
                print(
                    f"[base] seed={seed} fold={fold:03d}/{n_splits} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        oof_by_seed.append(oof)
        test_by_seed.append(test_sum / n_splits)
        print(
            f"[base] seed={seed} OOF MAE={mean_absolute_error(y, oof):.9f}",
            flush=True,
        )

    return (
        np.mean(np.column_stack(oof_by_seed), axis=1),
        np.mean(np.column_stack(test_by_seed), axis=1),
    )


def linkage_blocks(
    categorical: list[str],
) -> list[tuple[str, ...]]:
    """
    비교할 가치가 있는 행 쌍만 만드는 사전 고정 blocking 규칙이다.

    모든 행을 무조건 비교하면 3000행만으로도 약 450만 쌍이 생긴다. 핵심 숫자
    하나가 같거나, 범주형 다섯 개가 같거나, 보조 숫자 두 개가 같은 쌍만 먼저
    후보로 만든다. 이 단계에는 stress_score를 사용하지 않는다.
    """
    core = ["height", "weight", "cholesterol", "glucose"]
    other_numeric = [
        "age",
        "systolic_blood_pressure",
        "diastolic_blood_pressure",
        "bone_density",
    ]
    return (
        [(column,) for column in core]
        + list(combinations(categorical, 5))
        + list(combinations(other_numeric, 2))
    )


def train_candidate_pairs(
    features: pd.DataFrame,
    blocks: list[tuple[str, ...]],
) -> np.ndarray:
    """train 내부에서 blocking 조건을 만족하는 중복 없는 행 쌍을 만든다."""
    result: set[tuple[int, int]] = set()
    for block in blocks:
        grouper: str | list[str] = block[0] if len(block) == 1 else list(block)
        for group in features.groupby(grouper, dropna=False, sort=False).groups.values():
            members = list(group)
            if len(members) >= 2:
                result.update(combinations(members, 2))
    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(sorted(result), dtype=np.int32)


def train_test_candidate_pairs(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    blocks: list[tuple[str, ...]],
) -> np.ndarray:
    """
    각 test 행과 blocking 조건이 같은 train 행의 쌍만 만든다.

    결과의 첫 번째 열은 합친 데이터에서의 test 위치, 두 번째 열은 train 위치다.
    test-test 쌍은 만들지 않으며 test의 분포나 통계량도 계산하지 않는다.
    """
    n_train = len(train_features)
    combined = pd.concat([train_features, test_features], ignore_index=True)
    result: set[tuple[int, int]] = set()

    for block in blocks:
        grouper: str | list[str] = block[0] if len(block) == 1 else list(block)
        groups = combined.groupby(grouper, dropna=False, sort=False).groups.values()
        for group in groups:
            train_rows = [int(i) for i in group if i < n_train]
            test_rows = [int(i) for i in group if i >= n_train]
            for test_row in test_rows:
                for train_row in train_rows:
                    result.add((test_row, train_row))

    if not result:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(sorted(result), dtype=np.int32)


def robust_scales(
    features: pd.DataFrame,
    train_indices: np.ndarray,
    numeric: list[str],
) -> dict[str, float]:
    """지정된 train 행만으로 숫자 차이를 나눌 MAD 크기를 계산한다."""
    result: dict[str, float] = {}
    fold = features.iloc[train_indices]
    for column in numeric:
        values = fold[column].to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        result[column] = float(mad) if np.isfinite(mad) and mad > 0 else 1.0
    return result


def make_pair_features(
    features: pd.DataFrame,
    pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    scales: dict[str, float],
) -> pd.DataFrame:
    """두 행의 일치 여부와 숫자 차이를 레코드 연결 분류기의 입력으로 만든다."""
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    data: dict[str, np.ndarray] = {}
    numeric_equal: list[np.ndarray] = []
    numeric_difference: list[np.ndarray] = []

    for column in numeric:
        a = left[column].to_numpy(dtype=float)
        b = right[column].to_numpy(dtype=float)
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        difference = np.minimum(
            np.nan_to_num(np.abs(a - b) / scales[column], nan=10.0),
            10.0,
        )
        data[f"{column}_equal"] = equal.astype(np.float32)
        data[f"{column}_difference"] = difference.astype(np.float32)
        numeric_equal.append(equal)
        numeric_difference.append(difference)

    categorical_equal: list[np.ndarray] = []
    for column in categorical:
        a = left[column].astype("string").fillna("MISSING").to_numpy()
        b = right[column].astype("string").fillna("MISSING").to_numpy()
        equal = a == b
        data[f"{column}_equal"] = equal.astype(np.float32)
        categorical_equal.append(equal)

    core = ["height", "weight", "cholesterol", "glucose"]
    data["core_equal_count"] = np.column_stack(
        [left[c].to_numpy() == right[c].to_numpy() for c in core]
    ).sum(axis=1).astype(np.float32)
    data["numeric_equal_count"] = np.column_stack(numeric_equal).sum(axis=1)
    data["categorical_equal_count"] = np.column_stack(categorical_equal).sum(axis=1)
    data["numeric_difference_mean"] = np.column_stack(numeric_difference).mean(axis=1)
    data["numeric_difference_max"] = np.column_stack(numeric_difference).max(axis=1)
    return pd.DataFrame(data)


def balanced_pair_sample(
    pair_target: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """같은 점수 쌍은 모두 쓰고, 다른 점수 쌍은 최대 8배까지만 무작위 사용한다."""
    positive = np.flatnonzero(pair_target == 1)
    negative = np.flatnonzero(pair_target == 0)
    if len(positive) == 0:
        raise RuntimeError("레코드 연결 분류기를 학습할 같은-target 쌍이 없습니다.")
    kept_negative = rng.choice(
        negative,
        size=min(len(negative), 8 * len(positive)),
        replace=False,
    )
    keep = np.concatenate([positive, kept_negative])
    rng.shuffle(keep)
    return keep


def build_link_classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2,
        class_weight="balanced",
        random_state=random_state,
    )


def aggregate_link_candidates(
    query_pairs: np.ndarray,
    query_probability: np.ndarray,
    train_target: np.ndarray,
    n_train: int,
    n_output: int,
) -> dict[str, np.ndarray]:
    """행마다 같은 점수를 제안하는 후보들을 합쳐 최종 연결 후보 하나를 고른다."""
    label = np.full(n_output, np.nan)
    probability = np.zeros(n_output)
    margin = np.zeros(n_output)
    support = np.zeros(n_output, dtype=np.int16)
    if len(query_pairs) == 0:
        return {"label": label, "probability": probability, "margin": margin, "support": support}

    # train OOF 쌍의 query 위치는 0~n_train-1이고, test 쌍은 n_train부터 시작한다.
    output_row = query_pairs[:, 0] - (n_train if query_pairs[:, 0].min() >= n_train else 0)
    order = np.argsort(output_row, kind="stable")
    output_row = output_row[order]
    query_pairs = query_pairs[order]
    query_probability = query_probability[order]
    rows, starts = np.unique(output_row, return_index=True)
    ends = np.r_[starts[1:], len(output_row)]

    for row, start, end in zip(rows, starts, ends):
        candidates = pd.DataFrame(
            {
                "label": train_target[query_pairs[start:end, 1]],
                "probability": query_probability[start:end],
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
            + 0.03
            * np.maximum(
                grouped["sum_probability"] - grouped["max_probability"], 0
            )
        )
        grouped = grouped.sort_values("score", ascending=False)
        label[row] = float(grouped.index[0])
        probability[row] = float(grouped.iloc[0]["max_probability"])
        support[row] = int(grouped.iloc[0]["count"])
        first_score = float(grouped.iloc[0]["score"])
        second_score = float(grouped.iloc[1]["score"]) if len(grouped) > 1 else 0.0
        margin[row] = first_score - second_score

    return {
        "label": label,
        "probability": probability,
        "margin": margin,
        "support": support,
    }


def make_oof_link_prediction(
    train_features: pd.DataFrame,
    y: np.ndarray,
    all_pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    n_splits: int,
) -> dict[str, np.ndarray]:
    """검증 행 자신의 stress_score가 연결 학습에 들어가지 않는 OOF 연결값을 만든다."""
    n_rows = len(train_features)
    output = {
        "label": np.full(n_rows, np.nan),
        "probability": np.zeros(n_rows),
        "margin": np.zeros(n_rows),
        "support": np.zeros(n_rows, dtype=np.int16),
    }
    rng = np.random.default_rng(LINKAGE_SEED)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=LINKAGE_SEED)
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_features), 1):
        in_train = np.zeros(n_rows, dtype=bool)
        in_valid = np.zeros(n_rows, dtype=bool)
        in_train[train_idx] = True
        in_valid[valid_idx] = True
        train_pairs = all_pairs[
            in_train[all_pairs[:, 0]] & in_train[all_pairs[:, 1]]
        ]
        cross = (
            (in_valid[all_pairs[:, 0]] & in_train[all_pairs[:, 1]])
            | (in_valid[all_pairs[:, 1]] & in_train[all_pairs[:, 0]])
        )
        query_pairs = all_pairs[cross].copy()
        # query_pairs를 항상 (validation 행, train 행) 순서로 맞춘다.
        reverse = in_train[query_pairs[:, 0]]
        query_pairs[reverse] = query_pairs[reverse][:, ::-1]

        pair_target = (y[train_pairs[:, 0]] == y[train_pairs[:, 1]]).astype(np.int8)
        keep = balanced_pair_sample(pair_target, rng)
        scales = robust_scales(train_features, train_idx, numeric)
        classifier = build_link_classifier(LINKAGE_SEED + fold)
        classifier.fit(
            make_pair_features(
                train_features,
                train_pairs[keep],
                numeric,
                categorical,
                scales,
            ),
            pair_target[keep],
        )
        query_probability = classifier.predict_proba(
            make_pair_features(
                train_features, query_pairs, numeric, categorical, scales
            )
        )[:, 1]
        fold_result = aggregate_link_candidates(
            query_pairs, query_probability, y, n_train=n_rows, n_output=n_rows
        )
        for name in output:
            output[name][valid_idx] = fold_result[name][valid_idx]

        print(
            f"[link OOF] fold={fold:02d}/{n_splits} "
            f"train_pairs={len(train_pairs):,} query_pairs={len(query_pairs):,} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return output


def make_test_link_prediction(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, np.ndarray]:
    """전체 train으로 연결 분류기를 학습하고 각 test 행의 연결 후보를 예측한다."""
    rng = np.random.default_rng(LINKAGE_SEED)
    pair_target = (y[train_pairs[:, 0]] == y[train_pairs[:, 1]]).astype(np.int8)
    keep = balanced_pair_sample(pair_target, rng)
    full_train_index = np.arange(len(train_features))
    scales = robust_scales(train_features, full_train_index, numeric)
    classifier = build_link_classifier(LINKAGE_SEED)
    classifier.fit(
        make_pair_features(
            train_features,
            train_pairs[keep],
            numeric,
            categorical,
            scales,
        ),
        pair_target[keep],
    )

    combined = pd.concat([train_features, test_features], ignore_index=True)
    probability = classifier.predict_proba(
        make_pair_features(combined, test_pairs, numeric, categorical, scales)
    )[:, 1]
    return aggregate_link_candidates(
        test_pairs,
        probability,
        y,
        n_train=len(train_features),
        n_output=len(test_features),
    )


def high_confidence_mask(result: dict[str, np.ndarray]) -> np.ndarray:
    """OOF에서 미리 정한 0.97 기준만 사용한다. test를 보고 기준을 바꾸지 않는다."""
    return (
        np.isfinite(result["label"])
        & (result["probability"] >= LINK_PROBABILITY_THRESHOLD)
        & (result["support"] >= 1)
    )


def work_key(series: pd.Series) -> pd.Series:
    """mean_working의 결측도 하나의 별도 그룹으로 취급한다."""
    return series.fillna("__MISSING__").astype(str)


def crossfit_work_correction(
    train: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    linked: np.ndarray,
    seeds: tuple[int, ...],
    n_splits: int,
) -> np.ndarray:
    """
    각 행을 제외한 데이터로 mean_working별 잔차 중앙값을 계산한다.

    잔차는 실제 점수 - base 예측이다. 양수면 base가 낮게 예측했다는 뜻이고,
    음수면 높게 예측했다는 뜻이다. 연결된 행은 어차피 연결 점수로 교체되므로
    잔차 표를 만들 때 제외한다.
    """
    keys = work_key(train["mean_working"])
    corrections: list[np.ndarray] = []
    for seed in seeds:
        correction = np.zeros(len(train), dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(train):
            usable = train_idx[~linked[train_idx]]
            residual = y[usable] - base_oof[usable]
            table = pd.DataFrame(
                {"key": keys.iloc[usable].to_numpy(), "residual": residual}
            ).groupby("key")["residual"].median()
            fallback = float(np.median(residual))
            correction[valid_idx] = (
                keys.iloc[valid_idx].map(table).fillna(fallback).to_numpy(dtype=float)
            )
        corrections.append(correction)
    return np.mean(np.column_stack(corrections), axis=1)


def fit_final_work_table(
    train: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    linked: np.ndarray,
) -> tuple[pd.Series, float]:
    """전체 OOF 잔차로 test에 적용할 mean_working별 최종 보정표를 만든다."""
    keys = work_key(train["mean_working"])
    usable = np.flatnonzero(~linked)
    residual = y[usable] - base_oof[usable]
    table = pd.DataFrame(
        {"key": keys.iloc[usable].to_numpy(), "residual": residual}
    ).groupby("key")["residual"].median()
    return table, float(np.median(residual))


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """저장 직전 DACON 제출 형식과 예측값을 검사한다."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError("제출 파일 열이 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 파일의 ID 또는 순서가 sample_submission과 다릅니다.")
    prediction = submission[TARGET].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("제출 예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("stress_score 예측값이 0~1 범위를 벗어났습니다.")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    data_dir = resolve_data_dir(project_root, args.data_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_data(data_dir)
    y = train[TARGET].to_numpy(dtype=float)
    train_input = train.drop(columns=[TARGET])
    train_model = make_model_features(train_input)
    test_model = make_model_features(test)

    if args.quick:
        base_seeds = (11,)
        base_splits = 5
        link_splits = 5
        correction_seeds = (11,)
        correction_splits = 5
        suffix = "_quick_NOT_FOR_SUBMISSION"
        print("[주의] quick 모드는 동작 확인용이며 결과를 제출하면 안 됩니다.")
    else:
        base_seeds = BASE_SEEDS
        base_splits = BASE_N_SPLITS
        link_splits = LINKAGE_N_SPLITS
        correction_seeds = CORRECTION_SEEDS
        correction_splits = CORRECTION_N_SPLITS
        suffix = ""

    print(f"data_dir={data_dir}")
    print(f"train={train.shape}, test={test.shape}")
    base_oof, base_test = make_base_oof_and_test_prediction(
        train_model,
        test_model,
        y,
        seeds=base_seeds,
        n_splits=base_splits,
        quick=args.quick,
    )
    base_mae = mean_absolute_error(y, base_oof)

    # 레코드 연결에는 ID를 제외한 원본 설명변수를 사용한다.
    train_link = train.drop(columns=[ID_COLUMN, TARGET])
    test_link = test.drop(columns=[ID_COLUMN])
    numeric = train_link.select_dtypes(include=np.number).columns.tolist()
    categorical = train_link.select_dtypes(exclude=np.number).columns.tolist()
    blocks = linkage_blocks(categorical)
    train_pairs = train_candidate_pairs(train_link, blocks)
    test_pairs = train_test_candidate_pairs(train_link, test_link, blocks)
    print(
        f"[link] train candidate pairs={len(train_pairs):,}, "
        f"train-test candidate pairs={len(test_pairs):,}",
        flush=True,
    )

    oof_link = make_oof_link_prediction(
        train_link,
        y,
        train_pairs,
        numeric,
        categorical,
        n_splits=link_splits,
    )
    test_link_result = make_test_link_prediction(
        train_link,
        test_link,
        y,
        train_pairs,
        test_pairs,
        numeric,
        categorical,
    )
    oof_linked = high_confidence_mask(oof_link)
    test_linked = high_confidence_mask(test_link_result)

    hard_oof = base_oof.copy()
    hard_oof[oof_linked] = oof_link["label"][oof_linked]
    hard_mae = mean_absolute_error(y, hard_oof)

    # OOF용 보정은 각 행을 제외한 fold에서 계산한다.
    work_correction_oof = crossfit_work_correction(
        train,
        y,
        base_oof,
        oof_linked,
        seeds=correction_seeds,
        n_splits=correction_splits,
    )
    final_oof = np.clip(
        base_oof + WORK_CORRECTION_ALPHA * work_correction_oof,
        0,
        1,
    )
    final_oof[oof_linked] = oof_link["label"][oof_linked]
    final_mae = mean_absolute_error(y, final_oof)

    # test 보정표는 full-train 모델의 학습 잔차가 아니라 OOF 잔차로 계산한다.
    work_table, work_fallback = fit_final_work_table(
        train, y, base_oof, oof_linked
    )
    test_work_correction = (
        work_key(test["mean_working"])
        .map(work_table)
        .fillna(work_fallback)
        .to_numpy(dtype=float)
    )
    final_test = np.clip(
        base_test + WORK_CORRECTION_ALPHA * test_work_correction,
        0,
        1,
    )
    final_test[test_linked] = test_link_result["label"][test_linked]

    submission = sample.copy()
    submission[TARGET] = final_test
    validate_submission(submission, sample)

    submission_path = output_dir / f"best_record_linkage_submission{suffix}.csv"
    oof_path = output_dir / f"best_record_linkage_oof{suffix}.csv"
    metrics_path = output_dir / f"best_record_linkage_metrics{suffix}.json"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            "actual": y,
            "base_oof": base_oof,
            "link_label": oof_link["label"],
            "link_probability": oof_link["probability"],
            "linked": oof_linked,
            "work_correction": work_correction_oof,
            "final_oof": final_oof,
        }
    ).to_csv(oof_path, index=False)

    metrics = {
        "base_oof_mae": float(base_mae),
        "hard_link_oof_mae": float(hard_mae),
        "final_oof_mae": float(final_mae),
        "oof_linked_rows": int(oof_linked.sum()),
        "oof_linkage_coverage": float(oof_linked.mean()),
        "oof_linked_mae": float(
            mean_absolute_error(y[oof_linked], final_oof[oof_linked])
        ),
        "test_linked_rows": int(test_linked.sum()),
        "test_linkage_coverage": float(test_linked.mean()),
        "link_probability_threshold": LINK_PROBABILITY_THRESHOLD,
        "work_correction_alpha": WORK_CORRECTION_ALPHA,
        "quick_mode": bool(args.quick),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "submission_path": str(submission_path),
        "oof_path": str(oof_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 최종 결과 =====")
    print(f"Base OOF MAE        : {base_mae:.9f}")
    print(f"Hard-link OOF MAE   : {hard_mae:.9f}")
    print(f"Final OOF MAE       : {final_mae:.9f}")
    print(
        f"OOF linkage coverage: {oof_linked.mean():.2%} "
        f"({oof_linked.sum():,}/{len(train):,})"
    )
    print(
        f"Test linked rows    : {test_linked.mean():.2%} "
        f"({test_linked.sum():,}/{len(test):,})"
    )
    print(f"Submission          : {submission_path}")
    print(f"OOF details         : {oof_path}")
    print(f"Metrics             : {metrics_path}")
    if args.quick:
        print("\nquick 결과는 실제 제출용이 아닙니다. 기본 옵션으로 다시 실행하세요.")


if __name__ == "__main__":
    main()
