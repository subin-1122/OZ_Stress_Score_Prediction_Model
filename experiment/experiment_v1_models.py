"""독립적인 1차 개선 실험: 제한적 파생변수 + 3개 모델 + OOF 앙상블.

``baseline_lightgbm.py``는 최초 비교 기준으로 그대로 보존합니다. 이 파일은
공개된 상위권 코드를 복사하지 않고 다음 질문을 우리 데이터의 교차검증 결과로
직접 확인하기 위한 별도 실험입니다.

* 설명 가능한 건강 파생변수가 실제로 도움이 되는가?
* 트리 모델과 거리 기반 모델 중 어느 쪽이 이 데이터에 더 잘 맞는가?
* 서로 성격이 다른 모델을 섞었을 때 MAE가 안정적으로 좋아지는가?

실행 방법
---------
VS Code에서 이 파일을 열고 Run Python File을 누르거나 터미널에서 실행합니다.

    python experiment_v1_models.py

결과는 ``outputs/experiment_v1_*`` 이름으로 저장됩니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler
from sklearn.svm import SVR


TARGET_COLUMN = "stress_score"
ID_COLUMN = "ID"
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_MIN = 0.0
TARGET_MAX = 1.0

# 스크립트를 프로젝트 루트에 두든 ``experiment/`` 같은 하위 폴더로 옮기든
# 데이터 경로가 깨지지 않도록 현재 파일 위치에서 상위 폴더를 차례로 탐색합니다.
SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_dir(start_dir: Path) -> Path:
    """데이터 폴더 또는 Git 저장소가 있는 가장 가까운 상위 폴더를 찾습니다.

    ``open (3)/train.csv``가 있는 폴더를 가장 우선합니다. 데이터를 아직 받지 않은
    환경에서는 ``.git``이 있는 폴더를 프로젝트 루트로 사용합니다. 어느 쪽도 찾지
    못하면 스크립트 폴더를 반환하며, 이후 ``load_data``가 필요한 파일을 안내합니다.
    """

    candidates = [start_dir, *start_dir.parents]

    for candidate in candidates:
        if (candidate / "open (3)" / "train.csv").is_file():
            return candidate

    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate

    return start_dir


PROJECT_DIR = find_project_dir(SCRIPT_DIR)
DEFAULT_DATA_DIR = PROJECT_DIR / "open (3)"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    """VS Code 버튼 실행과 터미널 실행을 모두 지원하는 경로 옵션입니다."""

    parser = argparse.ArgumentParser(
        description="Domain features + LightGBM/ExtraTrees/SVR OOF experiment"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """상대 경로는 자동 탐색한 프로젝트 루트를 기준으로 해석합니다."""

    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """원본 데이터와 제출 양식을 읽고 치명적인 형식 오류를 먼저 확인합니다."""

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"

    for path in (train_path, test_path, sample_path):
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if TARGET_COLUMN not in train or TARGET_COLUMN in test:
        raise ValueError("train/test의 stress_score 열 구성을 확인하세요.")
    if ID_COLUMN not in train or ID_COLUMN not in test or ID_COLUMN not in sample:
        raise ValueError("train/test/sample_submission에 ID 열이 필요합니다.")

    train_features = train.drop(columns=[ID_COLUMN, TARGET_COLUMN]).columns.tolist()
    test_features = test.drop(columns=[ID_COLUMN]).columns.tolist()
    if train_features != test_features:
        raise ValueError("train과 test의 입력 변수 이름 또는 순서가 다릅니다.")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("test와 sample_submission의 ID 순서가 다릅니다.")

    return train, test, sample


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """타깃이나 다른 행의 통계값을 사용하지 않는 파생변수를 생성합니다.

    이 함수는 한 행 안에 이미 존재하는 측정값만 조합합니다. 따라서 교차검증 전에
    호출해도 검증 정답이나 테스트 분포가 학습 데이터로 유입되지 않습니다.

    파생변수를 무작정 늘리지 않고 해석 가능한 소수의 변수만 사용합니다.

    ``bmi``
        키와 체중을 함께 표현하는 체질량지수입니다.
    ``pulse_pressure``
        수축기 혈압과 이완기 혈압의 차이인 맥압입니다.
    ``mean_arterial_pressure``
        (수축기 + 2×이완기) / 3으로 계산한 평균동맥압 근사값입니다.
    ``bp_ratio``
        두 혈압 값의 상대적 관계를 표현합니다.
    ``*_history_present``
        병력 결측을 단순 제거하지 않고 '기록된 병력이 있는지'를 별도 신호로 둡니다.
    ``work_sleep_strain``
        근로시간과 수면 곤란이 동시에 나타나는 정도를 연속값으로 표현합니다.
    """

    result = df.copy()

    height_m = result["height"] / 100.0
    result["bmi"] = result["weight"] / height_m.pow(2)
    result["pulse_pressure"] = (
        result["systolic_blood_pressure"] - result["diastolic_blood_pressure"]
    )
    result["mean_arterial_pressure"] = (
        result["systolic_blood_pressure"]
        + 2.0 * result["diastolic_blood_pressure"]
    ) / 3.0
    result["bp_ratio"] = result["systolic_blood_pressure"] / (
        result["diastolic_blood_pressure"] + 1e-6
    )

    result["medical_history_present"] = result["medical_history"].notna().astype(int)
    result["family_history_present"] = (
        result["family_medical_history"].notna().astype(int)
    )
    sleep_difficulty = result["sleep_pattern"].eq("sleep difficulty").astype(int)
    result["work_sleep_strain"] = result["mean_working"] * sleep_difficulty

    return result


def make_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Fold 학습 데이터에만 맞춰지는 전처리기를 생성합니다.

    모든 모델에서 숫자 결측은 Fold 학습 중앙값으로 채우고, 결측 여부를 나타내는
    indicator 열도 추가합니다. 범주 결측은 ``__MISSING__``으로 보존하고 One-Hot
    Encoding합니다. 검증 및 테스트는 ``transform``만 거치므로 누수가 없습니다.

    SVR은 변수 크기에 민감하므로 숫자형 변수에 RobustScaler를 추가합니다.
    트리 모델은 크기 변화에 둔감하므로 불필요한 스케일링을 하지 않습니다.
    """

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", RobustScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_lightgbm(fold: int) -> LGBMRegressor:
    """기존 기준선과 비교할 L1 LightGBM 모델입니다."""

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=3_000,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=25,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.10,
        reg_lambda=1.00,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )


def make_extra_trees(fold: int) -> ExtraTreesRegressor:
    """많은 비선형 분기를 무작위화하여 평균내는 ExtraTrees 모델입니다."""

    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=0.70,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )


def make_svr(train_rows: int) -> TransformedTargetRegressor:
    """RobustScaler 전처리와 함께 사용할 RBF SVR 모델입니다.

    C, gamma, epsilon은 공개 코드의 값을 가져오지 않고 이 프로젝트의 동일한
    5-Fold 분할에서 작은 독립 후보군을 비교하여 정한 첫 실험값입니다.

    타깃 QuantileTransformer는 각 Fold의 y_train에만 맞춰집니다. 변환된 공간에서
    SVR을 학습한 뒤 predict 시 원래 0~1 스트레스 점수로 자동 역변환합니다.
    """

    return TransformedTargetRegressor(
        regressor=SVR(
            kernel="rbf",
            C=3.0,
            gamma=0.30,
            epsilon=0.02,
            cache_size=1_000,
        ),
        transformer=QuantileTransformer(
            n_quantiles=min(500, train_rows),
            output_distribution="normal",
            random_state=RANDOM_STATE,
        ),
    )


def fit_one_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """하나의 모델을 공통 Fold로 평가하고 Fold 평균 테스트 예측을 만듭니다."""

    oof = np.zeros(len(X), dtype=float)
    test_by_fold = np.zeros((len(X_test), len(folds)), dtype=float)
    records: list[dict[str, object]] = []

    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_valid = y.iloc[valid_index]

        # SVR에만 수치 스케일링을 적용합니다. 나머지 결측/범주 처리는 동일합니다.
        preprocessor = make_preprocessor(
            numeric_columns,
            categorical_columns,
            scale_numeric=model_name == "svr",
        )
        X_train_processed = preprocessor.fit_transform(X_train)
        X_valid_processed = preprocessor.transform(X_valid)
        X_test_processed = preprocessor.transform(X_test)

        best_iteration: int | None = None
        if model_name == "lightgbm":
            model = make_lightgbm(fold)
            model.fit(
                X_train_processed,
                y_train,
                eval_X=X_valid_processed,
                eval_y=y_valid,
                eval_metric="mae",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=150, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = int(model.best_iteration_ or model.n_estimators)
            valid_prediction = model.predict(
                X_valid_processed, num_iteration=best_iteration
            )
            test_prediction = model.predict(
                X_test_processed, num_iteration=best_iteration
            )
        elif model_name == "extra_trees":
            model = make_extra_trees(fold)
            model.fit(X_train_processed, y_train)
            valid_prediction = model.predict(X_valid_processed)
            test_prediction = model.predict(X_test_processed)
        elif model_name == "svr":
            model = make_svr(len(train_index))
            model.fit(X_train_processed, y_train)
            valid_prediction = model.predict(X_valid_processed)
            test_prediction = model.predict(X_test_processed)
        else:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

        valid_prediction = np.clip(valid_prediction, TARGET_MIN, TARGET_MAX)
        test_prediction = np.clip(test_prediction, TARGET_MIN, TARGET_MAX)
        fold_mae = mean_absolute_error(y_valid, valid_prediction)

        oof[valid_index] = valid_prediction
        test_by_fold[:, fold - 1] = test_prediction
        records.append(
            {
                "model": model_name,
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "best_iteration": best_iteration,
                "mae": fold_mae,
            }
        )
        iteration_text = (
            f" | best_iteration={best_iteration}"
            if best_iteration is not None
            else ""
        )
        print(
            f"  Fold {fold}/{N_SPLITS} | MAE={fold_mae:.6f}{iteration_text}"
        )

    return oof, test_by_fold.mean(axis=1), records


def find_blend(
    y: pd.Series,
    oof_by_model: dict[str, np.ndarray],
    *,
    step: float = 0.05,
) -> tuple[dict[str, float], np.ndarray, float]:
    """세 모델의 음수가 아닌 가중평균을 작은 격자로 탐색합니다.

    복잡한 최적화기는 3,000행의 OOF에 가중치까지 과적합할 수 있습니다. 따라서
    0.05 간격의 단순한 가중치만 비교합니다. 최종 가중치 합은 항상 1입니다.
    """

    names = list(oof_by_model)
    if len(names) != 3:
        raise ValueError("현재 blend 탐색은 세 모델을 기준으로 작성되었습니다.")

    best_mae = float("inf")
    best_weights: dict[str, float] = {}
    best_prediction = np.zeros(len(y), dtype=float)
    units = int(round(1.0 / step))

    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            weights = np.array([first, second, third], dtype=float) / units
            prediction = sum(
                weight * oof_by_model[name]
                for name, weight in zip(names, weights, strict=True)
            )
            prediction = np.clip(prediction, TARGET_MIN, TARGET_MAX)
            score = mean_absolute_error(y, prediction)
            if score < best_mae:
                best_mae = score
                best_prediction = prediction
                best_weights = dict(zip(names, weights, strict=True))

    return best_weights, best_prediction, best_mae


def main() -> None:
    """전체 실험을 실행하고 비교 결과 및 제출 파일을 저장합니다."""

    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_data(data_dir)
    train_ids = train[ID_COLUMN].copy()

    # ID와 TARGET을 제거한 뒤 Train/Test에 같은 결정적 함수를 적용합니다.
    X = create_domain_features(train.drop(columns=[ID_COLUMN, TARGET_COLUMN]))
    X_test = create_domain_features(test.drop(columns=[ID_COLUMN]))
    y = train[TARGET_COLUMN].astype(float)

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # list로 고정해 모든 모델이 정확히 같은 Train/Validation 행을 사용하게 합니다.
    folds = list(kfold.split(X))

    print("=" * 72)
    print("독립적인 1차 개선 실험")
    print("=" * 72)
    print(f"Train/Test: {len(X):,}/{len(X_test):,}행")
    print(f"입력 변수: 숫자형 {len(numeric_columns)}개, 범주형 {len(categorical_columns)}개")

    model_names = ["lightgbm", "extra_trees", "svr"]
    oof_by_model: dict[str, np.ndarray] = {}
    test_by_model: dict[str, np.ndarray] = {}
    fold_records: list[dict[str, object]] = []

    for model_name in model_names:
        print("\n" + "-" * 72)
        print(f"모델: {model_name}")
        oof, test_prediction, records = fit_one_model(
            model_name,
            X,
            y,
            X_test,
            folds,
            numeric_columns,
            categorical_columns,
        )
        oof_by_model[model_name] = oof
        test_by_model[model_name] = test_prediction
        fold_records.extend(records)
        print(f"  전체 OOF MAE: {mean_absolute_error(y, oof):.6f}")

    weights, blend_oof, blend_mae = find_blend(y, oof_by_model)
    blend_test = sum(
        weights[name] * test_by_model[name] for name in model_names
    )
    blend_test = np.clip(blend_test, TARGET_MIN, TARGET_MAX)

    individual_scores = {
        name: mean_absolute_error(y, prediction)
        for name, prediction in oof_by_model.items()
    }
    best_single_name = min(individual_scores, key=individual_scores.get)
    best_single_mae = individual_scores[best_single_name]

    # OOF에서 사실상 차이가 없는 앙상블이면 더 단순한 단일 모델을 선택합니다.
    # 최소 개선 폭을 두어 가중치 격자 자체에 대한 과적합을 줄입니다.
    minimum_blend_gain = 0.0005
    if best_single_mae - blend_mae >= minimum_blend_gain:
        selected_name = "blend"
        selected_oof = blend_oof
        selected_test = blend_test
        selected_mae = blend_mae
    else:
        selected_name = best_single_name
        selected_oof = oof_by_model[best_single_name]
        selected_test = test_by_model[best_single_name]
        selected_mae = best_single_mae

    print("\n" + "=" * 72)
    print("OOF 결과")
    print("=" * 72)
    for name, score in individual_scores.items():
        print(f"{name:12s}: {score:.6f}")
    print(f"blend       : {blend_mae:.6f} | weights={weights}")
    print(f"최종 선택   : {selected_name} | MAE={selected_mae:.6f}")

    score_rows = [
        {
            "candidate": name,
            "oof_mae": score,
            "weight_in_best_blend": weights.get(name, 0.0),
            "selected": name == selected_name,
        }
        for name, score in individual_scores.items()
    ]
    score_rows.append(
        {
            "candidate": "blend",
            "oof_mae": blend_mae,
            "weight_in_best_blend": 1.0,
            "selected": selected_name == "blend",
        }
    )

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            f"actual_{TARGET_COLUMN}": y,
            **{f"pred_{name}": pred for name, pred in oof_by_model.items()},
            "pred_blend": blend_oof,
            "pred_selected": selected_oof,
        }
    )
    fold_output = pd.DataFrame(fold_records)
    score_output = pd.DataFrame(score_rows).sort_values("oof_mae")

    submission = sample.copy()
    submission[TARGET_COLUMN] = selected_test

    submission_path = output_dir / "experiment_v1_submission.csv"
    oof_path = output_dir / "experiment_v1_oof.csv"
    fold_path = output_dir / "experiment_v1_fold_results.csv"
    score_path = output_dir / "experiment_v1_scores.csv"

    submission.to_csv(submission_path, index=False)
    oof_output.to_csv(oof_path, index=False)
    fold_output.to_csv(fold_path, index=False)
    score_output.to_csv(score_path, index=False)

    print("\n저장 완료")
    print(f"제출 파일 : {submission_path}")
    print(f"점수 비교 : {score_path}")
    print(f"OOF 예측  : {oof_path}")
    print(f"Fold 결과 : {fold_path}")


if __name__ == "__main__":
    main()
