"""스트레스 점수 예측 해커톤용 LightGBM 베이스라인.

이 파일은 VS Code에서 그대로 실행할 수 있는 단일 Python 스크립트입니다.
기존 DACON 베이스라인의 전체 흐름은 유지하되 다음 부분을 보완했습니다.

1. 5-Fold 교차검증으로 제출 전에 로컬 MAE를 확인합니다.
2. 각 Fold의 학습 데이터만으로 결측치 처리와 One-Hot Encoding을 학습합니다.
3. 테스트 데이터는 이미 학습된 전처리기에 ``transform``만 적용합니다.
4. LightGBM의 early stopping으로 불필요한 반복과 과적합을 줄입니다.
5. 5개 Fold 모델의 테스트 예측을 평균하여 제출값을 안정화합니다.

실행 예시
---------
VS Code 터미널에서 아래 명령을 실행합니다.

    python baseline_lightgbm.py

데이터 폴더나 결과 폴더가 다른 경우에는 다음처럼 지정할 수 있습니다.

    python baseline_lightgbm.py --data-dir "open (3)" --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# 대회와 재현성 관련 설정
# -----------------------------------------------------------------------------
# 목표 변수와 ID 변수는 여러 함수에서 반복 사용하므로 상수로 관리합니다.
TARGET_COLUMN = "stress_score"
ID_COLUMN = "ID"

# 모든 팀원이 같은 코드를 실행했을 때 최대한 같은 결과를 얻도록 난수 Seed를
# 고정합니다. Fold마다 모델 Seed에는 Fold 번호를 더해 완전히 동일한 모델만
# 반복해서 학습되는 것을 피합니다.
RANDOM_STATE = 42
N_SPLITS = 5

# 문제 설명상 stress_score는 0~1 범위입니다. 예측값을 이 범위로 자르면
# 실제 정답도 0~1일 때 MAE가 나빠질 수 없으며, 극단적인 외삽도 방지합니다.
# 이 범위는 테스트 데이터나 검증 정답에서 추정한 값이 아닙니다.
TARGET_MIN = 0.0
TARGET_MAX = 1.0

# 원본 파일이 있는 기본 폴더입니다. Path(__file__)을 사용하므로 VS Code에서
# 어떤 작업 폴더를 선택했는지와 관계없이 이 스크립트의 위치를 기준으로 찾습니다.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "open (3)"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    """명령행 옵션을 읽습니다.

    VS Code의 Run Python File 버튼으로 실행할 때는 기본값이 사용됩니다.
    터미널에서 실행할 때만 필요에 따라 데이터/결과 경로를 바꾸면 됩니다.
    """

    parser = argparse.ArgumentParser(
        description="Leakage-safe 5-Fold LightGBM baseline for stress prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="train.csv, test.csv, sample_submission.csv가 들어 있는 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="제출 파일과 OOF 예측을 저장할 폴더",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """상대 경로를 프로젝트 폴더 기준의 절대 경로로 변환합니다."""

    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """대회 데이터 세 파일을 불러오고 기본 구조를 점검합니다.

    파일이나 필수 열이 잘못된 상태로 학습을 진행하면 마지막 제출 단계에서야
    오류를 발견하기 쉽습니다. 따라서 시작할 때 명확한 오류 메시지를 냅니다.
    """

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "sample_submission.csv"

    required_files = [train_path, test_path, submission_path]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        missing_text = "\n  - ".join(missing_files)
        raise FileNotFoundError(f"다음 파일을 찾을 수 없습니다:\n  - {missing_text}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)

    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise ValueError(f"train.csv와 test.csv에 '{ID_COLUMN}' 열이 필요합니다.")
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"train.csv에 목표 변수 '{TARGET_COLUMN}' 열이 필요합니다.")
    if TARGET_COLUMN in test.columns:
        raise ValueError(
            f"test.csv에 '{TARGET_COLUMN}' 열이 있습니다. "
            "학습/테스트 파일을 반대로 지정하지 않았는지 확인하세요."
        )

    # ID와 TARGET을 제외하면 Train/Test 입력 열은 이름과 순서가 같아야 합니다.
    train_feature_columns = train.drop(columns=[ID_COLUMN, TARGET_COLUMN]).columns.tolist()
    test_feature_columns = test.drop(columns=[ID_COLUMN]).columns.tolist()
    if train_feature_columns != test_feature_columns:
        raise ValueError("train.csv와 test.csv의 입력 변수 구성이 서로 다릅니다.")

    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("중복 ID가 발견되었습니다. 원본 데이터를 확인하세요.")

    if len(test) != len(submission):
        raise ValueError("test.csv와 sample_submission.csv의 행 수가 다릅니다.")
    if not submission[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError(
            "test.csv와 sample_submission.csv의 ID 순서가 다릅니다. "
            "제출값이 다른 사람에게 연결되는 것을 막기 위해 실행을 중단합니다."
        )

    if train[TARGET_COLUMN].isna().any():
        raise ValueError(f"목표 변수 '{TARGET_COLUMN}'에 결측값이 있습니다.")

    return train, test, submission


def show_data_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """팀원이 실행 로그만 보고도 데이터 상태를 이해할 수 있게 요약합니다."""

    print("=" * 72)
    print("1. 데이터 확인")
    print("=" * 72)
    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    print(
        f"Target range: {train[TARGET_COLUMN].min():.2f} ~ "
        f"{train[TARGET_COLUMN].max():.2f}"
    )

    missing_summary = pd.DataFrame(
        {
            "train_missing": train.isna().sum(),
            "test_missing": test.isna().sum(),
        }
    )
    missing_summary = missing_summary.loc[missing_summary.max(axis=1) > 0]

    if missing_summary.empty:
        print("\n결측값이 없습니다.")
    else:
        print("\n결측값이 있는 열:")
        print(missing_summary.to_string())


def split_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """ID/TARGET을 모델 입력에서 분리합니다.

    ID는 샘플 식별자일 뿐 의미 있는 신체 정보가 아니므로 입력 변수에서 뺍니다.
    숫자처럼 보이는 ID 일련번호를 모델에 넣으면 우연한 순서 패턴에 과적합될 수
    있습니다.
    """

    train_ids = train[ID_COLUMN].copy()
    X = train.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    y = train[TARGET_COLUMN].astype(float).copy()
    X_test = test.drop(columns=[ID_COLUMN]).copy()
    return X, y, X_test, train_ids


def make_preprocessor(
    numeric_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    """Fold 학습 데이터만으로 학습할 전처리기를 만듭니다.

    중요: 이 함수는 전처리 방법만 정의합니다. 실제 median, 범주 목록 등은 아래
    교차검증 반복문에서 ``X_train_fold``에 대해 ``fit_transform``을 호출할 때
    정해집니다. 검증 Fold나 테스트 데이터의 통계값은 전처리 학습에 쓰지 않습니다.

    숫자형 변수
        결측값을 해당 Fold 학습 데이터의 중앙값으로 대체합니다. 중앙값은 평균보다
        극단값의 영향을 덜 받습니다.

    범주형 변수
        결측값을 최빈값으로 덮지 않고 ``__MISSING__``이라는 별도 상태로 남깁니다.
        이후 One-Hot Encoding을 적용합니다. ``handle_unknown='ignore'`` 덕분에
        검증/테스트에서 처음 보는 범주가 나와도 테스트 범주를 미리 학습하지 않고
        안전하게 변환할 수 있습니다.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

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
        # 이번 데이터는 작기 때문에 Dense 배열로 통일하면 디버깅이 쉽습니다.
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_model(fold: int) -> LGBMRegressor:
    """Fold별 LightGBM 회귀 모델을 생성합니다.

    MAE의 이론적인 최적 예측은 조건부 평균이 아니라 조건부 중앙값입니다.
    따라서 제곱오차용 기본 objective 대신 L1 objective를 사용합니다.

    현재 값들은 강한 튜닝 결과가 아니라 안정적인 시작점입니다. OOF 기준선을 만든
    뒤 num_leaves, min_child_samples, feature_fraction, 정규화 강도 등을 한 번에
    하나씩 비교하는 것이 좋습니다.
    """

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=3_000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
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


def train_with_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """누수 없는 5-Fold 학습, OOF 예측, 테스트 Fold 앙상블을 수행합니다.

    OOF(Out-Of-Fold) 예측은 각 샘플을 학습에 사용하지 않은 모델로 예측한 값입니다.
    따라서 전체 OOF MAE는 제출 전에 모델의 일반화 성능을 비교하는 기준이 됩니다.

    테스트 데이터는 각 Fold에서 학습된 전처리기와 모델로 예측한 후 평균합니다.
    이 방식은 단일 Train/Validation 분할보다 Seed와 분할 운의 영향을 줄여줍니다.
    """

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    print("\n" + "=" * 72)
    print("2. 입력 변수 구성")
    print("=" * 72)
    print(f"숫자형 변수 ({len(numeric_columns)}개): {numeric_columns}")
    print(f"범주형 변수 ({len(categorical_columns)}개): {categorical_columns}")

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 각 Train 샘플은 정확히 한 번 검증 Fold에 포함되므로 길이는 Train과 같습니다.
    oof_predictions = np.zeros(len(X), dtype=float)

    # 열 하나가 Fold 하나의 테스트 예측입니다. 마지막에 행 방향 평균을 냅니다.
    test_fold_predictions = np.zeros((len(X_test), N_SPLITS), dtype=float)
    fold_records: list[dict[str, float | int]] = []

    print("\n" + "=" * 72)
    print("3. 5-Fold 교차검증 학습")
    print("=" * 72)

    for fold, (train_index, valid_index) in enumerate(kfold.split(X), start=1):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_valid_fold = y.iloc[valid_index]

        # 매 Fold마다 새 전처리기를 만듭니다. 이전 Fold의 통계값/범주가 다음 Fold로
        # 넘어가면 검증 점수가 낙관적으로 보일 수 있기 때문입니다.
        preprocessor = make_preprocessor(numeric_columns, categorical_columns)

        # fit_transform은 오직 Fold 학습 데이터에만 호출합니다.
        X_train_processed = preprocessor.fit_transform(X_train_fold)

        # 검증과 테스트는 Fold 학습 데이터로 이미 학습된 전처리기를 사용합니다.
        X_valid_processed = preprocessor.transform(X_valid_fold)
        X_test_processed = preprocessor.transform(X_test)

        model = make_model(fold)
        model.fit(
            X_train_processed,
            y_train_fold,
            # LightGBM 4.7부터 권장되는 검증 데이터 인자입니다. 검증 데이터는
            # early stopping에만 사용되며 모델의 입력 전처리 학습에는 쓰이지 않습니다.
            eval_X=X_valid_processed,
            eval_y=y_valid_fold,
            eval_metric="mae",
            callbacks=[
                # 150회 연속으로 MAE가 좋아지지 않으면 학습을 멈춥니다.
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                # Fold 결과만 간결하게 출력하기 위해 매 iteration 로그는 끕니다.
                lgb.log_evaluation(period=0),
            ],
        )

        # best_iteration_을 명시해 early stopping에서 가장 좋았던 트리 수로 예측합니다.
        best_iteration = int(model.best_iteration_ or model.n_estimators)
        valid_prediction = model.predict(
            X_valid_processed, num_iteration=best_iteration
        )
        test_prediction = model.predict(X_test_processed, num_iteration=best_iteration)

        # 대회에서 가능한 점수 범위로 자릅니다. 소수 둘째 자리 반올림은 OOF로
        # 유리함이 확인되지 않았으므로 적용하지 않습니다.
        valid_prediction = np.clip(valid_prediction, TARGET_MIN, TARGET_MAX)
        test_prediction = np.clip(test_prediction, TARGET_MIN, TARGET_MAX)

        oof_predictions[valid_index] = valid_prediction
        test_fold_predictions[:, fold - 1] = test_prediction

        fold_mae = mean_absolute_error(y_valid_fold, valid_prediction)
        fold_records.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "best_iteration": best_iteration,
                "mae": fold_mae,
            }
        )
        print(
            f"Fold {fold}/{N_SPLITS} | "
            f"best_iteration={best_iteration:4d} | MAE={fold_mae:.6f}"
        )

    overall_mae = mean_absolute_error(y, oof_predictions)
    print("-" * 72)
    print(f"전체 OOF MAE: {overall_mae:.6f}")

    # 다섯 모델의 예측 평균. MAE에서 평균 앙상블이 항상 최적이라는 뜻은 아니지만,
    # 동일 모델의 Fold 평균은 일반적으로 분산을 낮추는 안정적인 베이스라인입니다.
    test_predictions = test_fold_predictions.mean(axis=1)
    test_predictions = np.clip(test_predictions, TARGET_MIN, TARGET_MAX)

    fold_results = pd.DataFrame(fold_records)
    return oof_predictions, test_predictions, fold_results


def save_results(
    output_dir: Path,
    submission_template: pd.DataFrame,
    train_ids: pd.Series,
    y: pd.Series,
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
    fold_results: pd.DataFrame,
) -> None:
    """제출 파일, OOF 예측, Fold별 점수를 CSV로 저장합니다."""

    output_dir.mkdir(parents=True, exist_ok=True)

    submission = submission_template.copy()
    submission[TARGET_COLUMN] = test_predictions

    oof_result = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            f"actual_{TARGET_COLUMN}": y,
            f"predicted_{TARGET_COLUMN}": oof_predictions,
            "absolute_error": np.abs(y.to_numpy() - oof_predictions),
        }
    )

    submission_path = output_dir / "baseline_lightgbm_submission.csv"
    oof_path = output_dir / "baseline_lightgbm_oof.csv"
    fold_path = output_dir / "baseline_lightgbm_cv_results.csv"

    submission.to_csv(submission_path, index=False)
    oof_result.to_csv(oof_path, index=False)
    fold_results.to_csv(fold_path, index=False)

    print("\n" + "=" * 72)
    print("4. 결과 저장 완료")
    print("=" * 72)
    print(f"제출 파일      : {submission_path}")
    print(f"OOF 예측 파일  : {oof_path}")
    print(f"Fold 결과 파일 : {fold_path}")
    print("\nDACON에는 baseline_lightgbm_submission.csv 파일을 제출하면 됩니다.")


def main() -> None:
    """데이터 로드부터 결과 저장까지 전체 파이프라인을 실행합니다."""

    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)

    train, test, submission = load_data(data_dir)
    show_data_summary(train, test)

    X, y, X_test, train_ids = split_features(train, test)
    oof_predictions, test_predictions, fold_results = train_with_cross_validation(
        X, y, X_test
    )

    save_results(
        output_dir=output_dir,
        submission_template=submission,
        train_ids=train_ids,
        y=y,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        fold_results=fold_results,
    )


if __name__ == "__main__":
    main()
