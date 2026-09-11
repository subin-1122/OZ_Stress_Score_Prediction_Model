"""
스트레스 점수 예측 - 실험 v2
================================

목적
----
하루 한 번만 제출할 수 있으므로, 리더보드에 여러 파일을 시험하지 않고
train.csv 내부 교차검증으로 후보를 충분히 비교한 뒤 최종 파일 하나만 만든다.

이번 실험에서 채택한 핵심
-------------------------
1. 파생변수는 BMI 하나만 사용한다.
2. ExtraTrees의 max_features를 1로 두어 나무 사이 다양성을 크게 만든다.
3. 20-Fold 교차검증을 서로 다른 세 개의 seed로 반복한다.
4. 변수 제거 조합 두 개를 각각 학습한 뒤 50:50으로 평균한다.
5. OOF 예측에서 MAE가 실제로 개선될 때만 선형 보정을 적용한다.
6. 최종 OOF MAE가 0.11 미만일 때만 제출 파일을 저장한다.
7. 이전 제출에서 관측한 OOF와 Public 점수 차이로 예상 DACON 점수를 표시한다.

데이터 누수 방지
----------------
- 결측치 중앙값, 범주형 결측 토큰, 원-핫 인코딩은 매 fold의 학습 부분으로만 fit한다.
- test.csv는 변환과 예측에만 사용하며 통계량 계산이나 모델 선택에 사용하지 않는다.
- 최종 후보 선택도 train.csv의 OOF MAE만 이용한다.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# 실험 설정
# ---------------------------------------------------------------------------
TARGET = "stress_score"
ID_COLUMN = "ID"

# 20-Fold를 세 번 반복하므로 각 변수 조합마다 60개 모델을 학습한다.
# 두 변수 조합을 사용하므로 전체 ExtraTrees 모델 수는 120개다.
CV_SEEDS = (42, 123, 2026)
N_SPLITS = 20
N_ESTIMATORS = 1000

# 앞선 ablation에서 안정적으로 좋았던 두 조합이다.
# Variant A는 sleep_pattern을, Variant B는 smoke_status와 이완기 혈압을
# 추가로 제외한다. 서로 실수가 조금 다른 두 모델을 평균하는 것이 목적이다.
VARIANTS: dict[str, tuple[str, ...]] = {
    "A_drop_work_age_sleep": (
        "mean_working",
        "age",
        "sleep_pattern",
    ),
    "B_drop_work_age_smoke_dia": (
        "mean_working",
        "age",
        "smoke_status",
        "diastolic_blood_pressure",
    ),
}

# 보정 효과가 이 값보다 작으면 우연일 가능성을 고려해 보정하지 않는다.
CALIBRATION_MIN_GAIN = 0.0005
CALIBRATION_SEED = 910

# 하루 한 번뿐인 제출 기회를 보호하기 위한 강제 기준이다.
# 0.110000과 같은 값은 통과가 아니며 반드시 0.110000보다 작아야 한다.
SUBMISSION_OOF_THRESHOLD = 0.11

# 첫 제출에서 실제로 확인한 값이다.
# 관측 차이가 양수이므로 DACON 점수가 내부 OOF보다 조금 나빴다는 뜻이다.
# 제출 사례가 한 번뿐이므로 확정적인 보정이 아니라 임시 추정치로만 사용한다.
REFERENCE_V1_OOF_MAE = 0.17739770000000007
REFERENCE_V1_PUBLIC_MAE = 0.17982
OBSERVED_PUBLIC_GAP = REFERENCE_V1_PUBLIC_MAE - REFERENCE_V1_OOF_MAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="반복 20-Fold ExtraTrees 앙상블로 v2 제출 파일을 생성합니다."
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
        help="결과 파일을 저장할 폴더. 기본값은 프로젝트의 outputs 폴더",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="코드 동작 확인용: 5-Fold, seed 1개, 나무 100개로 축소",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """
    VS Code의 실행 위치가 프로젝트 루트이든 experiment 폴더이든 작동하도록
    현재 파일 위치와 현재 작업 폴더의 상위 경로를 차례로 확인한다.
    """
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
            f"데이터 폴더에 다음 파일이 없습니다: {missing}\n확인 경로: {data_dir}"
        )
    return data_dir


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    원본 CSV를 읽고 train/test/submission의 기본 관계를 먼저 검증한다.
    여기서는 값을 변경하지 않는다.
    """
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if TARGET not in train.columns:
        raise ValueError(f"train.csv에 타깃 열 {TARGET!r}이 없습니다.")
    if TARGET in test.columns:
        raise ValueError(f"test.csv에 타깃 열 {TARGET!r}이 있으면 안 됩니다.")
    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise ValueError(f"train/test에 ID 열 {ID_COLUMN!r}이 필요합니다.")
    if len(test) != len(sample):
        raise ValueError("test.csv와 sample_submission.csv의 행 수가 다릅니다.")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("test.csv와 sample_submission.csv의 ID 순서가 다릅니다.")

    return train, test, sample


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 행의 기존 height와 weight를 이용해 BMI를 계산한다.

    BMI = 체중(kg) / 키(m)의 제곱

    별도의 외부 값이나 사람이 직접 입력한 값이 아니다. 키가 0이거나
    height/weight가 결측이면 BMI도 결측으로 두고, 이후 fold 내부 중앙값으로
    처리한다.
    """
    result = df.copy()
    height_m = pd.to_numeric(result["height"], errors="coerce") / 100.0
    weight_kg = pd.to_numeric(result["weight"], errors="coerce")
    valid_height = height_m.where(height_m > 0)
    result["bmi"] = weight_kg / valid_height.pow(2)
    result["bmi"] = result["bmi"].replace([np.inf, -np.inf], np.nan)
    return result


def make_variant_features(
    base: pd.DataFrame,
    drop_columns: tuple[str, ...],
) -> pd.DataFrame:
    """ID와 지정 변수들을 제외하고 BMI를 추가한 입력 행렬을 만든다."""
    missing_drop_columns = [column for column in drop_columns if column not in base]
    if missing_drop_columns:
        raise ValueError(f"제거하려는 열이 데이터에 없습니다: {missing_drop_columns}")

    features = base.drop(columns=[ID_COLUMN, *drop_columns], errors="raise")
    return add_bmi(features)


def build_preprocessor(x_train_fold: pd.DataFrame) -> ColumnTransformer:
    """
    해당 fold 학습 데이터의 자료형을 기준으로 전처리기를 만든다.

    숫자형:
      - 학습 fold 중앙값으로 결측치를 채운다.
      - 결측이 있었던 열은 결측 여부 indicator도 추가한다.

    범주형:
      - 학습 fold에서 결측을 문자열 MISSING으로 바꾼다.
      - 학습 fold로만 원-핫 인코더를 fit한다.
      - 검증/test에서 처음 본 범주는 모두 0으로 안전하게 처리한다.
    """
    categorical_columns = x_train_fold.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_columns = [
        column for column in x_train_fold.columns if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            )
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="MISSING"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(random_state: int, n_estimators: int) -> ExtraTreesRegressor:
    """
    앞선 자체 실험에서 가장 안정적이었던 ExtraTrees 설정.

    max_features=1은 각 분기에서 후보 변수 하나만 무작위로 확인하게 해
    트리 사이의 다양성을 키운다. 모든 행을 사용하는 ExtraTrees 특성상
    작은 데이터에서도 빠르고, 반복 CV 앙상블과 잘 맞았다.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        criterion="squared_error",
        max_features=1,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )


def run_repeated_cv(
    variant_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    seeds: tuple[int, ...],
    n_splits: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """
    반복 K-Fold OOF와 test 예측을 생성한다.

    OOF는 각 행을 학습에 사용하지 않은 모델의 예측만 모은 값이다.
    test 예측은 모든 fold 모델의 결과를 단순 평균한다.
    """
    oof_sum = np.zeros(len(x), dtype=float)
    oof_count = np.zeros(len(x), dtype=np.int16)
    test_prediction_sum = np.zeros(len(x_test), dtype=float)
    fold_records: list[dict[str, object]] = []

    total_models = len(seeds) * n_splits
    completed_models = 0

    for repeat_number, seed in enumerate(seeds, start=1):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_number, (train_index, valid_index) in enumerate(
            splitter.split(x),
            start=1,
        ):
            started_at = time.perf_counter()
            x_train_fold = x.iloc[train_index]
            x_valid_fold = x.iloc[valid_index]
            y_train_fold = y.iloc[train_index]
            y_valid_fold = y.iloc[valid_index]

            # 중요: 전처리기는 오직 현재 학습 fold로만 fit한다.
            preprocessor = build_preprocessor(x_train_fold)
            x_train_transformed = preprocessor.fit_transform(x_train_fold)
            x_valid_transformed = preprocessor.transform(x_valid_fold)
            x_test_transformed = preprocessor.transform(x_test)

            model_seed = seed * 100 + fold_number
            model = build_model(model_seed, n_estimators)
            model.fit(x_train_transformed, y_train_fold)

            valid_prediction = model.predict(x_valid_transformed)
            test_prediction = model.predict(x_test_transformed)

            oof_sum[valid_index] += valid_prediction
            oof_count[valid_index] += 1
            test_prediction_sum += test_prediction

            fold_mae = mean_absolute_error(y_valid_fold, valid_prediction)
            completed_models += 1
            elapsed = time.perf_counter() - started_at
            fold_records.append(
                {
                    "variant": variant_name,
                    "repeat": repeat_number,
                    "cv_seed": seed,
                    "fold": fold_number,
                    "train_rows": len(train_index),
                    "valid_rows": len(valid_index),
                    "fold_mae": fold_mae,
                    "seconds": elapsed,
                }
            )
            print(
                f"[{variant_name}] {completed_models:>2}/{total_models} "
                f"| seed={seed} fold={fold_number:>2} "
                f"| MAE={fold_mae:.6f} | {elapsed:.1f}s",
                flush=True,
            )

    expected_count = len(seeds)
    if not np.all(oof_count == expected_count):
        raise RuntimeError(
            "OOF 예측 횟수가 예상과 다릅니다. "
            f"예상={expected_count}, 실제 범위={oof_count.min()}~{oof_count.max()}"
        )

    oof_prediction = oof_sum / oof_count
    test_prediction = test_prediction_sum / total_models
    return oof_prediction, test_prediction, fold_records


def crossfit_linear_calibration(
    prediction: np.ndarray,
    y: pd.Series,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    MAE에 맞춘 1차 선형 보정을 교차검증 방식으로 평가한다.

    QuantileRegressor의 quantile=0.5는 절대오차 손실의 중앙값 회귀에 해당한다.
    입력은 모델 예측 하나뿐이므로 학습되는 값은 기울기와 절편이다.
    """
    prediction_2d = prediction.reshape(-1, 1)
    calibrated_oof = np.zeros_like(prediction, dtype=float)
    records: list[dict[str, float]] = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=CALIBRATION_SEED)

    for fold_number, (train_index, valid_index) in enumerate(
        splitter.split(prediction_2d),
        start=1,
    ):
        calibrator = QuantileRegressor(
            quantile=0.5,
            alpha=0.0,
            solver="highs",
        )
        calibrator.fit(prediction_2d[train_index], y.iloc[train_index])
        calibrated_oof[valid_index] = calibrator.predict(
            prediction_2d[valid_index]
        )
        records.append(
            {
                "fold": float(fold_number),
                "intercept": float(calibrator.intercept_),
                "slope": float(calibrator.coef_[0]),
            }
        )

    return np.clip(calibrated_oof, 0.0, 1.0), records


def fit_final_calibrator(
    oof_prediction: np.ndarray,
    y: pd.Series,
    test_prediction: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """전체 OOF 관계로 최종 기울기와 절편을 학습해 test 예측에 적용한다."""
    calibrator = QuantileRegressor(
        quantile=0.5,
        alpha=0.0,
        solver="highs",
    )
    calibrator.fit(oof_prediction.reshape(-1, 1), y)
    calibrated_test = calibrator.predict(test_prediction.reshape(-1, 1))
    return (
        np.clip(calibrated_test, 0.0, 1.0),
        float(calibrator.intercept_),
        float(calibrator.coef_[0]),
    )


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    """저장 전에 대회 제출 형식과 값 범위를 강하게 검사한다."""
    expected_columns = [ID_COLUMN, TARGET]
    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            f"제출 열이 잘못되었습니다. 예상={expected_columns}, "
            f"실제={submission.columns.tolist()}"
        )
    if len(submission) != len(sample):
        raise ValueError("제출 파일의 행 수가 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 파일의 ID 또는 순서가 sample_submission과 다릅니다.")
    if submission[TARGET].isna().any():
        raise ValueError("제출 예측에 결측값이 있습니다.")
    if not np.isfinite(submission[TARGET]).all():
        raise ValueError("제출 예측에 무한대가 있습니다.")
    if not submission[TARGET].between(0.0, 1.0).all():
        raise ValueError("제출 예측이 허용 범위 0~1을 벗어났습니다.")


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

    if args.quick:
        seeds = (42,)
        n_splits = 5
        n_estimators = 100
        run_label = "quick_check"
    else:
        seeds = CV_SEEDS
        n_splits = N_SPLITS
        n_estimators = N_ESTIMATORS
        run_label = "full"

    print("=" * 78)
    print("스트레스 점수 예측 - ExtraTrees v2")
    print(f"프로젝트: {project_root}")
    print(f"데이터:   {data_dir}")
    print(f"출력:     {output_dir}")
    print(
        f"설정:     {n_splits}-Fold x {len(seeds)} seeds, "
        f"{n_estimators} trees, mode={run_label}"
    )
    print("=" * 78)

    train, test, sample = load_data(data_dir)
    y = pd.to_numeric(train[TARGET], errors="raise")
    train_base = train.drop(columns=[TARGET])
    test_base = test.copy()

    variant_oof: dict[str, np.ndarray] = {}
    variant_test: dict[str, np.ndarray] = {}
    all_fold_records: list[dict[str, object]] = []
    score_records: list[dict[str, object]] = []

    for variant_name, drop_columns in VARIANTS.items():
        print(f"\n변수 조합 시작: {variant_name}")
        print(f"제외 열: {list(drop_columns)}")
        x = make_variant_features(train_base, drop_columns)
        x_test = make_variant_features(test_base, drop_columns)

        oof_prediction, test_prediction, fold_records = run_repeated_cv(
            variant_name=variant_name,
            x=x,
            y=y,
            x_test=x_test,
            seeds=seeds,
            n_splits=n_splits,
            n_estimators=n_estimators,
        )
        variant_oof[variant_name] = oof_prediction
        variant_test[variant_name] = test_prediction
        all_fold_records.extend(fold_records)

        variant_mae = mean_absolute_error(y, oof_prediction)
        score_records.append(
            {
                "candidate": variant_name,
                "oof_mae": variant_mae,
                "selected": False,
                "note": "반복 CV 평균 OOF",
            }
        )
        print(f"{variant_name} 전체 반복 OOF MAE: {variant_mae:.6f}")

    # 두 변수 조합의 비중은 탐색 결과가 안정적이었던 단순 50:50으로 고정한다.
    variant_names = list(VARIANTS)
    blend_oof = np.mean(
        np.column_stack([variant_oof[name] for name in variant_names]),
        axis=1,
    )
    blend_test = np.mean(
        np.column_stack([variant_test[name] for name in variant_names]),
        axis=1,
    )
    raw_blend_mae = mean_absolute_error(y, blend_oof)
    score_records.append(
        {
            "candidate": "blend_A50_B50_raw",
            "oof_mae": raw_blend_mae,
            "selected": False,
            "note": "A와 B 예측의 동일 가중 평균",
        }
    )

    calibrated_oof, calibration_fold_records = crossfit_linear_calibration(
        blend_oof,
        y,
    )
    calibrated_mae = mean_absolute_error(y, calibrated_oof)
    calibration_gain = raw_blend_mae - calibrated_mae

    # 보정은 OOF에서 최소 개선 폭을 넘을 때만 채택한다.
    use_calibration = calibration_gain >= CALIBRATION_MIN_GAIN
    if use_calibration:
        final_test_prediction, final_intercept, final_slope = fit_final_calibrator(
            blend_oof,
            y,
            blend_test,
        )
        final_oof_prediction = calibrated_oof
        selected_name = "blend_A50_B50_crossfit_calibrated"
        selected_mae = calibrated_mae
    else:
        final_test_prediction = np.clip(blend_test, 0.0, 1.0)
        final_oof_prediction = np.clip(blend_oof, 0.0, 1.0)
        final_intercept = 0.0
        final_slope = 1.0
        selected_name = "blend_A50_B50_raw"
        selected_mae = raw_blend_mae

    score_records.append(
        {
            "candidate": "blend_A50_B50_crossfit_calibrated",
            "oof_mae": calibrated_mae,
            "selected": use_calibration,
            "note": (
                f"교차 보정 평가, raw 대비 개선={calibration_gain:.6f}, "
                f"채택 기준={CALIBRATION_MIN_GAIN:.6f}"
            ),
        }
    )

    # selected 열은 최종 선택된 후보 한 개만 True가 되도록 정리한다.
    for record in score_records:
        record["selected"] = record["candidate"] == selected_name

    # 이전 제출에서 관측한 차이를 더해 다음 DACON 점수를 보수적으로 추정한다.
    # 이 값은 모델 예측값 자체를 변경하지 않으며, 제출 판단용 참고 지표다.
    estimated_public_mae = selected_mae + OBSERVED_PUBLIC_GAP

    # -----------------------------------------------------------------------
    # 제출 파일 생성 게이트
    # -----------------------------------------------------------------------
    # 자체 검증 점수가 기준을 통과하지 못하면 CSV를 만들지 않고 종료한다.
    # 따라서 실수로 기준 미달 모델을 DACON에 제출할 가능성을 줄일 수 있다.
    passes_submission_gate = selected_mae < SUBMISSION_OOF_THRESHOLD
    if not passes_submission_gate:
        print("\n" + "=" * 78)
        print("제출 파일 생성 안 함")
        print(f"최종 OOF MAE: {selected_mae:.6f}")
        print(f"예상 DACON MAE: {estimated_public_mae:.6f}")
        print(f"이전 관측 차이:  +{OBSERVED_PUBLIC_GAP:.7f}")
        print(f"필수 기준:     {SUBMISSION_OOF_THRESHOLD:.6f} 미만")
        print("기준 미달이므로 submission CSV와 결과 파일을 저장하지 않습니다.")
        print("=" * 78)
        return

    submission = sample[[ID_COLUMN]].copy()
    submission[TARGET] = final_test_prediction
    validate_submission(submission, sample)

    oof_result = train[[ID_COLUMN, TARGET]].copy()
    for name in variant_names:
        oof_result[f"oof_{name}"] = variant_oof[name]
    oof_result["oof_blend_raw"] = blend_oof
    oof_result["oof_blend_crossfit_calibrated"] = calibrated_oof
    oof_result["oof_selected"] = final_oof_prediction
    oof_result["absolute_error_selected"] = np.abs(
        y.to_numpy() - final_oof_prediction
    )

    configuration = {
        "run_mode": run_label,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm_installed": lightgbm.__version__,
        "rows_train": len(train),
        "rows_test": len(test),
        "n_splits": n_splits,
        "cv_seeds": json.dumps(seeds),
        "n_estimators": n_estimators,
        "max_features": 1,
        "variants": json.dumps(VARIANTS, ensure_ascii=False),
        "blend_weight_A": 0.5,
        "blend_weight_B": 0.5,
        "raw_blend_oof_mae": raw_blend_mae,
        "calibrated_crossfit_oof_mae": calibrated_mae,
        "calibration_gain": calibration_gain,
        "calibration_min_gain": CALIBRATION_MIN_GAIN,
        "calibration_applied": use_calibration,
        "final_calibration_intercept": final_intercept,
        "final_calibration_slope": final_slope,
        "selected_candidate": selected_name,
        "selected_oof_mae": selected_mae,
        "submission_oof_threshold": SUBMISSION_OOF_THRESHOLD,
        "submission_gate_passed": passes_submission_gate,
        "reference_v1_oof_mae": REFERENCE_V1_OOF_MAE,
        "previous_public_score_v1": REFERENCE_V1_PUBLIC_MAE,
        "observed_public_minus_oof_gap": OBSERVED_PUBLIC_GAP,
        "estimated_public_mae": estimated_public_mae,
    }

    submission_path = output_dir / "experiment_v2_submission.csv"
    oof_path = output_dir / "experiment_v2_oof.csv"
    scores_path = output_dir / "experiment_v2_scores.csv"
    folds_path = output_dir / "experiment_v2_fold_results.csv"
    config_path = output_dir / "experiment_v2_configuration.csv"

    submission.to_csv(submission_path, index=False)
    oof_result.to_csv(oof_path, index=False)
    pd.DataFrame(score_records).to_csv(scores_path, index=False)
    pd.DataFrame(all_fold_records).to_csv(folds_path, index=False)
    pd.DataFrame([configuration]).to_csv(config_path, index=False)

    print("\n" + "=" * 78)
    print("실험 완료")
    print(f"Raw blend OOF MAE:        {raw_blend_mae:.6f}")
    print(f"Calibrated OOF MAE:       {calibrated_mae:.6f}")
    print(f"Calibration gain:         {calibration_gain:.6f}")
    print(f"Calibration applied:      {use_calibration}")
    print(f"Selected candidate:       {selected_name}")
    print(f"Selected OOF MAE:         {selected_mae:.6f}")
    print(f"Estimated DACON MAE:      {estimated_public_mae:.6f}")
    print(f"Observed Public-OOF gap:  +{OBSERVED_PUBLIC_GAP:.7f}")
    print(f"Submission threshold:     < {SUBMISSION_OOF_THRESHOLD:.6f}")
    print(f"Submission gate passed:   {passes_submission_gate}")
    print(f"Final calibrator:         y = {final_slope:.6f} * pred + {final_intercept:.6f}")
    print(f"제출 후보: {submission_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
