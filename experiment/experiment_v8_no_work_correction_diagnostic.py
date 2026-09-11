"""
mean_working 보정 효과를 분리하기 위한 진단용 제출 파일
=======================================================

중요
----
이 파일은 최저 점수를 목표로 하는 제출 모델이 아니다. 현재 파이프라인에서
mean_working 잔차 보정만 제거하고 다음 두 요소만 남긴다.

1. 레코드 연결(학습형 연결 + v4 결정 규칙)
2. train target의 0.01 간격으로 snapping

DACON Public 점수를 얻으면 아래 두 gap을 비교할 수 있다.

- 보정 포함: 현재 snapping 모델의 Public MAE - 보정 포함 OOF MAE
- 보정 제외: 이 진단 제출의 Public MAE - 보정 제외 OOF MAE

두 gap이 비슷하면 Public-OOF 차이는 mean_working 보정보다 데이터 분할이나
기본 모델에서 발생했을 가능성이 크다. 보정 제외 gap이 크게 줄면 보정 레이어의
일반화 차이가 Public gap에 일부 기여했다고 해석할 수 있다.

누수 방지
---------
- 레코드 연결과 보정표는 train만으로 학습한다.
- test는 연결 후보 탐색과 예측 변환에만 사용한다.
- test의 통계량이나 정답은 사용하지 않는다.
- 제거하는 보정값도 train OOF 잔차로 만든 기존 보정표에서 계산한다.

실행
----
프로젝트 루트에서 다음 명령을 실행한다.

    .venv/bin/python experiment/experiment_v8_no_work_correction_diagnostic.py

VS Code에서는 이 파일을 열고 오른쪽 위의 "Python 파일 실행" 버튼을 누른다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


TARGET = "stress_score"
ID_COLUMN = "ID"
GRID_STEP = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mean_working 보정만 제거한 진단용 DACON 파일을 만듭니다."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def find_project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "open (3)" / "train.csv").exists():
                return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(prediction: np.ndarray) -> np.ndarray:
    """예측값을 가장 가까운 0.01 target 격자로 이동한다."""
    return np.clip(np.rint(prediction / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def make_oof_comparison(
    v4,
    train: pd.DataFrame,
    oof: pd.DataFrame,
) -> list[dict[str, float | int]]:
    """v4의 다섯 연결 split에서 보정 포함/제외 OOF를 비교한다."""
    y = train[TARGET].to_numpy(float)
    base_oof = oof["base_oof"].to_numpy(float)
    learned_label = oof["link_label"].to_numpy(float)
    learned_mask = oof["linked"].astype(bool).to_numpy()
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(features, numeric, categorical)

    results: list[dict[str, float | int]] = []
    for seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            rule_pairs, y, len(train), seed
        )
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]

        # 진단 후보: base OOF에 연결값만 덮어쓰고 snapping한다.
        no_work_prediction = base_oof.copy()
        no_work_prediction[union_mask] = union_label[union_mask]
        no_work_prediction = snap(no_work_prediction)

        # 비교 대상: 기존 mean_working 보정을 포함한 v4 OOF를 snapping한다.
        correction = v4.crossfit_work_correction(
            train, y, base_oof, union_mask
        )
        with_work_prediction = np.clip(
            base_oof + v4.WORK_CORRECTION_ALPHA * correction,
            0.0,
            1.0,
        )
        with_work_prediction[union_mask] = union_label[union_mask]
        with_work_prediction = snap(with_work_prediction)

        no_work_mae = mean_absolute_error(y, no_work_prediction)
        with_work_mae = mean_absolute_error(y, with_work_prediction)
        results.append(
            {
                "seed": int(seed),
                "union_coverage": float(union_mask.mean()),
                "with_work_snap_mae": float(with_work_mae),
                "no_work_snap_mae": float(no_work_mae),
                "work_correction_oof_gain": float(no_work_mae - with_work_mae),
            }
        )
    return results


def make_test_masks(
    v3,
    v4,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """학습형, 결정 규칙, 두 방법의 합집합 test 연결 마스크를 만든다."""
    train_features = train.drop(columns=[ID_COLUMN, TARGET])
    test_features = test.drop(columns=[ID_COLUMN])
    numeric = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical = train_features.select_dtypes(exclude=np.number).columns.tolist()
    blocks = v3.linkage_blocks(categorical)
    train_pairs = v3.train_candidate_pairs(train_features, blocks)
    test_pairs = v3.train_test_candidate_pairs(
        train_features, test_features, blocks
    )
    learned_result = v3.make_test_link_prediction(
        train_features,
        test_features,
        y,
        train_pairs,
        test_pairs,
        numeric,
        categorical,
    )
    learned_mask = v3.high_confidence_mask(learned_result)
    deterministic_label = v4.make_test_rule_labels(
        train_features,
        test_features,
        y,
        numeric,
        categorical,
    )
    deterministic_mask = np.isfinite(deterministic_label)
    return learned_mask, deterministic_mask, learned_mask | deterministic_mask


def recover_base_test_prediction(
    v3,
    train: pd.DataFrame,
    test: pd.DataFrame,
    oof: pd.DataFrame,
    current_v4_submission: pd.DataFrame,
    union_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    현재 v4 제출의 unlinked 행에서 mean_working 보정값을 정확히 뺀다.

    기존 v3는 ``clip(base_test + 0.75 * correction)``을 저장했다. 이 데이터에서
    unlinked 예측은 0이나 1 경계에 닿지 않아 clipping 정보 손실이 없음을 검사한
    뒤 동일 보정값을 빼면 base_test를 정확히 복원할 수 있다. 이 방법은 300개의
    100-fold 모델을 다시 학습할 필요가 없고 기존 제출과 수치적으로 일치한다.
    """
    y = train[TARGET].to_numpy(float)
    base_oof = oof["base_oof"].to_numpy(float)
    learned_oof_mask = oof["linked"].astype(bool).to_numpy()
    work_table, fallback = v3.fit_final_work_table(
        train, y, base_oof, learned_oof_mask
    )
    correction = (
        v3.work_key(test["mean_working"])
        .map(work_table)
        .fillna(fallback)
        .to_numpy(float)
    )

    current = current_v4_submission[TARGET].to_numpy(float)
    recovered = current.copy()
    recovered[~union_mask] = (
        current[~union_mask]
        - v3.WORK_CORRECTION_ALPHA * correction[~union_mask]
    )

    # ExtraTrees 회귀 예측은 train target 범위인 0~1 안에 있어야 한다. 범위를
    # 벗어나면 기존 clip 때문에 원래 값을 정확히 복원할 수 없다는 뜻이다.
    if ((recovered[~union_mask] < 0) | (recovered[~union_mask] > 1)).any():
        raise ValueError(
            "일부 base_test 값이 0~1 밖으로 복원되어 clipping 정보 손실이 있습니다."
        )
    reapplied = np.clip(
        recovered[~union_mask]
        + v3.WORK_CORRECTION_ALPHA * correction[~union_mask],
        0.0,
        1.0,
    )
    if not np.allclose(reapplied, current[~union_mask], atol=1e-12):
        raise ValueError("mean_working 보정 제거/복원 검사가 실패했습니다.")
    return recovered, correction


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    if list(submission.columns) != list(sample.columns):
        raise ValueError("제출 파일 열이 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 ID 또는 순서가 sample_submission과 다릅니다.")
    prediction = submission[TARGET].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("예측값이 0~1 범위를 벗어났습니다.")
    if not np.allclose(
        prediction / GRID_STEP,
        np.rint(prediction / GRID_STEP),
        atol=1e-9,
    ):
        raise ValueError("예측값이 0.01 target 격자에 맞지 않습니다.")


def main() -> None:
    args = parse_args()
    root = find_project_root()
    data_dir = args.data_dir.resolve() if args.data_dir else root / "open (3)"
    output_dir = args.output_dir.resolve() if args.output_dir else root / "outputs"
    v3 = load_module(
        "experiment_v3",
        root / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        root / "experiment/experiment_v4_deterministic_union.py",
    )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    oof = pd.read_csv(output_dir / "best_record_linkage_oof.csv")
    current_v4 = pd.read_csv(output_dir / "best_union_submission.csv")
    current_snap = pd.read_csv(
        output_dir / "best_union_grid_snap_submission.csv"
    )
    y = train[TARGET].to_numpy(float)

    repeated = make_oof_comparison(v4, train, oof)
    with_work_mean = float(
        np.mean([row["with_work_snap_mae"] for row in repeated])
    )
    no_work_mean = float(
        np.mean([row["no_work_snap_mae"] for row in repeated])
    )

    learned_mask, deterministic_mask, union_mask = make_test_masks(
        v3, v4, train, test, y
    )
    recovered_base, correction = recover_base_test_prediction(
        v3, train, test, oof, current_v4, union_mask
    )
    diagnostic_prediction = recovered_base.copy()
    # 연결된 행은 current_v4에 이미 train 점수가 들어 있으므로 그대로 유지한다.
    diagnostic_prediction[union_mask] = current_v4.loc[
        union_mask, TARGET
    ].to_numpy(float)
    diagnostic_prediction = snap(diagnostic_prediction)

    submission = current_v4.copy()
    submission[TARGET] = diagnostic_prediction
    validate_submission(submission, sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "diagnostic_link_snap_no_work_submission.csv"
    metrics_path = output_dir / "diagnostic_link_snap_no_work_metrics.json"
    submission.to_csv(submission_path, index=False, float_format="%.2f")

    current_snap_prediction = current_snap[TARGET].to_numpy(float)
    metrics = {
        "purpose": "Public-OOF gap diagnosis; not a best-score candidate",
        "method": "learned linkage + deterministic linkage + 0.01 snap; no mean_working correction",
        "with_work_snap_oof_mae_mean": with_work_mean,
        "no_work_snap_oof_mae_mean": no_work_mean,
        "work_correction_oof_gain": no_work_mean - with_work_mean,
        "repeated_results": repeated,
        "test_learned_linked_rows": int(learned_mask.sum()),
        "test_deterministic_linked_rows": int(deterministic_mask.sum()),
        "test_union_linked_rows": int(union_mask.sum()),
        "test_unlinked_rows_changed_from_current_snap": int(
            np.sum(
                np.abs(
                    diagnostic_prediction[~union_mask]
                    - current_snap_prediction[~union_mask]
                )
                > 1e-12
            )
        ),
        "removed_work_correction_min": float(correction[~union_mask].min()),
        "removed_work_correction_max": float(correction[~union_mask].max()),
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== mean_working 보정 제거 진단 =====")
    print(f"보정 포함+Snap OOF : {with_work_mean:.12f}")
    print(f"보정 제외+Snap OOF : {no_work_mean:.12f}")
    print(f"보정의 OOF 개선량  : {no_work_mean-with_work_mean:.12f}")
    print(f"Test union 연결 행  : {union_mask.sum():,}/{len(test):,}")
    print(
        "현재 Snap과 다른 행 : "
        f"{metrics['test_unlinked_rows_changed_from_current_snap']:,}"
    )
    print(f"진단 제출 파일      : {submission_path}")
    print(f"결과 요약           : {metrics_path}")
    print("주의: 이 파일은 최고점 후보가 아니라 gap 분리용입니다.")


if __name__ == "__main__":
    main()
