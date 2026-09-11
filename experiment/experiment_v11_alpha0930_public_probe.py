"""
mean_working 보정 강도 alpha=0.930 탐색 제출
================================================

이 파일이 하는 일
------------------
1. 기존 레코드 연결 + ExtraTrees OOF 결과를 그대로 불러온다.
2. train 데이터만 이용해 mean_working 구간별 잔차 보정값을 계산한다.
3. 미연결 행에 alpha=0.930만큼 보정하고, 연결 행은 연결값을 유지한다.
4. stress_score의 0.01 간격에 맞춰 예측값을 반올림한다.
5. DACON 제출 형식과 ID 순서를 검사한 뒤 CSV를 저장한다.

선택 이유
---------
- alpha=0.900의 Public MAE 0.12558이 현재 최고 기록이다.
- alpha=0.930은 0.900보다 OOF MAE가 0.000046 나쁘므로 최종 확정값이 아니다.
- 다만 test 예측 127행이 달라져, 강한 보정이 Public에서도 계속 유효한지
  확인하기 위한 탐색용 제출로는 alpha=0.910보다 정보량이 많다.

누수 방지
---------
- 보정값과 레코드 연결 규칙은 train 데이터에서만 학습한다.
- test 데이터의 정답, 평균, 중앙값 또는 결측치 통계는 사용하지 않는다.
- alpha=0.930은 실행 전에 고정되어 있으며 test 결과로 선택하지 않는다.

실행 방법
---------
프로젝트 루트에서 아래 명령을 실행한다.

    .venv/bin/python experiment/experiment_v11_alpha0930_public_probe.py
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
CURRENT_ALPHA = 0.90
SELECTED_ALPHA = 0.93
GRID_STEP = 0.01


def parse_args() -> argparse.Namespace:
    """VS Code와 터미널에서 데이터/출력 경로를 선택할 수 있게 한다."""
    parser = argparse.ArgumentParser(
        description="alpha=0.930 탐색 제출 파일을 생성합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def find_project_root() -> Path:
    """실행 위치와 관계없이 open (3)/train.csv가 있는 프로젝트를 찾는다."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "open (3)" / "train.csv").exists():
                return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


def load_module(name: str, path: Path):
    """앞선 실험에서 검증한 함수를 중복 작성하지 않고 불러온다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(prediction: np.ndarray) -> np.ndarray:
    """train target에서 확인한 0.01 간격으로 예측값을 맞춘다."""
    return np.clip(np.rint(prediction / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def evaluate_oof(v4, train: pd.DataFrame, oof: pd.DataFrame) -> list[dict]:
    """동일한 OOF 조건에서 alpha=0.90과 0.93만 직접 비교한다."""
    y = train[TARGET].to_numpy(float)
    base_oof = oof["base_oof"].to_numpy(float)
    learned_label = oof["link_label"].to_numpy(float)
    learned_mask = oof["linked"].astype(bool).to_numpy()

    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(features, numeric, categorical)

    results: list[dict] = []
    for seed in v4.LINK_SPLIT_SEEDS:
        # deterministic 연결도 해당 반복에서 허용된 train 부분만 이용한다.
        deterministic_label = v4.aggregate_oof_labels(
            rule_pairs, y, len(train), seed
        )
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]

        # 각 행의 보정값은 그 행의 정답을 제외하는 cross-fitting으로 만든다.
        correction = v4.crossfit_work_correction(
            train, y, base_oof, union_mask
        )

        predictions: dict[float, np.ndarray] = {}
        for alpha in (CURRENT_ALPHA, SELECTED_ALPHA):
            prediction = np.clip(base_oof + alpha * correction, 0.0, 1.0)
            prediction[union_mask] = union_label[union_mask]
            predictions[alpha] = snap(prediction)

        results.append(
            {
                "seed": int(seed),
                "alpha090_oof_mae": float(
                    mean_absolute_error(y, predictions[CURRENT_ALPHA])
                ),
                "alpha093_oof_mae": float(
                    mean_absolute_error(y, predictions[SELECTED_ALPHA])
                ),
            }
        )
    return results


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """업로드 전에 자주 발생하는 제출 파일 오류를 모두 차단한다."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError("제출 파일 열이 sample_submission과 다릅니다.")
    if len(submission) != len(sample):
        raise ValueError("제출 파일 행 수가 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 ID 또는 순서가 sample_submission과 다릅니다.")

    prediction = submission[TARGET].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("제출 예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("제출 예측값이 0~1 범위를 벗어났습니다.")
    if not np.allclose(
        prediction / GRID_STEP,
        np.rint(prediction / GRID_STEP),
        atol=1e-9,
    ):
        raise ValueError("제출 예측값이 0.01 간격에 맞지 않습니다.")


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
    v8 = load_module(
        "experiment_v8",
        root / "experiment/experiment_v8_no_work_correction_diagnostic.py",
    )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    oof = pd.read_csv(output_dir / "best_record_linkage_oof.csv")
    current_v4 = pd.read_csv(output_dir / "best_union_submission.csv")
    alpha090_submission = pd.read_csv(
        output_dir / "experimental_alpha090_link_snap_submission.csv"
    )
    y = train[TARGET].to_numpy(float)

    repeated = evaluate_oof(v4, train, oof)
    result_frame = pd.DataFrame(repeated)
    alpha090_oof = float(result_frame["alpha090_oof_mae"].mean())
    alpha093_oof = float(result_frame["alpha093_oof_mae"].mean())

    # 실제 test에서는 전체 train 3,000행을 연결 후보로 사용한다.
    # test는 변환과 예측에만 사용되며 학습 통계에는 포함되지 않는다.
    _, _, union_mask = v8.make_test_masks(v3, v4, train, test, y)
    recovered_base, correction = v8.recover_base_test_prediction(
        v3, train, test, oof, current_v4, union_mask
    )

    prediction = recovered_base.copy()
    prediction[~union_mask] = np.clip(
        recovered_base[~union_mask]
        + SELECTED_ALPHA * correction[~union_mask],
        0.0,
        1.0,
    )
    # 연결된 행의 값은 mean_working 보정을 적용하지 않고 연결값을 유지한다.
    prediction[union_mask] = current_v4.loc[
        union_mask, TARGET
    ].to_numpy(float)
    prediction = snap(prediction)

    submission = sample.copy()
    submission[TARGET] = prediction
    validate_submission(submission, sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "experimental_alpha093_link_snap_submission.csv"
    metrics_path = output_dir / "experimental_alpha093_link_snap_metrics.json"
    submission.to_csv(submission_path, index=False, float_format="%.2f")

    alpha090_prediction = alpha090_submission[TARGET].to_numpy(float)
    changed_mask = np.abs(prediction - alpha090_prediction) > 1e-12
    oof_delta = result_frame["alpha093_oof_mae"] - result_frame["alpha090_oof_mae"]
    metrics = {
        "purpose": "Public probe for stronger mean_working correction",
        "current_alpha": CURRENT_ALPHA,
        "selected_alpha": SELECTED_ALPHA,
        "grid_step": GRID_STEP,
        "alpha090_public_mae_reported": 0.12558,
        "alpha090_oof_mae": alpha090_oof,
        "alpha093_oof_mae": alpha093_oof,
        "alpha093_oof_change_vs_090": alpha093_oof - alpha090_oof,
        "alpha093_seed_wins_vs_090": int((oof_delta < 0).sum()),
        "alpha093_seed_ties_vs_090": int((oof_delta == 0).sum()),
        "alpha093_seed_losses_vs_090": int((oof_delta > 0).sum()),
        "test_union_linked_rows": int(union_mask.sum()),
        "test_changed_from_alpha090": int(changed_mask.sum()),
        # 부동소수점의 아주 작은 차이를 제외하고, 실제 0.01 값이 바뀐 행만 센다.
        "test_changed_up": int(((prediction > alpha090_prediction) & changed_mask).sum()),
        "test_changed_down": int(((prediction < alpha090_prediction) & changed_mask).sum()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "repeated_results": repeated,
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== alpha=0.930 탐색 제출 =====")
    print(f"alpha=0.900 Snap OOF : {alpha090_oof:.12f}")
    print(f"alpha=0.930 Snap OOF : {alpha093_oof:.12f}")
    print(f"OOF 변화             : {alpha093_oof-alpha090_oof:+.12f}")
    print(
        "seed 승/무/패        : "
        f"{metrics['alpha093_seed_wins_vs_090']}/"
        f"{metrics['alpha093_seed_ties_vs_090']}/"
        f"{metrics['alpha093_seed_losses_vs_090']}"
    )
    print(
        "alpha=0.900과 다른 행: "
        f"{metrics['test_changed_from_alpha090']:,}/{len(test):,}"
    )
    print(f"제출 파일            : {submission_path}")
    print(f"결과 요약            : {metrics_path}")


if __name__ == "__main__":
    main()
