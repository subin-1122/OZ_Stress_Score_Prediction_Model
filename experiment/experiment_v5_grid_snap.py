"""
stress_score의 0.01 간격을 이용한 최종 예측값 반올림
====================================================

이 실험은 모델을 다시 학습하지 않는다. 현재 가장 좋은 제출 파일인
``outputs/best_union_submission.csv``의 예측값을 train에서 확인한 유효 간격
0.01에 맞춰 가장 가까운 값으로 반올림한다.

왜 가능한가?
------------
train의 stress_score는 0.00부터 1.00까지 0.01 간격으로만 존재한다. 모델은
0.43627처럼 격자 사이의 값을 예측할 수 있지만 실제 정답은 0.43 또는 0.44와
같은 값이다. 따라서 가장 가까운 0.01 단위로 맞추는 실험을 할 수 있다.

누수 방지
---------
- target 간격은 train의 stress_score로만 확인한다.
- test의 분포, 평균, 중앙값, 정답은 사용하지 않는다.
- OOF 비교에서는 각 학습 fold만으로도 0.01 간격이 확인되는지 검사한다.
- test에는 train에서 확정한 동일한 0.01 규칙만 적용한다.

실행 방법
---------
프로젝트 루트에서 다음 명령을 실행한다.

    .venv/bin/python experiment/experiment_v5_grid_snap.py

VS Code에서는 이 파일을 열고 오른쪽 위의 "Python 파일 실행" 버튼을 눌러도
된다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


TARGET = "stress_score"
ID_COLUMN = "ID"
GRID_DECIMALS = 2
GRID_SCALE = 10**GRID_DECIMALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="현재 최고 제출값을 train target의 0.01 간격으로 반올림합니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="train.csv와 sample_submission.csv가 있는 폴더",
    )
    parser.add_argument(
        "--input-submission",
        type=Path,
        default=None,
        help="반올림할 제출 파일. 기본값은 outputs/best_union_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="결과 폴더. 기본값은 프로젝트의 outputs 폴더",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """실행 위치와 관계없이 프로젝트 루트를 찾는다."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "open (3)" / "train.csv").exists():
                return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


def infer_grid_step(target: np.ndarray) -> float:
    """
    train target을 정수 단위로 바꾼 뒤 값 사이 간격의 최대공약수를 구한다.

    예를 들어 0.00, 0.01, 0.02는 0, 1, 2로 바뀌며 최대공약수는 1이다.
    이를 다시 100으로 나누면 target 간격 0.01을 얻는다.
    """
    scaled = target * GRID_SCALE
    rounded = np.rint(scaled).astype(int)
    if not np.allclose(scaled, rounded, atol=1e-9):
        raise ValueError(
            f"stress_score가 소수점 {GRID_DECIMALS}자리 격자에 맞지 않습니다."
        )

    unique = np.unique(rounded)
    if len(unique) < 2:
        raise ValueError("target 고유값이 하나뿐이라 간격을 계산할 수 없습니다.")

    grid_gcd = 0
    for difference in np.diff(unique):
        grid_gcd = gcd(grid_gcd, int(difference))
    if grid_gcd <= 0:
        raise ValueError("올바른 target 간격을 계산하지 못했습니다.")
    return grid_gcd / GRID_SCALE


def snap_to_grid(prediction: np.ndarray, step: float) -> np.ndarray:
    """예측값을 가장 가까운 target 격자로 이동하고 0~1 범위로 제한한다."""
    return np.clip(np.rint(prediction / step) * step, 0.0, 1.0)


def load_v4_module(project_root: Path):
    """v4의 결정 규칙 OOF 함수를 재사용하기 위해 모듈을 불러온다."""
    path = project_root / "experiment/experiment_v4_deterministic_union.py"
    spec = importlib.util.spec_from_file_location("experiment_v4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_grid_in_each_fold(y: np.ndarray, expected_step: float) -> None:
    """검증 행을 제외한 각 fold의 target만으로도 같은 간격이 나오는지 확인한다."""
    splitter = KFold(n_splits=100, shuffle=True, random_state=11)
    learned_steps = [infer_grid_step(y[train_idx]) for train_idx, _ in splitter.split(y)]
    if not np.allclose(learned_steps, expected_step, atol=1e-12):
        raise ValueError("일부 OOF 학습 fold에서 target 간격이 달라졌습니다.")


def evaluate_repeated_v4_oof(
    project_root: Path,
    train: pd.DataFrame,
    step: float,
) -> list[dict[str, float | int]]:
    """v4의 다섯 연결 split에서 반올림 전후 OOF MAE를 비교한다."""
    v4 = load_v4_module(project_root)
    oof = pd.read_csv(project_root / "outputs/best_record_linkage_oof.csv")
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

        correction = v4.crossfit_work_correction(
            train, y, base_oof, union_mask
        )
        before_prediction = np.clip(
            base_oof + v4.WORK_CORRECTION_ALPHA * correction, 0.0, 1.0
        )
        before_prediction[union_mask] = union_label[union_mask]
        after_prediction = snap_to_grid(before_prediction, step)

        before_mae = mean_absolute_error(y, before_prediction)
        after_mae = mean_absolute_error(y, after_prediction)
        results.append(
            {
                "seed": int(seed),
                "before_mae": float(before_mae),
                "after_mae": float(after_mae),
                "improvement": float(before_mae - after_mae),
                "changed_rows": int(
                    np.sum(np.abs(before_prediction - after_prediction) > 1e-12)
                ),
            }
        )
    return results


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    step: float,
) -> None:
    """DACON 제출 형식, 값 범위, target 격자를 저장 전에 검사한다."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError("제출 파일 열이 sample_submission과 다릅니다.")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 파일의 ID 또는 순서가 sample_submission과 다릅니다.")

    prediction = submission[TARGET].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("예측값이 0~1 범위를 벗어났습니다.")
    if not np.allclose(prediction / step, np.rint(prediction / step), atol=1e-9):
        raise ValueError("일부 예측값이 train target 격자에 맞지 않습니다.")


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    data_dir = args.data_dir.resolve() if args.data_dir else project_root / "open (3)"
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "outputs"
    input_path = (
        args.input_submission.resolve()
        if args.input_submission
        else output_dir / "best_union_submission.csv"
    )

    train = pd.read_csv(data_dir / "train.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    previous = pd.read_csv(input_path)
    y = train[TARGET].to_numpy(float)

    step = infer_grid_step(y)
    validate_grid_in_each_fold(y, step)
    if not np.isclose(step, 0.01):
        raise ValueError(f"예상한 0.01과 다른 target 간격이 나왔습니다: {step}")

    repeated_oof = evaluate_repeated_v4_oof(project_root, train, step)
    before_mean = float(np.mean([row["before_mae"] for row in repeated_oof]))
    after_mean = float(np.mean([row["after_mae"] for row in repeated_oof]))

    final_submission = previous.copy()
    before_test = final_submission[TARGET].to_numpy(float)
    after_test = snap_to_grid(before_test, step)
    final_submission[TARGET] = after_test
    validate_submission(final_submission, sample, step)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "best_union_grid_snap_submission.csv"
    metrics_path = output_dir / "best_union_grid_snap_metrics.json"
    # float_format을 고정하여 제출 파일에서도 0.01 단위가 명확하게 보이도록 한다.
    final_submission.to_csv(submission_path, index=False, float_format="%.2f")

    metrics = {
        "method": "best_union prediction snapped to train target grid",
        "target_grid_step": step,
        "target_unique_values": int(train[TARGET].nunique()),
        "oof_before_mae_mean": before_mean,
        "oof_after_mae_mean": after_mean,
        "oof_improvement_mean": before_mean - after_mean,
        "oof_all_repeats_improved": bool(
            all(row["improvement"] > 0 for row in repeated_oof)
        ),
        "oof_repeated_results": repeated_oof,
        "test_changed_rows": int(np.sum(np.abs(after_test - before_test) > 1e-12)),
        "input_submission": str(input_path),
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 0.01 Grid Snap 결과 =====")
    print(f"Target 간격          : {step:.2f}")
    print(f"반복 OOF 반올림 전   : {before_mean:.12f}")
    print(f"반복 OOF 반올림 후   : {after_mean:.12f}")
    print(f"반복 OOF 평균 개선   : {before_mean-after_mean:.12f}")
    print(f"Test 변경 행         : {metrics['test_changed_rows']:,}/{len(previous):,}")
    print(f"제출 파일            : {submission_path}")
    print(f"결과 요약            : {metrics_path}")


if __name__ == "__main__":
    main()
