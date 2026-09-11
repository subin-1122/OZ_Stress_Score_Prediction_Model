"""
mean_working 보정 강도 alpha=0.90 실전 확인용 제출
=================================================

목적
----
현재 최고 제출은 mean_working별 OOF 잔차 중앙값을 alpha=0.75만큼 반영한다.
보정 제거 진단에서 이 신호가 Public에서 OOF 예상보다 크게 작동했으므로,
보정 강도를 0.90으로 높인 버전을 한 번 실전 확인한다.

주의
----
- 최고 OOF 후보는 여전히 alpha=0.75다.
- alpha=0.90은 snapping OOF가 평균 0.000028 나빴다.
- 다만 사전에 정한 허용 범위 0.0001 안이어서 Public 확인 후보로만 만든다.
- alpha=1.00은 OOF가 0.000192 나빠 허용 범위를 벗어나므로 사용하지 않는다.

누수 방지
---------
- 보정값은 train OOF 잔차로만 만든다.
- test 통계량이나 test 정답을 사용하지 않는다.
- alpha 후보 비교도 train OOF에서만 수행한다.
- test에는 OOF에서 고정한 alpha=0.90을 그대로 적용한다.

실행
----
프로젝트 루트에서 다음 명령을 실행한다.

    .venv/bin/python experiment/experiment_v9_alpha090_public_check.py
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
CURRENT_ALPHA = 0.75
SELECTED_ALPHA = 0.90
GRID_STEP = 0.01
ALPHAS_CHECKED = (0.75, 0.85, 0.90, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="mean_working alpha=0.90 실전 확인용 파일을 만듭니다."
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
    return np.clip(np.rint(prediction / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def evaluate_alphas(
    v4,
    train: pd.DataFrame,
    oof: pd.DataFrame,
) -> list[dict[str, float | int]]:
    """동일한 100-fold 교차 보정값에 alpha만 바꿔 OOF를 비교한다."""
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

        for alpha in ALPHAS_CHECKED:
            raw_prediction = np.clip(
                base_oof + alpha * correction, 0.0, 1.0
            )
            raw_prediction[union_mask] = union_label[union_mask]
            snapped_prediction = snap(raw_prediction)
            results.append(
                {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "raw_oof_mae": float(
                        mean_absolute_error(y, raw_prediction)
                    ),
                    "snapped_oof_mae": float(
                        mean_absolute_error(y, snapped_prediction)
                    ),
                }
            )
    return results


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
        raise ValueError("제출 예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("제출 예측값이 0~1 범위를 벗어났습니다.")
    if not np.allclose(
        prediction / GRID_STEP,
        np.rint(prediction / GRID_STEP),
        atol=1e-9,
    ):
        raise ValueError("제출 예측값이 0.01 격자에 맞지 않습니다.")


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
    current_snap = pd.read_csv(
        output_dir / "best_union_grid_snap_submission.csv"
    )
    y = train[TARGET].to_numpy(float)

    repeated = evaluate_alphas(v4, train, oof)
    result_frame = pd.DataFrame(repeated)
    summary = (
        result_frame.groupby("alpha")
        .agg(
            raw_oof_mae_mean=("raw_oof_mae", "mean"),
            snapped_oof_mae_mean=("snapped_oof_mae", "mean"),
        )
        .reset_index()
    )
    current_oof = float(
        summary.loc[
            summary["alpha"] == CURRENT_ALPHA,
            "snapped_oof_mae_mean",
        ].iloc[0]
    )
    selected_oof = float(
        summary.loc[
            summary["alpha"] == SELECTED_ALPHA,
            "snapped_oof_mae_mean",
        ].iloc[0]
    )

    learned_mask, deterministic_mask, union_mask = v8.make_test_masks(
        v3, v4, train, test, y
    )
    recovered_base, correction = v8.recover_base_test_prediction(
        v3,
        train,
        test,
        oof,
        current_v4,
        union_mask,
    )
    selected_prediction = recovered_base.copy()
    selected_prediction[~union_mask] = np.clip(
        recovered_base[~union_mask]
        + SELECTED_ALPHA * correction[~union_mask],
        0.0,
        1.0,
    )
    # 연결된 행은 train에서 전달받은 점수를 그대로 유지한다.
    selected_prediction[union_mask] = current_v4.loc[
        union_mask, TARGET
    ].to_numpy(float)
    selected_prediction = snap(selected_prediction)

    submission = current_v4.copy()
    submission[TARGET] = selected_prediction
    validate_submission(submission, sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "experimental_alpha090_link_snap_submission.csv"
    metrics_path = output_dir / "experimental_alpha090_link_snap_metrics.json"
    submission.to_csv(submission_path, index=False, float_format="%.2f")

    current_snap_prediction = current_snap[TARGET].to_numpy(float)
    pivot = result_frame.pivot(
        index="seed", columns="alpha", values="snapped_oof_mae"
    )
    selected_split_gain = pivot[CURRENT_ALPHA] - pivot[SELECTED_ALPHA]
    metrics = {
        "purpose": "Public calibration experiment; alpha=0.75 remains the best OOF setting",
        "current_alpha": CURRENT_ALPHA,
        "selected_alpha": SELECTED_ALPHA,
        "grid_step": GRID_STEP,
        "current_alpha_oof_mae": current_oof,
        "selected_alpha_oof_mae": selected_oof,
        "selected_alpha_oof_change": selected_oof - current_oof,
        "selected_split_wins": int((selected_split_gain > 0).sum()),
        "selected_split_ties": int((selected_split_gain == 0).sum()),
        "selected_split_losses": int((selected_split_gain < 0).sum()),
        "alpha_summary": summary.to_dict(orient="records"),
        "repeated_results": repeated,
        "test_union_linked_rows": int(union_mask.sum()),
        "test_changed_from_alpha075_snap": int(
            np.sum(
                np.abs(selected_prediction - current_snap_prediction) > 1e-12
            )
        ),
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== alpha=0.90 실전 확인 후보 =====")
    print(f"alpha=0.75 Snap OOF : {current_oof:.12f}")
    print(f"alpha=0.90 Snap OOF : {selected_oof:.12f}")
    print(f"OOF 변화            : {selected_oof-current_oof:+.12f}")
    print(
        "Split 승/무/패       : "
        f"{metrics['selected_split_wins']}/"
        f"{metrics['selected_split_ties']}/"
        f"{metrics['selected_split_losses']}"
    )
    print(
        "alpha=0.75와 다른 행: "
        f"{metrics['test_changed_from_alpha075_snap']:,}/{len(test):,}"
    )
    print(f"실험 제출 파일       : {submission_path}")
    print(f"결과 요약            : {metrics_path}")


if __name__ == "__main__":
    main()
