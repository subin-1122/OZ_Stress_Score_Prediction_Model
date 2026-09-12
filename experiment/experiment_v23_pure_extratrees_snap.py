"""
순수 3-seed ExtraTrees + 0.01 snapping 진단 제출
==================================================

왜 이 파일을 만드는가
---------------------
현재 최고 Public 파이프라인에는 레코드 연결과 mean_working 잔차 보정이 들어간다.
이 진단 제출은 두 레이어를 모두 제거해서 순수 ExtraTrees의 실제 Public 점수를
측정한다. 점수를 보고 hyperparameter를 고르는 목적이 아니라, train OOF에서
큰 이득을 보인 레코드 연결이 실제 test에서도 같은 정도로 전이되는지 분리해서
확인하기 위한 한 번의 대조 실험이다.

구성
----
1. 기존 최고 base와 똑같이 seed 11, 101, 1001을 사용한다.
2. 각 seed에서 100-fold 모델의 test 예측을 평균한다.
3. 각 트리 예측의 위·아래 10%를 제외한 절사평균을 사용한다.
4. 레코드 연결과 mean_working 보정은 적용하지 않는다.
5. 마지막에 train target에서 확인한 0.01 격자로만 반올림한다.

누수 방지
---------
- 결측치 처리와 원-핫 인코딩은 각 fold의 train 부분으로만 fit한다.
- test는 학습이 끝난 변환과 predict에만 사용한다.
- test의 평균, 중앙값, 범주 목록, 결측치 통계를 학습하지 않는다.
- test target이나 외부 데이터는 사용하지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v23_pure_extratrees_snap.py

전체 재현은 300개 모델과 총 110,000개 트리를 학습하므로 시간이 걸린다.
--quick은 코드 점검용이며 그 결과를 제출하면 안 된다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
GRID_STEP = 0.01
OLD_SIMPLE_PUBLIC_GAP = 0.002422


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="순수 3-seed ExtraTrees + 0.01 snap 제출을 만듭니다."
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
        help="결과 폴더. 기본값은 프로젝트의 outputs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="5-fold, 50-tree 코드 점검용. 제출 금지",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    """기존 v3와 v4에서 검증된 전처리·연결 함수를 재사용한다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    """예측을 0.01 격자로 반올림하고 target 범위인 0~1로 제한한다."""
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def make_test_prediction(v3, train, test, y, quick: bool) -> np.ndarray:
    """
    v3와 완전히 같은 seed, fold, 트리 설정으로 순수 test 예측만 재현한다.

    저장된 base OOF는 이미 있으므로 OOF를 다시 계산하지 않는다. 하지만 v3는
    순수 base test 예측을 별도 파일로 저장하지 않았기 때문에 test 예측 모델은
    다시 학습해야 한다.
    """
    train_features = v3.make_model_features(train.drop(columns=[TARGET]))
    test_features = v3.make_model_features(test)
    seeds = v3.BASE_SEEDS
    n_splits = 5 if quick else v3.BASE_N_SPLITS
    seed_predictions = []

    for seed in seeds:
        n_estimators = 50 if quick else v3.N_ESTIMATORS_BY_SEED[seed]
        test_sum = np.zeros(len(test), dtype=float)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        started = time.perf_counter()

        for fold, (train_idx, _) in enumerate(splitter.split(train_features), start=1):
            preprocessor = v3.build_preprocessor(train_features.iloc[train_idx])
            x_train = preprocessor.fit_transform(train_features.iloc[train_idx])
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
            test_sum += v3.trimmed_tree_prediction(
                model,
                x_test,
                v3.TRIM_RATIO,
            )

            report_every = 1 if quick else 10
            if fold % report_every == 0 or fold == n_splits:
                print(
                    f"[pure base] seed={seed} fold={fold:03d}/{n_splits} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        seed_predictions.append(test_sum / n_splits)

    return np.mean(np.column_stack(seed_predictions), axis=1)


def compare_link_thresholds(v4, train, y, stored) -> dict:
    """
    저장된 0.97 OOF 연결 중 probability 0.98 이상만 남겼을 때를 비교한다.

    0.98 전용 분류기를 다시 학습할 필요는 없다. threshold만 높이면 기존 0.97
    연결의 부분집합이 되므로, 저장된 probability와 나머지 안전 조건을 통과한
    linked 마스크를 그대로 이용할 수 있다.
    """
    base_oof = stored["base_oof"].to_numpy(float)
    learned_label = stored["link_label"].to_numpy(float)
    probability = stored["link_probability"].to_numpy(float)
    mask_097 = stored["linked"].astype(bool).to_numpy()
    mask_098 = mask_097 & (probability >= 0.98)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(features, numeric, categorical)
    repeated = []

    for seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            pairs,
            y,
            len(train),
            int(seed),
        )
        deterministic_mask = np.isfinite(deterministic_label)
        scores = {}

        for name, learned_mask in (("threshold_097", mask_097), ("threshold_098", mask_098)):
            union_mask = learned_mask | deterministic_mask
            union_label = learned_label.copy()
            union_label[deterministic_mask] = deterministic_label[deterministic_mask]
            correction = v4.crossfit_work_correction(
                train,
                y,
                base_oof,
                union_mask,
            )
            prediction = snap(base_oof + v4.WORK_CORRECTION_ALPHA * correction)
            prediction[union_mask] = union_label[union_mask]
            scores[name] = float(mean_absolute_error(y, prediction))

        repeated.append(
            {
                "seed": int(seed),
                **scores,
                "gain_098_vs_097": scores["threshold_097"] - scores["threshold_098"],
            }
        )

    lost = mask_097 & ~mask_098
    gains = np.asarray([row["gain_098_vs_097"] for row in repeated])
    return {
        "linked_rows_097": int(mask_097.sum()),
        "linked_rows_098": int(mask_098.sum()),
        "rows_removed_by_098": int(lost.sum()),
        "removed_rows_exact_accuracy": float(
            np.mean(learned_label[lost] == y[lost])
        ),
        "gain_098_vs_097_mean": float(gains.mean()),
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "repeated": repeated,
        "decision": "reject_threshold_098",
    }


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """DACON 제출 열, ID, 행 수, 결측치, 범위와 0.01 격자를 검사한다."""
    if list(submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("제출 열은 ID, stress_score 순서여야 합니다.")
    if submission.shape != sample.shape:
        raise ValueError(
            f"제출 크기가 sample_submission과 다릅니다: {submission.shape}"
        )
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 ID 또는 순서가 sample_submission과 다릅니다.")
    prediction = submission[TARGET].to_numpy(float)
    if not np.isfinite(prediction).all():
        raise ValueError("제출 예측값에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0.0) | (prediction > 1.0)).any():
        raise ValueError("제출 예측값이 0~1 범위를 벗어났습니다.")
    if not np.allclose(prediction / GRID_STEP, np.rint(prediction / GRID_STEP)):
        raise ValueError("제출 예측값 중 0.01 격자에 맞지 않는 값이 있습니다.")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve() if args.data_dir else ROOT / "open (3)"
    output_dir = args.output_dir.resolve() if args.output_dir else ROOT / "outputs"
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train, test, sample = v3.load_data(data_dir)
    stored = pd.read_csv(output_dir / "best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    if not stored[ID_COLUMN].equals(train[ID_COLUMN]):
        raise ValueError("저장된 base OOF의 ID 순서가 train과 다릅니다.")

    raw_test = make_test_prediction(v3, train, test, y, args.quick)
    snapped_test = snap(raw_test)
    submission = sample.copy()
    submission[TARGET] = snapped_test
    validate_submission(submission, sample)

    base_oof = stored["base_oof"].to_numpy(float)
    raw_oof_mae = mean_absolute_error(y, base_oof)
    snapped_oof_mae = mean_absolute_error(y, snap(base_oof))
    threshold_audit = compare_link_thresholds(v4, train, y, stored)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    submission_path = (
        output_dir / f"pure_extratrees_3seed_snap_submission{suffix}.csv"
    )
    metrics_path = output_dir / f"pure_extratrees_3seed_snap_metrics{suffix}.json"
    submission.to_csv(submission_path, index=False, float_format="%.2f")
    metrics = {
        "method": "pure 3-seed ExtraTrees trimmed mean plus 0.01 snapping",
        "test_rows": int(len(test)),
        "base_seeds": list(v3.BASE_SEEDS),
        "base_n_splits": 5 if args.quick else v3.BASE_N_SPLITS,
        "quick_mode": bool(args.quick),
        "record_linkage_used": False,
        "mean_working_correction_used": False,
        "test_statistics_used": False,
        "raw_base_oof_mae": float(raw_oof_mae),
        "snapped_base_oof_mae": float(snapped_oof_mae),
        "snap_oof_gain": float(raw_oof_mae - snapped_oof_mae),
        "illustrative_old_gap": OLD_SIMPLE_PUBLIC_GAP,
        "illustrative_public_estimate": float(
            snapped_oof_mae + OLD_SIMPLE_PUBLIC_GAP
        ),
        "threshold_audit": threshold_audit,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 순수 ExtraTrees + snap 진단 제출 =====")
    print(f"Raw base OOF MAE       : {raw_oof_mae:.9f}")
    print(f"Snapped base OOF MAE   : {snapped_oof_mae:.9f}")
    print(
        f"과거 gap 단순 적용값  : "
        f"{snapped_oof_mae + OLD_SIMPLE_PUBLIC_GAP:.9f} "
        "(참고용, 보장값 아님)"
    )
    print(
        "threshold 0.98 효과  : "
        f"{threshold_audit['gain_098_vs_097_mean']:+.9f} "
        f"승/무/패={threshold_audit['seed_wins']}/"
        f"{threshold_audit['seed_ties']}/"
        f"{threshold_audit['seed_losses']}"
    )
    print(f"제출 파일              : {submission_path}")
    print(f"결과 요약              : {metrics_path}")


if __name__ == "__main__":
    main()
