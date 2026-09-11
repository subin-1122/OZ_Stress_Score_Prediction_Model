"""
unlinked 전용 ExtraTrees 20% 혼합과 0.01 snapping 결합 실험
=============================================================

목적
----
현재 최고 모델(v4)의 예측에 다음 두 단계를 순서대로 적용한다.

1. 레코드 연결로 해결되지 않는 행에만 unlinked 전용 모델의 차이를 20% 반영한다.
2. 최종 예측을 train target의 유효 간격인 0.01 단위로 반올림한다.

검증 원칙
---------
- unlinked 전용 학습 행은 각 OOF fold의 학습 부분 안에서만 판정한다.
- 결측치 중앙값과 원-핫 인코딩도 각 모델의 학습 행으로만 fit한다.
- test는 각 fold에서 transform/predict만 하며 학습 통계에 사용하지 않는다.
- 앞선 선택에 쓰지 않은 세 seed(7, 77, 777)의 20-fold 평균을 사용한다.
- 현재 snapping 단독 OOF보다 결합 OOF가 낮을 때만 제출 CSV를 생성한다.

실행
----
프로젝트 루트에서 다음 명령을 실행한다.

    .venv/bin/python experiment/experiment_v6_unlinked_blend_grid_snap.py

VS Code에서는 파일을 열고 오른쪽 위의 "Python 파일 실행"을 눌러도 된다.
전체 검증과 test 예측을 함께 만들기 때문에 몇 분 정도 걸릴 수 있다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


TARGET = "stress_score"
ID_COLUMN = "ID"
DELTA_SEEDS = (7, 77, 777)
DELTA_N_SPLITS = 20
N_ESTIMATORS = 300
TRIM_RATIO = 0.05
UNLINKED_BLEND_WEIGHT = 0.20
GRID_STEP = 0.01
IMPROVEMENT_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="unlinked 20% 혼합과 0.01 snapping을 결합해 비교합니다."
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
    """같은 프로젝트의 v3/v4 함수를 복사하지 않고 안전하게 재사용한다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_added_features(v3, frame: pd.DataFrame) -> pd.DataFrame:
    """
    기존 v3 피처에 unlinked 그룹에서 검증할 raw/교차 피처를 추가한다.

    mean_working은 숫자형 원본으로 넣는다. 근로시간과 activity/gender의 조합은
    문자열 범주로 만들어 트리가 특정 조합을 한 번의 분할 후보로 볼 수 있게 한다.
    이 값들은 모두 같은 행 안의 정보만 사용하므로 행 사이 통계가 섞이지 않는다.
    """
    result = v3.make_model_features(frame)
    work_key = frame["mean_working"].astype("string").fillna("MISSING")
    result["mean_working"] = frame["mean_working"].to_numpy()
    result["age"] = frame["age"].to_numpy()
    result["sleep_pattern"] = frame["sleep_pattern"].to_numpy()
    result["work_x_activity"] = (
        work_key + "__" + frame["activity"].astype("string").fillna("MISSING")
    )
    result["work_x_gender"] = (
        work_key + "__" + frame["gender"].astype("string").fillna("MISSING")
    )
    return result


def trimmed_prediction(model: ExtraTreesRegressor, transformed: np.ndarray) -> np.ndarray:
    """트리 예측의 위아래 5%를 제외하고 평균한다."""
    values = np.column_stack(
        [tree.predict(transformed) for tree in model.estimators_]
    )
    values.sort(axis=1)
    cut = int(len(model.estimators_) * TRIM_RATIO)
    return values[:, cut:-cut].mean(axis=1)


def fit_extra_trees(
    v3,
    features: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_features: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """지정된 학습 행만으로 전처리와 ExtraTrees를 fit한다."""
    preprocessor = v3.build_preprocessor(features.iloc[train_idx])
    x_train = preprocessor.fit_transform(features.iloc[train_idx])
    x_valid = preprocessor.transform(features.iloc[valid_idx])
    x_test = preprocessor.transform(test_features)

    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        criterion="squared_error",
        max_features=1,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x_train, y[train_idx])
    return (
        trimmed_prediction(model, x_valid),
        trimmed_prediction(model, x_test),
    )


def make_delta_oof_and_test(
    v3,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """
    기존 모델과 unlinked 전용 모델의 예측 차이를 반복 20-fold로 만든다.

    각 fold에서는 먼저 레코드 연결 분류기를 fold-train 쌍으로만 학습한다.
    fold-train 안에서 연결 가능성이 높은 행을 제외한 뒤 추가 피처 모델을 학습한다.
    validation/test의 정보는 unlinked 학습 행을 정할 때 사용하지 않는다.
    """
    raw_train = train.drop(columns=[TARGET])
    control_train = v3.make_model_features(raw_train)
    control_test = v3.make_model_features(test)
    added_train = make_added_features(v3, raw_train)
    added_test = make_added_features(v3, test)

    link_features = train.drop(columns=[ID_COLUMN, TARGET]).reset_index(drop=True)
    numeric = link_features.select_dtypes(include=np.number).columns.tolist()
    categorical = link_features.select_dtypes(exclude=np.number).columns.tolist()
    blocks = v3.linkage_blocks(categorical)
    all_pairs = v3.train_candidate_pairs(link_features, blocks)
    n_rows = len(train)

    seed_delta_oof: list[np.ndarray] = []
    seed_delta_test: list[np.ndarray] = []
    diagnostics: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for seed in DELTA_SEEDS:
        control_oof = np.zeros(n_rows, dtype=float)
        added_oof = np.zeros(n_rows, dtype=float)
        control_test_sum = np.zeros(len(test), dtype=float)
        added_test_sum = np.zeros(len(test), dtype=float)
        train_unlinked_sizes: list[int] = []
        rng = np.random.default_rng(v3.LINKAGE_SEED + seed)
        splitter = KFold(
            n_splits=DELTA_N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(train), start=1
        ):
            in_train = np.zeros(n_rows, dtype=bool)
            in_train[train_idx] = True
            train_pairs = all_pairs[
                in_train[all_pairs[:, 0]] & in_train[all_pairs[:, 1]]
            ]

            pair_target = (
                y[train_pairs[:, 0]] == y[train_pairs[:, 1]]
            ).astype(np.int8)
            keep = v3.balanced_pair_sample(pair_target, rng)
            scales = v3.robust_scales(
                link_features, train_idx, numeric
            )
            classifier = v3.build_link_classifier(
                v3.LINKAGE_SEED + seed + fold
            )
            classifier.fit(
                v3.make_pair_features(
                    link_features,
                    train_pairs[keep],
                    numeric,
                    categorical,
                    scales,
                ),
                pair_target[keep],
            )

            # 아래 확률은 fold-train 쌍에 대해서만 계산한다. 고신뢰 쌍의 양쪽
            # 행을 제거해 unlinked 전용 학습 집합을 만든다.
            train_pair_probability = classifier.predict_proba(
                v3.make_pair_features(
                    link_features,
                    train_pairs,
                    numeric,
                    categorical,
                    scales,
                )
            )[:, 1]
            linked_edges = train_pairs[
                train_pair_probability >= v3.LINK_PROBABILITY_THRESHOLD
            ]
            train_linked = np.zeros(n_rows, dtype=bool)
            if len(linked_edges):
                train_linked[np.unique(linked_edges)] = True
            unlinked_train_idx = train_idx[~train_linked[train_idx]]
            train_unlinked_sizes.append(len(unlinked_train_idx))

            control_valid, control_test_fold = fit_extra_trees(
                v3,
                control_train,
                y,
                train_idx,
                valid_idx,
                control_test,
                random_state=seed * 1000 + fold * 10,
            )
            added_valid, added_test_fold = fit_extra_trees(
                v3,
                added_train,
                y,
                unlinked_train_idx,
                valid_idx,
                added_test,
                random_state=seed * 1000 + fold * 10 + 1,
            )
            control_oof[valid_idx] = control_valid
            added_oof[valid_idx] = added_valid
            control_test_sum += control_test_fold
            added_test_sum += added_test_fold

            print(
                f"[delta] seed={seed} fold={fold:02d}/{DELTA_N_SPLITS} "
                f"train_unlinked={len(unlinked_train_idx):,} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

        delta_oof = added_oof - control_oof
        delta_test = (
            added_test_sum - control_test_sum
        ) / DELTA_N_SPLITS
        seed_delta_oof.append(delta_oof)
        seed_delta_test.append(delta_test)
        diagnostics.append(
            {
                "seed": int(seed),
                "average_unlinked_train_rows": float(
                    np.mean(train_unlinked_sizes)
                ),
                "delta_oof_mean": float(np.mean(delta_oof)),
                "delta_oof_std": float(np.std(delta_oof)),
            }
        )

    return (
        np.mean(np.column_stack(seed_delta_oof), axis=1),
        np.mean(np.column_stack(seed_delta_test), axis=1),
        diagnostics,
    )


def snap(prediction: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(prediction / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def evaluate_against_v4(
    v4,
    train: pd.DataFrame,
    base_oof_detail: pd.DataFrame,
    delta_oof: np.ndarray,
) -> list[dict[str, float | int]]:
    """v4의 다섯 연결 split 각각에서 snapping 단독과 결합 모델을 비교한다."""
    y = train[TARGET].to_numpy(float)
    base_oof = base_oof_detail["base_oof"].to_numpy(float)
    learned_label = base_oof_detail["link_label"].to_numpy(float)
    learned_mask = base_oof_detail["linked"].astype(bool).to_numpy()
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
        current_prediction = np.clip(
            base_oof + v4.WORK_CORRECTION_ALPHA * correction,
            0.0,
            1.0,
        )
        current_prediction[union_mask] = union_label[union_mask]
        snap_only = snap(current_prediction)

        combined = np.clip(
            current_prediction + UNLINKED_BLEND_WEIGHT * delta_oof,
            0.0,
            1.0,
        )
        # 연결된 행은 train의 알려진 점수가 가장 강한 신호이므로 delta를 적용하지 않는다.
        combined[union_mask] = union_label[union_mask]
        combined_snap = snap(combined)

        snap_mae = mean_absolute_error(y, snap_only)
        combined_mae = mean_absolute_error(y, combined_snap)
        results.append(
            {
                "seed": int(seed),
                "snap_only_mae": float(snap_mae),
                "combined_mae": float(combined_mae),
                "combined_improvement": float(snap_mae - combined_mae),
                "union_coverage": float(union_mask.mean()),
            }
        )
    return results


def make_test_union_mask(
    v3,
    v4,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
) -> np.ndarray:
    """v3 학습형 연결과 v4 결정 규칙 중 하나라도 연결한 test 행을 찾는다."""
    train_link = train.drop(columns=[ID_COLUMN, TARGET])
    test_link = test.drop(columns=[ID_COLUMN])
    numeric = train_link.select_dtypes(include=np.number).columns.tolist()
    categorical = train_link.select_dtypes(exclude=np.number).columns.tolist()
    blocks = v3.linkage_blocks(categorical)
    train_pairs = v3.train_candidate_pairs(train_link, blocks)
    test_pairs = v3.train_test_candidate_pairs(train_link, test_link, blocks)
    learned_result = v3.make_test_link_prediction(
        train_link,
        test_link,
        y,
        train_pairs,
        test_pairs,
        numeric,
        categorical,
    )
    learned_mask = v3.high_confidence_mask(learned_result)
    deterministic_label = v4.make_test_rule_labels(
        train_link,
        test_link,
        y,
        numeric,
        categorical,
    )
    return learned_mask | np.isfinite(deterministic_label)


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

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    y = train[TARGET].to_numpy(float)
    v3 = load_module(
        "experiment_v3",
        root / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        root / "experiment/experiment_v4_deterministic_union.py",
    )

    delta_oof, delta_test, delta_diagnostics = make_delta_oof_and_test(
        v3, train, test, y
    )
    oof_detail = pd.read_csv(output_dir / "best_record_linkage_oof.csv")
    repeated = evaluate_against_v4(v4, train, oof_detail, delta_oof)
    snap_only_mean = float(
        np.mean([row["snap_only_mae"] for row in repeated])
    )
    combined_mean = float(
        np.mean([row["combined_mae"] for row in repeated])
    )
    improvement = snap_only_mean - combined_mean

    print("\n===== 결합 모델 OOF 비교 =====")
    for row in repeated:
        print(
            f"seed={row['seed']} snap={row['snap_only_mae']:.9f} "
            f"combined={row['combined_mae']:.9f} "
            f"improvement={row['combined_improvement']:+.9f}"
        )
    print(f"Snapping 단독 평균 : {snap_only_mean:.12f}")
    print(f"20% 혼합+Snap 평균 : {combined_mean:.12f}")
    print(f"평균 개선           : {improvement:+.12f}")

    metrics = {
        "method": "v4 + 20% unlinked-model delta + 0.01 grid snapping",
        "delta_seeds": list(DELTA_SEEDS),
        "delta_n_splits": DELTA_N_SPLITS,
        "n_estimators": N_ESTIMATORS,
        "trim_ratio": TRIM_RATIO,
        "unlinked_blend_weight": UNLINKED_BLEND_WEIGHT,
        "grid_step": GRID_STEP,
        "snap_only_oof_mae_mean": snap_only_mean,
        "combined_oof_mae_mean": combined_mean,
        "oof_improvement": improvement,
        "repeated_results": repeated,
        "delta_diagnostics": delta_diagnostics,
        "submission_created": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "unlinked_blend_grid_snap_metrics.json"
    if improvement > IMPROVEMENT_EPSILON:
        current_submission = pd.read_csv(
            output_dir / "best_union_submission.csv"
        )
        union_mask = make_test_union_mask(v3, v4, train, test, y)
        current_test = current_submission[TARGET].to_numpy(float)
        combined_test = current_test.copy()
        combined_test[~union_mask] = np.clip(
            current_test[~union_mask]
            + UNLINKED_BLEND_WEIGHT * delta_test[~union_mask],
            0.0,
            1.0,
        )
        final_test = snap(combined_test)
        submission = current_submission.copy()
        submission[TARGET] = final_test
        validate_submission(submission, sample)

        submission_path = (
            output_dir / "best_union_unlinked20_grid_snap_submission.csv"
        )
        submission.to_csv(
            submission_path,
            index=False,
            float_format="%.2f",
        )
        metrics.update(
            {
                "submission_created": True,
                "test_union_linked_rows": int(union_mask.sum()),
                "test_delta_applied_rows": int((~union_mask).sum()),
                "changed_from_snap_only_rows": int(
                    np.sum(
                        np.abs(
                            final_test
                            - snap(current_test)
                        )
                        > 1e-12
                    )
                ),
                "submission_path": str(submission_path),
            }
        )
        print(f"제출 파일           : {submission_path}")
    else:
        print("결합 모델이 snapping 단독보다 좋지 않아 제출 파일을 만들지 않습니다.")

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"결과 요약           : {metrics_path}")


if __name__ == "__main__":
    main()
