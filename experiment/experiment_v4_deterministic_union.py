"""
오늘의 제출 후보: 학습형 연결 + 설명 가능한 결정 규칙 합집합
================================================================

이 파일은 experiment_v3_best_record_linkage.py를 먼저 실행해 만들어진
다음 두 파일을 입력으로 사용한다.

* outputs/best_record_linkage_submission.csv
* outputs/best_record_linkage_oof.csv

추가 규칙
---------
아래 조건을 모두 만족하는 test 행과 train 행만 같은 기록 후보로 본다.

1. gender, activity 등 범주형 7개 값이 모두 같다.
2. 숫자형 9개 값 중 2개 이상이 정확히 같다. 결측끼리도 같은 값으로 센다.
3. height, weight, cholesterol, glucose 중 1개 이상이 정확히 같다.
4. 후보 train 행이 여러 개면 그 행들의 stress_score가 모두 같아야 한다.

이 규칙은 train-only OOF에서 5개 분할 seed로 반복 검증했다. test의 정답,
test 통계량, test-test 관계는 사용하지 않는다. 최종 파일은 기존 학습형 연결
예측에 위 결정 규칙이 확실하게 찾은 행만 추가로 덮어쓴 합집합 예측이다.

실행
----
VS Code에서 이 파일을 열고 "Python 파일 실행" 버튼을 누르거나:

    .venv/bin/python experiment/experiment_v4_deterministic_union.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


TARGET = "stress_score"
ID_COLUMN = "ID"
CORE_COLUMNS = ["height", "weight", "cholesterol", "glucose"]
LINK_SPLIT_SEEDS = (11, 101, 1001, 2026, 31415)
CORRECTION_SEEDS = (11, 101, 1001)
N_SPLITS = 100
WORK_CORRECTION_ALPHA = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습형 연결과 결정 규칙을 합친 오늘의 제출 파일을 만듭니다."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-submission", type=Path, default=None)
    parser.add_argument("--base-oof", type=Path, default=None)
    return parser.parse_args()


def find_project_root() -> Path:
    candidates: list[Path] = []
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend([start, *start.parents])
    for candidate in candidates:
        if (candidate / "open (3)" / "train.csv").exists():
            return candidate
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


def row_equal(left: pd.Series, right: pd.Series) -> bool:
    """두 값이 같거나 둘 다 결측이면 같은 것으로 판단한다."""
    return bool((left == right) or (pd.isna(left) and pd.isna(right)))


def equality_counts(
    frame: pd.DataFrame,
    pairs: np.ndarray,
    numeric_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """행 쌍마다 전체 숫자와 핵심 숫자가 각각 몇 개 같은지 계산한다."""
    left = frame.iloc[pairs[:, 0]].reset_index(drop=True)
    right = frame.iloc[pairs[:, 1]].reset_index(drop=True)

    def equal_array(column: str) -> np.ndarray:
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        return (a == b) | (pd.isna(a) & pd.isna(b))

    numeric_equal = np.column_stack(
        [equal_array(column) for column in numeric_columns]
    ).sum(axis=1)
    core_equal = np.column_stack(
        [equal_array(column) for column in CORE_COLUMNS]
    ).sum(axis=1)
    return numeric_equal, core_equal


def make_rule_pairs(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    """
    모든 범주형 값이 같은 행끼리만 먼저 묶고 숫자 일치 조건을 적용한다.

    이 함수는 feature만 사용한다. stress_score는 후보 생성에 사용하지 않는다.
    """
    pairs: set[tuple[int, int]] = set()
    grouper: str | list[str] = (
        categorical_columns[0]
        if len(categorical_columns) == 1
        else categorical_columns
    )
    for members in features.groupby(grouper, dropna=False, sort=False).groups.values():
        pairs.update(combinations(list(members), 2))

    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    candidate = np.asarray(sorted(pairs), dtype=np.int32)
    numeric_equal, core_equal = equality_counts(
        features, candidate, numeric_columns
    )
    keep = (numeric_equal >= 2) & (core_equal >= 1)
    return candidate[keep]


def aggregate_oof_labels(
    pairs: np.ndarray,
    y: np.ndarray,
    n_rows: int,
    seed: int,
) -> np.ndarray:
    """각 validation 행을 제외한 train 후보들이 만장일치할 때만 점수를 연결한다."""
    result = np.full(n_rows, np.nan)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for train_idx, valid_idx in splitter.split(np.arange(n_rows)):
        in_train = np.zeros(n_rows, dtype=bool)
        in_valid = np.zeros(n_rows, dtype=bool)
        in_train[train_idx] = True
        in_valid[valid_idx] = True
        cross = (
            (in_valid[pairs[:, 0]] & in_train[pairs[:, 1]])
            | (in_valid[pairs[:, 1]] & in_train[pairs[:, 0]])
        )
        query = pairs[cross].copy()
        reverse = in_train[query[:, 0]]
        query[reverse] = query[reverse][:, ::-1]
        order = np.argsort(query[:, 0], kind="stable")
        query = query[order]
        rows, starts = np.unique(query[:, 0], return_index=True)
        ends = np.r_[starts[1:], len(query)]
        for row, start, end in zip(rows, starts, ends):
            proposed = np.unique(y[query[start:end, 1]])
            if len(proposed) == 1:
                result[row] = proposed[0]
    return result


def crossfit_work_correction(
    train: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    linked: np.ndarray,
) -> np.ndarray:
    """각 행을 제외한 fold에서 근로시간별 잔차 중앙값을 계산한다."""
    keys = train["mean_working"].astype("string").fillna("__MISSING__")
    corrections: list[np.ndarray] = []
    for seed in CORRECTION_SEEDS:
        correction = np.zeros(len(train), dtype=float)
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, valid_idx in splitter.split(train):
            usable = train_idx[~linked[train_idx]]
            residual = y[usable] - base_oof[usable]
            table = pd.DataFrame(
                {"key": keys.iloc[usable].to_numpy(), "residual": residual}
            ).groupby("key")["residual"].median()
            fallback = float(np.median(residual))
            correction[valid_idx] = (
                keys.iloc[valid_idx].map(table).fillna(fallback).to_numpy(float)
            )
        corrections.append(correction)
    return np.mean(np.column_stack(corrections), axis=1)


def categorical_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """결측을 명시적으로 포함한 범주형 조합 키를 만든다."""
    converted = frame[columns].astype("string").fillna("__MISSING__")
    return converted.agg("||".join, axis=1)


def make_test_rule_labels(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    """test마다 결정 규칙을 만족하는 train 후보들의 만장일치 점수를 구한다."""
    labels = np.full(len(test_features), np.nan)
    train_keys = categorical_key(train_features, categorical_columns)
    test_keys = categorical_key(test_features, categorical_columns)
    train_groups = pd.Series(train_features.index, index=train_keys).groupby(level=0)
    test_groups = pd.Series(test_features.index, index=test_keys).groupby(level=0)
    train_group_indices = {key: values.to_numpy() for key, values in train_groups}

    for key, test_rows_series in test_groups:
        train_rows = train_group_indices.get(key)
        if train_rows is None:
            continue
        test_rows = test_rows_series.to_numpy()
        train_num = train_features.loc[train_rows, numeric_columns].to_numpy()
        test_num = test_features.loc[test_rows, numeric_columns].to_numpy()
        equal = (
            (test_num[:, None, :] == train_num[None, :, :])
            | (
                pd.isna(test_num[:, None, :])
                & pd.isna(train_num[None, :, :])
            )
        )
        numeric_equal = equal.sum(axis=2)
        core_positions = [numeric_columns.index(column) for column in CORE_COLUMNS]
        core_equal = equal[:, :, core_positions].sum(axis=2)
        eligible = (numeric_equal >= 2) & (core_equal >= 1)

        for local_test, test_row in enumerate(test_rows):
            candidates = train_rows[eligible[local_test]]
            if len(candidates) == 0:
                continue
            proposed = np.unique(y[candidates])
            if len(proposed) == 1:
                labels[test_row] = proposed[0]
    return labels


def main() -> None:
    args = parse_args()
    root = find_project_root()
    data_dir = args.data_dir.resolve() if args.data_dir else root / "open (3)"
    output_dir = args.output_dir.resolve() if args.output_dir else root / "outputs"
    base_submission_path = (
        args.base_submission.resolve()
        if args.base_submission
        else output_dir / "best_record_linkage_submission.csv"
    )
    base_oof_path = (
        args.base_oof.resolve()
        if args.base_oof
        else output_dir / "best_record_linkage_oof.csv"
    )

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    base_submission = pd.read_csv(base_submission_path)
    oof_detail = pd.read_csv(base_oof_path)
    y = train[TARGET].to_numpy(float)
    base_oof = oof_detail["base_oof"].to_numpy(float)
    learned_label = oof_detail["link_label"].to_numpy(float)
    learned_mask = oof_detail["linked"].astype(bool).to_numpy()

    train_features = train.drop(columns=[ID_COLUMN, TARGET])
    test_features = test.drop(columns=[ID_COLUMN])
    numeric = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical = train_features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = make_rule_pairs(train_features, numeric, categorical)

    repeated_scores: list[float] = []
    repeated_coverages: list[float] = []
    for seed in LINK_SPLIT_SEEDS:
        deterministic_label = aggregate_oof_labels(rule_pairs, y, len(train), seed)
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]
        correction = crossfit_work_correction(
            train, y, base_oof, union_mask
        )
        prediction = np.clip(
            base_oof + WORK_CORRECTION_ALPHA * correction, 0, 1
        )
        prediction[union_mask] = union_label[union_mask]
        score = mean_absolute_error(y, prediction)
        repeated_scores.append(float(score))
        repeated_coverages.append(float(union_mask.mean()))
        print(
            f"seed={seed} union_coverage={union_mask.mean():.2%} "
            f"OOF_MAE={score:.9f}",
            flush=True,
        )

    # 위에서 train OOF로 확정한 동일 규칙을 test에 그대로 적용한다.
    test_rule_label = make_test_rule_labels(
        train_features,
        test_features,
        y,
        numeric,
        categorical,
    )
    test_rule_mask = np.isfinite(test_rule_label)
    final_submission = base_submission.copy()
    before = final_submission[TARGET].to_numpy(float).copy()
    final_submission.loc[test_rule_mask, TARGET] = test_rule_label[test_rule_mask]
    after = final_submission[TARGET].to_numpy(float)

    if list(final_submission.columns) != list(sample.columns):
        raise ValueError("제출 파일의 열이 sample_submission과 다릅니다.")
    if not final_submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("제출 파일의 ID 또는 순서가 다릅니다.")
    if not np.isfinite(after).all() or ((after < 0) | (after > 1)).any():
        raise ValueError("예측값에 NaN/무한대가 있거나 0~1 범위를 벗어났습니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "best_union_submission.csv"
    metrics_path = output_dir / "best_union_metrics.json"
    final_submission.to_csv(submission_path, index=False)
    metrics = {
        "method": "learned linkage + deterministic rule union",
        "rule": "all 7 categorical equal, >=2 numeric equal, >=1 core equal, unanimous train target",
        "oof_mae_mean": float(np.mean(repeated_scores)),
        "oof_mae_min": float(np.min(repeated_scores)),
        "oof_mae_max": float(np.max(repeated_scores)),
        "oof_union_coverage_mean": float(np.mean(repeated_coverages)),
        "test_deterministic_rows": int(test_rule_mask.sum()),
        "changed_from_previous_submission": int(
            np.sum(np.abs(after - before) > 1e-12)
        ),
        "previous_public_mae": 0.1258158348,
        "provisional_public_gap": 0.007281544157580824,
        "estimated_public_mae": float(
            np.mean(repeated_scores) + 0.007281544157580824
        ),
        "submission_path": str(submission_path),
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== 오늘의 제출 후보 =====")
    print(f"반복 OOF 평균       : {np.mean(repeated_scores):.9f}")
    print(
        f"반복 OOF 범위       : {np.min(repeated_scores):.9f}"
        f" ~ {np.max(repeated_scores):.9f}"
    )
    print(f"결정 규칙 test 연결 : {test_rule_mask.sum():,}/{len(test):,}")
    print(f"기존 제출과 다른 행 : {np.sum(np.abs(after-before)>1e-12):,}")
    print(f"제출 파일           : {submission_path}")
    print(f"결과 요약           : {metrics_path}")


if __name__ == "__main__":
    main()
