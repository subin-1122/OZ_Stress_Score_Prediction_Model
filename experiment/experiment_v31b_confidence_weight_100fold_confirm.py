"""
v31 confidence sample-weight 3-seed × 100-fold 확인
===================================================

v31의 사전 고정 네 설정 가운데 balanced_0p5_2p0만 10-fold 최종 파이프라인에서
연결 seed 5/5 개선했다. 이 파일은 해당 설정 하나를 현재 기준 모델과 동일한
3-seed × 100-fold에서 재확인한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 각 fold의 duplicate confidence, 결측치 중앙값, 인코더와 모델은 fold-train
  행으로만 계산/학습한다.
- validation target은 예측 후 OOF 평가에만 사용한다.
- 비교 대상 baseline은 동일한 3-seed × 100-fold로 저장된 기존 raw OOF다.
- 목표 0.0022를 넘지 않으면 제출 파일을 만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v31b_confidence_weight_100fold_confirm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (11, 101, 1001)
N_SPLITS = 100
LOW_WEIGHT = 0.50
HIGH_WEIGHT = 2.00
TARGET_OOF_GAIN = 0.0022


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def make_weighted_oof(
    v3,
    v31,
    train,
    y,
    model_features,
    pairs,
    equal_counts,
    seed,
):
    """한 seed의 100-fold weighted OOF를 만든다."""
    prediction = np.zeros(len(train), dtype=float)
    diagnostics = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    started = time.perf_counter()

    # 현재 기준 모델과 같은 seed별 tree 수를 사용한다.
    v31.N_ESTIMATORS = v3.N_ESTIMATORS_BY_SEED[seed]
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
        confidence, confidence_diag = v31.duplicate_confidence(
            train_idx,
            pairs,
            equal_counts,
            y,
        )
        sample_weight = v31.make_sample_weight(
            confidence,
            LOW_WEIGHT,
            HIGH_WEIGHT,
        )
        prediction[valid_idx] = v31.fit_predict(
            v3,
            model_features.iloc[train_idx],
            y[train_idx],
            model_features.iloc[valid_idx],
            seed * 1000 + fold,
            sample_weight,
        )
        diagnostics.append(
            {
                "fold": fold,
                **confidence_diag,
                "weight_minimum": float(sample_weight.min()),
                "weight_maximum": float(sample_weight.max()),
            }
        )
        if fold % 10 == 0 or fold == N_SPLITS:
            print(
                f"[100-fold] seed={seed} fold={fold:03d}/{N_SPLITS} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    return prediction, diagnostics


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v3 = v31.load_module(
        "experiment_v3_for_v31b",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v31.load_module(
        "experiment_v4_for_v31b",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v31b",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    raw_features = train.drop(columns=[ID_COLUMN, TARGET])
    model_features = v3.make_model_features(train.drop(columns=[TARGET]))
    categorical = raw_features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v3.train_candidate_pairs(
        raw_features,
        v3.linkage_blocks(categorical),
    )
    equal_counts = v31.pair_equal_counts(raw_features, pairs)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    weighted_by_seed = []
    training_diagnostics = {}
    started = time.perf_counter()
    for seed in SEEDS:
        prediction, diagnostics = make_weighted_oof(
            v3,
            v31,
            train,
            y,
            model_features,
            pairs,
            equal_counts,
            int(seed),
        )
        weighted_by_seed.append(prediction)
        training_diagnostics[int(seed)] = diagnostics

    weighted = np.mean(np.column_stack(weighted_by_seed), axis=1)
    baseline = stored["base_oof"].to_numpy(float)
    raw_baseline_mae = mean_absolute_error(y, baseline)
    raw_weighted_mae = mean_absolute_error(y, weighted)
    raw_gain = raw_baseline_mae - raw_weighted_mae

    context_results = []
    for context in contexts:
        current = context["current"]
        candidate = v24.finalize(
            v4,
            train,
            y,
            weighted,
            context["mask"],
            context["label"],
        )
        current_mae = mean_absolute_error(y, current)
        candidate_mae = mean_absolute_error(y, candidate)
        context_results.append(
            {
                "link_seed": int(context["seed"]),
                "current_oof_mae": float(current_mae),
                "candidate_oof_mae": float(candidate_mae),
                "gain_vs_current": float(current_mae - candidate_mae),
            }
        )

    gains = np.asarray([row["gain_vs_current"] for row in context_results])
    scores = np.asarray([row["candidate_oof_mae"] for row in context_results])
    passes = bool(gains.mean() >= TARGET_OOF_GAIN and np.all(gains > 0))
    metrics = {
        "protocol": {
            "test_used": False,
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "trees_by_seed": v3.N_ESTIMATORS_BY_SEED,
            "low_weight": LOW_WEIGHT,
            "high_weight": HIGH_WEIGHT,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "raw_baseline_oof_mae": float(raw_baseline_mae),
        "raw_weighted_oof_mae": float(raw_weighted_mae),
        "raw_gain": float(raw_gain),
        "context_results": context_results,
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean": float(gains.mean()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "reaches_target": passes,
            "submission_created": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        "training_diagnostics": training_diagnostics,
    }
    path = (
        ROOT
        / "outputs/experiment_v31b_confidence_weight_100fold_metrics.json"
    )
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v31b_confidence_weight_100fold_oof.npz",
        baseline=baseline,
        weighted=weighted,
        **{
            f"weighted_seed_{seed}": prediction
            for seed, prediction in zip(SEEDS, weighted_by_seed)
        },
    )

    print("\n===== 100-fold confidence-weight 확인 =====")
    print(
        f"raw baseline={raw_baseline_mae:.9f} "
        f"weighted={raw_weighted_mae:.9f} gain={raw_gain:+.9f}"
    )
    print(
        f"final OOF={scores.mean():.9f} gain={gains.mean():+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/"
        f"{(gains < 0).sum()} 목표도달={passes}"
    )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
