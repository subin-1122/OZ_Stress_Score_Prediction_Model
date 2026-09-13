"""
v30 대조학습 최근접 후보의 순수 품질 진단
==========================================

v30의 nested gate가 대부분 no-op을 선택했기 때문에, 임계값이 지나치게
보수적이었던 것인지 임베딩 후보 자체에 신호가 없는 것인지 분리해서 확인한다.

주의
----
- test.csv를 읽지 않으며 train-only 5-fold OOF만 만든다.
- confidence 상위 비율별 결과를 모두 본 사후 스크리닝이므로, 이 파일의 가장
  좋은 숫자를 그대로 고르거나 제출에 사용하는 것은 금지한다.
- 목표는 새 제출 모델 선택이 아니라 v30의 실패 원인 진단이다.

실행
----
    .venv/bin/python experiment/experiment_v30b_contrastive_retrieval_diagnostic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 1.00)


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


def make_oof_retrieval(v30, features, y, pairs, equal_counts, seed):
    """각 validation 행을 제외한 fold-train encoder로 최근접 후보를 찾는다."""
    output = {
        "label": np.full(len(features), np.nan),
        "confidence": np.full(len(features), np.nan),
        "distance": np.full(len(features), np.nan),
        "margin": np.full(len(features), np.nan),
        "support": np.zeros(len(features), dtype=np.int16),
    }
    fold_rows = []
    splitter = KFold(
        n_splits=v30.OUTER_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(features),
        start=1,
    ):
        preprocessor, encoder, training = v30.fit_encoder(
            features,
            y,
            pairs,
            equal_counts,
            fit_indices,
            seed * 100 + fold,
        )
        stats = v30.retrieve(
            features,
            y,
            fit_indices,
            valid_indices,
            preprocessor,
            encoder,
        )
        output["label"][valid_indices] = stats["label"]
        output["confidence"][valid_indices] = stats["confidence"]
        output["distance"][valid_indices] = stats["scaled_distance"]
        output["margin"][valid_indices] = stats["scaled_margin"]
        output["support"][valid_indices] = stats["support"]
        fold_rows.append({"fold": fold, **training})
        print(
            f"[diagnostic] seed={seed} fold={fold}/{v30.OUTER_SPLITS} "
            f"positive={training['positive_pairs']}",
            flush=True,
        )
    return output, fold_rows


def evaluate(y, contexts, retrieval_by_seed):
    """사전에 고정한 confidence 상위 비율을 하나씩 독립적으로 비교한다."""
    rows = []
    for embedding_seed, stats in retrieval_by_seed.items():
        for context in contexts:
            available = ~context["mask"]
            available_confidence = stats["confidence"][available]
            current_mae = mean_absolute_error(y, context["current"])
            for fraction in FRACTIONS:
                threshold = (
                    -np.inf
                    if fraction == 1.0
                    else float(
                        np.quantile(available_confidence, 1.0 - fraction)
                    )
                )
                use = available & (stats["confidence"] >= threshold)
                candidate = context["current"].copy()
                candidate[use] = stats["label"][use]
                candidate_mae = mean_absolute_error(y, candidate)
                rows.append(
                    {
                        "embedding_seed": int(embedding_seed),
                        "link_seed": int(context["seed"]),
                        "top_fraction": fraction,
                        "selected_rows": int(use.sum()),
                        "selected_exact_accuracy": float(
                            np.mean(stats["label"][use] == y[use])
                        ),
                        "selected_copy_mae": float(
                            mean_absolute_error(y[use], stats["label"][use])
                        ),
                        "candidate_oof_mae": float(candidate_mae),
                        "gain_vs_current": float(current_mae - candidate_mae),
                    }
                )
    return rows


def main():
    torch.set_num_threads(1)
    v30 = __import__("experiment_v30_contrastive_linkage")
    v3 = v30.load_module(
        "experiment_v3_for_v30b",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = v30.load_module(
        "experiment_v4_for_v30b",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v30.load_module(
        "experiment_v24_for_v30b",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v3.train_candidate_pairs(features, v3.linkage_blocks(categorical))
    equal_counts = v30.pair_equal_counts(features, pairs)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    retrieval_by_seed = {}
    training = {}
    for seed in v30.EMBEDDING_SEEDS:
        retrieval, fold_rows = make_oof_retrieval(
            v30,
            features,
            y,
            pairs,
            equal_counts,
            int(seed),
        )
        retrieval_by_seed[int(seed)] = retrieval
        training[int(seed)] = fold_rows

    rows = evaluate(y, contexts, retrieval_by_seed)
    summary = []
    print("\n===== Contrastive retrieval diagnostic =====")
    for fraction in FRACTIONS:
        subset = [row for row in rows if row["top_fraction"] == fraction]
        gains = np.asarray([row["gain_vs_current"] for row in subset])
        accuracy = np.asarray(
            [row["selected_exact_accuracy"] for row in subset]
        )
        copy_mae = np.asarray([row["selected_copy_mae"] for row in subset])
        item = {
            "top_fraction": fraction,
            "gain_mean": float(gains.mean()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "exact_accuracy_mean": float(accuracy.mean()),
            "copy_mae_mean": float(copy_mae.mean()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
        }
        summary.append(item)
        print(
            f"confidence 상위 {fraction:>5.0%}: "
            f"gain={item['gain_mean']:+.9f}, "
            f"exact={item['exact_accuracy_mean']:.2%}, "
            f"copy_MAE={item['copy_mae_mean']:.6f}, "
            f"승/무/패={item['wins']}/{item['ties']}/{item['losses']}"
        )

    metrics = {
        "protocol": {
            "test_used": False,
            "purpose": "post-hoc diagnosis only; not for model selection",
            "fractions": FRACTIONS,
        },
        "training_diagnostics": training,
        "results": rows,
        "summary": summary,
        "submission_created": False,
    }
    path = ROOT / "outputs/experiment_v30b_contrastive_retrieval_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
