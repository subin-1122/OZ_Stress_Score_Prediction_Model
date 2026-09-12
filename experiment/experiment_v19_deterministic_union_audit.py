"""
Deterministic union 규칙별 OOF 감사
===================================

learned 연결만 쓴 파이프라인과 deterministic 연결을 추가한 파이프라인을
동일한 100-fold x 5-seed에서 비교한다. 기존 규칙보다 엄격한 두 규칙도
사전에 고정해 확인하지만, 개선폭 0.0001과 5/5 seed 개선을 통과하기 전에는
제출 파일을 만들지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
WORK_ALPHA = 0.75
GRID_STEP = 0.01
MIN_GAIN = 0.0001


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def make_final_prediction(v4, train, y, base_oof, learned_label, learned_mask, pairs, seed):
    deterministic_label = v4.aggregate_oof_labels(pairs, y, len(train), seed)
    deterministic_mask = np.isfinite(deterministic_label)
    union_mask = learned_mask | deterministic_mask
    union_label = learned_label.copy()
    union_label[deterministic_mask] = deterministic_label[deterministic_mask]
    correction = v4.crossfit_work_correction(train, y, base_oof, union_mask)
    prediction = snap(np.clip(base_oof + WORK_ALPHA * correction, 0.0, 1.0))
    prediction[union_mask] = union_label[union_mask]
    return prediction, deterministic_label, deterministic_mask, union_mask


def main() -> None:
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    base_oof = stored["base_oof"].to_numpy(float)
    learned_label = stored["link_label"].to_numpy(float)
    learned_mask = stored["linked"].astype(bool).to_numpy()
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()

    current_pairs = v4.make_rule_pairs(features, numeric, categorical)
    numeric_equal, core_equal = v4.equality_counts(features, current_pairs, numeric)
    pair_sets = {
        "current_num2_core1": current_pairs,
        "strict_num3_core1": current_pairs[numeric_equal >= 3],
        "strict_num2_core2": current_pairs[core_equal >= 2],
        "strict_num3_or_core2": current_pairs[
            (numeric_equal >= 3) | (core_equal >= 2)
        ],
    }

    repeated: list[dict] = []
    for seed in v4.LINK_SPLIT_SEEDS:
        # learned-only 기준선도 같은 seed의 correction fold로 계산한다.
        learned_correction = v4.crossfit_work_correction(
            train, y, base_oof, learned_mask
        )
        learned_prediction = snap(
            np.clip(base_oof + WORK_ALPHA * learned_correction, 0.0, 1.0)
        )
        learned_prediction[learned_mask] = learned_label[learned_mask]
        learned_mae = mean_absolute_error(y, learned_prediction)

        variants: dict[str, dict] = {}
        for name, pairs in pair_sets.items():
            prediction, deterministic_label, deterministic_mask, union_mask = (
                make_final_prediction(
                    v4,
                    train,
                    y,
                    base_oof,
                    learned_label,
                    learned_mask,
                    pairs,
                    int(seed),
                )
            )
            added = deterministic_mask & ~learned_mask
            conflicts = (
                deterministic_mask
                & learned_mask
                & (deterministic_label != learned_label)
            )
            score = mean_absolute_error(y, prediction)
            variants[name] = {
                "pair_count": int(len(pairs)),
                "oof_mae": float(score),
                "gain_vs_learned_only": float(learned_mae - score),
                "deterministic_rows": int(deterministic_mask.sum()),
                "added_rows": int(added.sum()),
                "added_exact_accuracy": (
                    float(np.mean(deterministic_label[added] == y[added]))
                    if added.any()
                    else None
                ),
                "added_mae": (
                    float(mean_absolute_error(y[added], deterministic_label[added]))
                    if added.any()
                    else None
                ),
                "conflicts_with_learned": int(conflicts.sum()),
                "union_rows": int(union_mask.sum()),
            }
        repeated.append(
            {
                "seed": int(seed),
                "learned_only_oof_mae": float(learned_mae),
                "variants": variants,
            }
        )

    summary = {}
    for name in pair_sets:
        gains = np.asarray(
            [row["variants"][name]["gain_vs_learned_only"] for row in repeated]
        )
        scores = np.asarray([row["variants"][name]["oof_mae"] for row in repeated])
        accuracies = [
            row["variants"][name]["added_exact_accuracy"] for row in repeated
        ]
        summary[name] = {
            "oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_learned_only": float(gains.mean()),
            "seed_wins": int((gains > 0).sum()),
            "seed_ties": int((gains == 0).sum()),
            "seed_losses": int((gains < 0).sum()),
            "added_accuracy_mean": float(np.mean(accuracies)),
            "passes_gate": bool(gains.mean() >= MIN_GAIN and (gains > 0).all()),
        }

    metrics = {
        "protocol": {
            "work_alpha": WORK_ALPHA,
            "grid_step": GRID_STEP,
            "minimum_gain": MIN_GAIN,
            "requires_all_five_seed_wins": True,
            "test_used": False,
        },
        "learned_linked_rows": int(learned_mask.sum()),
        "learned_linked_exact_accuracy": float(
            np.mean(learned_label[learned_mask] == y[learned_mask])
        ),
        "summary": summary,
        "repeated": repeated,
    }
    output_path = ROOT / "outputs/experiment_v19_deterministic_union_audit_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== deterministic union 감사 =====")
    print(
        f"learned 연결={learned_mask.sum():,}, "
        f"정확도={metrics['learned_linked_exact_accuracy']:.6f}"
    )
    for name, result in summary.items():
        print(
            f"{name:22s} OOF={result['oof_mae_mean']:.9f} "
            f"gain={result['gain_mean_vs_learned_only']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"added_acc={result['added_accuracy_mean']:.4f} "
            f"통과={result['passes_gate']}"
        )
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
