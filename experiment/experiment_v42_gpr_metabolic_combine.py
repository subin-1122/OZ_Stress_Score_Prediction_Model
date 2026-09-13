"""
GPR과 대사 복합 플래그의 OOF 결합 실험
=========================================

v32와 v35의 5-fold 스크리닝에서 방향이 가장 좋았던 두 후보를 결합한다.

- GPR 후보: Matern(l=8, nu=2.5, noise=0.10)을 unlinked에 25% 혼합
- 대사 후보: fold-train 60% 위험 경계 ExtraTrees를 75% 혼합
- mean_best: 두 최선 raw 예측의 단순 평균
- additive_best: 두 후보가 baseline에서 이동시킨 보정량을 모두 더함

두 후보 모두 train-only OOF이며 test.csv를 읽지 않는다. 새 성공 기준은 평균
OOF 개선 0.001이다. 이 스크리닝을 통과해야만 100-fold 재학습 대상으로 삼는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
TARGET_OOF_GAIN = 0.001
GPR_WEIGHT = 0.25
METABOLIC_WEIGHT = 0.75


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


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v4 = v31.load_module(
        "experiment_v4_for_v42",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v42",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    contexts = v24.make_link_contexts(v4, train, y, stored)
    canonical_unlinked = ~contexts[0]["mask"]
    baseline = stored["base_oof"].to_numpy(float)

    gpr_cache = np.load(ROOT / "outputs/experiment_v32_gaussian_process_oof.npz")
    metabolic_cache = np.load(ROOT / "outputs/experiment_v35_metabolic_flags_oof.npz")
    gpr = gpr_cache["prediction_matern_l8_n010"]
    metabolic = metabolic_cache["quantile_60"]
    if not (
        len(gpr) == len(train)
        and len(metabolic) == len(train)
        and np.isfinite(gpr).all()
        and np.isfinite(metabolic).all()
    ):
        raise ValueError("저장된 OOF 후보의 길이 또는 값이 올바르지 않습니다.")

    gpr_best = baseline.copy()
    gpr_best[canonical_unlinked] = (
        (1.0 - GPR_WEIGHT) * baseline[canonical_unlinked]
        + GPR_WEIGHT * gpr[canonical_unlinked]
    )
    metabolic_best = (
        (1.0 - METABOLIC_WEIGHT) * baseline + METABOLIC_WEIGHT * metabolic
    )
    candidates = {
        "mean_best": 0.5 * (gpr_best + metabolic_best),
        "additive_best": baseline + (gpr_best - baseline) + (metabolic_best - baseline),
    }

    results = []
    for name, raw in candidates.items():
        rows = []
        for context in contexts:
            prediction = v24.finalize(
                v4, train, y, raw, context["mask"], context["label"]
            )
            current_mae = mean_absolute_error(y, context["current"])
            score = mean_absolute_error(y, prediction)
            rows.append(
                {
                    "link_seed": int(context["seed"]),
                    "candidate_oof_mae": float(score),
                    "gain_vs_current": float(current_mae - score),
                }
            )
        gains = np.asarray([row["gain_vs_current"] for row in rows])
        scores = np.asarray([row["candidate_oof_mae"] for row in rows])
        results.append(
            {
                "candidate": name,
                "candidate_oof_mae_mean": float(scores.mean()),
                "gain_mean": float(gains.mean()),
                "gain_min": float(gains.min()),
                "gain_max": float(gains.max()),
                "wins": int((gains > 0).sum()),
                "ties": int((gains == 0).sum()),
                "losses": int((gains < 0).sum()),
                "passes_screen": bool(gains.mean() >= TARGET_OOF_GAIN),
                "context_results": rows,
            }
        )
    results.sort(key=lambda row: row["gain_mean"], reverse=True)

    # 두 후보 이동량이 같은 정보를 고치는지 확인하는 보조 진단이다.
    unlinked = canonical_unlinked
    gpr_shift = gpr_best[unlinked] - baseline[unlinked]
    metabolic_shift = metabolic_best[unlinked] - baseline[unlinked]
    residual = y[unlinked] - baseline[unlinked]
    diagnostics = {
        "shift_correlation": float(np.corrcoef(gpr_shift, metabolic_shift)[0, 1]),
        "gpr_shift_residual_correlation": float(np.corrcoef(gpr_shift, residual)[0, 1]),
        "metabolic_shift_residual_correlation": float(
            np.corrcoef(metabolic_shift, residual)[0, 1]
        ),
    }
    metrics = {
        "protocol": {
            "test_used": False,
            "gpr_weight": GPR_WEIGHT,
            "metabolic_weight": METABOLIC_WEIGHT,
            "target_oof_gain": TARGET_OOF_GAIN,
            "candidate_count": len(candidates),
        },
        "diagnostics": diagnostics,
        "results": results,
        "best": results[0],
        "submission_created": False,
    }
    path = ROOT / "outputs/experiment_v42_gpr_metabolic_combine_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v42_gpr_metabolic_combine_oof.npz",
        baseline=baseline,
        gpr_best=gpr_best,
        metabolic_best=metabolic_best,
        **candidates,
    )

    print("===== GPR + metabolic combination =====")
    for row in results:
        print(
            f"{row['candidate']:14s} OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print(
        f"shift correlation={diagnostics['shift_correlation']:+.4f}; "
        f"성공 기준={TARGET_OOF_GAIN:.4f}"
    )
    print("test를 읽지 않았으며 이 단계에서는 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
