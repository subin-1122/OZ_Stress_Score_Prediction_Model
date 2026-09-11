"""
alpha 구간 평균 실험
====================

목적
----
mean_working 보정 강도 0.75, 0.80, 0.85, 0.90을 하나씩 고르는 대신
네 예측을 평균했을 때 OOF가 안정적으로 좋아지는지 확인한다.

중요한 성질
-----------
snapping 전에 예측을 평균하면 수식상 단일 alpha=0.825와 같다.
따라서 이 파일은 0.01 snapping을 각각 먼저 적용한 뒤 평균하고 다시 snapping한
결과가 단일 alpha=0.825와 실제로 달라지는지만 검증한다.

누수 방지
---------
- 이미 train에서 만든 OOF 예측만 사용한다.
- test 결과나 DACON Public 점수로 조합을 선택하지 않는다.
- 사전에 고정한 네 alpha 외에 결과를 보고 후보를 추가하지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ALPHAS = (0.75, 0.80, 0.85, 0.90)
EQUIVALENT_ALPHA = float(np.mean(COMPONENT_ALPHAS))
GRID_STEP = 0.01


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def main() -> None:
    # v10에는 동일한 OOF 프로토콜로 계산한 alpha별 예측이 들어 있다.
    v10 = load_module(
        "experiment_v10",
        ROOT / "experiment/experiment_v10_alpha_fine_compare.py",
    )

    seed_results: list[dict] = []
    for seed in v10.v4.LINK_SPLIT_SEEDS:
        component_predictions = np.column_stack(
            [v10.oof_predictions[(int(seed), alpha)] for alpha in COMPONENT_ALPHAS]
        )
        snap_then_average = snap(component_predictions.mean(axis=1))
        direct_0825 = v10.oof_predictions[(int(seed), EQUIVALENT_ALPHA)]
        baseline_075 = v10.oof_predictions[(int(seed), 0.75)]

        ensemble_mae = mean_absolute_error(v10.y, snap_then_average)
        direct_mae = mean_absolute_error(v10.y, direct_0825)
        baseline_mae = mean_absolute_error(v10.y, baseline_075)
        seed_results.append(
            {
                "seed": int(seed),
                "ensemble_oof_mae": float(ensemble_mae),
                "direct_alpha0825_oof_mae": float(direct_mae),
                "alpha075_oof_mae": float(baseline_mae),
                "ensemble_gain_vs_alpha075": float(baseline_mae - ensemble_mae),
                "ensemble_change_vs_alpha0825": float(ensemble_mae - direct_mae),
                "rows_different_from_alpha0825": int(
                    np.sum(np.abs(snap_then_average - direct_0825) > 1e-12)
                ),
            }
        )

    # test에서는 값의 차이만 진단한다. 이 값으로 채택 여부를 결정하지 않는다.
    test_components = np.column_stack(
        [v10.test_predictions[alpha] for alpha in COMPONENT_ALPHAS]
    )
    test_ensemble = snap(test_components.mean(axis=1))
    test_direct = v10.test_predictions[EQUIVALENT_ALPHA]
    test_alpha090 = v10.test_predictions[0.90]

    gains = np.array(
        [item["ensemble_gain_vs_alpha075"] for item in seed_results], dtype=float
    )
    ensemble_mean = float(
        np.mean([item["ensemble_oof_mae"] for item in seed_results])
    )
    baseline_mean = float(
        np.mean([item["alpha075_oof_mae"] for item in seed_results])
    )
    metrics = {
        "component_alphas": list(COMPONENT_ALPHAS),
        "equivalent_raw_alpha": EQUIVALENT_ALPHA,
        "alpha075_oof_mae": baseline_mean,
        "ensemble_oof_mae": ensemble_mean,
        "ensemble_gain_vs_alpha075": baseline_mean - ensemble_mean,
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "passes_5_of_5": bool((gains > 0).all()),
        "passes_minimum_gain_0_0001": bool(baseline_mean - ensemble_mean >= 0.0001),
        "test_rows_different_from_direct_alpha0825": int(
            np.sum(np.abs(test_ensemble - test_direct) > 1e-12)
        ),
        "test_rows_different_from_alpha090": int(
            np.sum(np.abs(test_ensemble - test_alpha090) > 1e-12)
        ),
        "seed_results": seed_results,
    }

    output_path = ROOT / "outputs/experiment_v12_alpha_ensemble_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== alpha 평균 OOF 결과 =====")
    print(f"alpha=0.75 OOF             : {baseline_mean:.12f}")
    print(f"네 alpha 평균 OOF          : {ensemble_mean:.12f}")
    print(f"개선폭                      : {baseline_mean-ensemble_mean:+.12f}")
    print(
        "seed 승/무/패               : "
        f"{metrics['seed_wins']}/{metrics['seed_ties']}/{metrics['seed_losses']}"
    )
    print(
        "단일 alpha=0.825와 다른 test 행: "
        f"{metrics['test_rows_different_from_direct_alpha0825']:,}"
    )
    print(f"결과 요약                   : {output_path}")


if __name__ == "__main__":
    main()
