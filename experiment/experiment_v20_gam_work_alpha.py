"""GAM 20% 혼합 모델 전용 mean_working alpha를 OOF에서 다시 확인한다."""

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
GRID_STEP = 0.01
CURRENT_ALPHA = 0.75
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


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
    v4 = load_module("v4", ROOT / "experiment/experiment_v4_deterministic_union.py")
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    cache = np.load(ROOT / "outputs/experiment_v18_gam_full_predictions.npz")
    y = train[TARGET].to_numpy(float)
    base = stored["base_oof"].to_numpy(float)
    blend = cache["blended_oof"]
    learned = stored["linked"].astype(bool).to_numpy()
    learned_label = stored["link_label"].to_numpy(float)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(features, numeric, categorical)

    records = []
    for seed in v4.LINK_SPLIT_SEEDS:
        det_label = v4.aggregate_oof_labels(pairs, y, len(train), seed)
        det = np.isfinite(det_label)
        union = learned | det
        label = learned_label.copy()
        label[det] = det_label[det]
        base_correction = v4.crossfit_work_correction(train, y, base, union)
        blend_correction = v4.crossfit_work_correction(train, y, blend, union)

        old = snap(np.clip(base + CURRENT_ALPHA * base_correction, 0, 1))
        old[union] = label[union]
        old_mae = mean_absolute_error(y, old)

        for alpha in ALPHAS:
            prediction = snap(np.clip(blend + alpha * blend_correction, 0, 1))
            prediction[union] = label[union]
            score = mean_absolute_error(y, prediction)
            records.append(
                {
                    "seed": int(seed),
                    "alpha": float(alpha),
                    "oof_mae": float(score),
                    "gain_vs_current": float(old_mae - score),
                }
            )

    frame = pd.DataFrame(records)
    summary = (
        frame.groupby("alpha")
        .agg(
            oof_mae_mean=("oof_mae", "mean"),
            gain_mean=("gain_vs_current", "mean"),
            gain_min=("gain_vs_current", "min"),
            gain_max=("gain_vs_current", "max"),
        )
        .reset_index()
    )
    pivot = frame.pivot(index="seed", columns="alpha", values="gain_vs_current")
    summary["seed_wins"] = [int((pivot[a] > 0).sum()) for a in summary["alpha"]]
    summary["seed_ties"] = [int((pivot[a] == 0).sum()) for a in summary["alpha"]]
    summary["seed_losses"] = [int((pivot[a] < 0).sum()) for a in summary["alpha"]]
    best = summary.loc[summary["oof_mae_mean"].idxmin()].to_dict()
    metrics = {
        "alphas": ALPHAS.tolist(),
        "comparison": "GAM20 blended base versus original base alpha=0.75",
        "best": best,
        "summary": summary.to_dict(orient="records"),
        "test_used": False,
    }
    output_path = ROOT / "outputs/experiment_v20_gam_work_alpha_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== GAM 전용 mean_working alpha =====")
    print(summary.to_string(index=False))
    print("\nBEST", json.dumps(best, ensure_ascii=False))
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
