"""GAM 개선이 최종 파이프라인에서 사라진 위치를 층별로 확인한다."""

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
ALPHA = 0.75
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

    results = []
    for seed in v4.LINK_SPLIT_SEEDS:
        det_label = v4.aggregate_oof_labels(pairs, y, len(train), seed)
        det = np.isfinite(det_label)
        union = learned | det
        unlinked = ~union
        label = learned_label.copy()
        label[det] = det_label[det]

        base_corr = v4.crossfit_work_correction(train, y, base, union)
        blend_corr = v4.crossfit_work_correction(train, y, blend, union)
        layers = {
            "base_raw_unlinked": mean_absolute_error(y[unlinked], base[unlinked]),
            "blend_raw_unlinked": mean_absolute_error(y[unlinked], blend[unlinked]),
            "base_work_raw_unlinked": mean_absolute_error(
                y[unlinked], np.clip(base + ALPHA * base_corr, 0, 1)[unlinked]
            ),
            "blend_work_raw_unlinked": mean_absolute_error(
                y[unlinked], np.clip(blend + ALPHA * blend_corr, 0, 1)[unlinked]
            ),
        }
        base_final = snap(np.clip(base + ALPHA * base_corr, 0, 1))
        blend_final = snap(np.clip(blend + ALPHA * blend_corr, 0, 1))
        base_final[union] = label[union]
        blend_final[union] = label[union]
        layers["base_final"] = mean_absolute_error(y, base_final)
        layers["blend_final"] = mean_absolute_error(y, blend_final)
        results.append({"seed": int(seed), **{k: float(v) for k, v in layers.items()}})

    keys = [key for key in results[0] if key != "seed"]
    means = {key: float(np.mean([row[key] for row in results])) for key in keys}
    diagnostics = {
        "means": means,
        "raw_unlinked_gain": means["base_raw_unlinked"] - means["blend_raw_unlinked"],
        "work_raw_unlinked_gain": (
            means["base_work_raw_unlinked"] - means["blend_work_raw_unlinked"]
        ),
        "final_gain": means["base_final"] - means["blend_final"],
        "results": results,
    }
    output_path = ROOT / "outputs/experiment_v18b_gam_layer_diagnostic_metrics.json"
    output_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
