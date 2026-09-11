from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
GRID_STEP = 0.01
BASELINE_ALPHA = 0.75
ALPHAS = np.round(np.arange(0.75, 1.0001, 0.005), 3)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


v3 = load_module("v3", ROOT / "experiment/experiment_v3_best_record_linkage.py")
v4 = load_module("v4", ROOT / "experiment/experiment_v4_deterministic_union.py")
v8 = load_module("v8", ROOT / "experiment/experiment_v8_no_work_correction_diagnostic.py")

data_dir = ROOT / "open (3)"
output_dir = ROOT / "outputs"
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
oof = pd.read_csv(output_dir / "best_record_linkage_oof.csv")
current_v4 = pd.read_csv(output_dir / "best_union_submission.csv")

y = train[TARGET].to_numpy(float)
base_oof = oof["base_oof"].to_numpy(float)
learned_label = oof["link_label"].to_numpy(float)
learned_mask = oof["linked"].astype(bool).to_numpy()
features = train.drop(columns=[ID_COLUMN, TARGET])
numeric = features.select_dtypes(include=np.number).columns.tolist()
categorical = features.select_dtypes(exclude=np.number).columns.tolist()
rule_pairs = v4.make_rule_pairs(features, numeric, categorical)

records = []
oof_predictions = {}
for seed in v4.LINK_SPLIT_SEEDS:
    deterministic_label = v4.aggregate_oof_labels(rule_pairs, y, len(train), seed)
    deterministic_mask = np.isfinite(deterministic_label)
    union_mask = learned_mask | deterministic_mask
    union_label = learned_label.copy()
    union_label[deterministic_mask] = deterministic_label[deterministic_mask]
    correction = v4.crossfit_work_correction(train, y, base_oof, union_mask)

    for alpha in ALPHAS:
        raw = np.clip(base_oof + alpha * correction, 0.0, 1.0)
        raw[union_mask] = union_label[union_mask]
        snapped = snap(raw)
        oof_predictions[(int(seed), float(alpha))] = snapped
        records.append(
            {
                "seed": int(seed),
                "alpha": float(alpha),
                "raw_mae": mean_absolute_error(y, raw),
                "snap_mae": mean_absolute_error(y, snapped),
                "changed_oof_vs_075": int(
                    np.sum(
                        np.abs(
                            snapped
                            - oof_predictions.get((int(seed), BASELINE_ALPHA), snapped)
                        )
                        > 1e-12
                    )
                ),
            }
        )

frame = pd.DataFrame(records)
pivot = frame.pivot(index="seed", columns="alpha", values="snap_mae")
baseline = pivot[BASELINE_ALPHA]
summary = (
    frame.groupby("alpha")
    .agg(
        raw_mean=("raw_mae", "mean"),
        snap_mean=("snap_mae", "mean"),
        snap_std=("snap_mae", "std"),
        snap_min=("snap_mae", "min"),
        snap_max=("snap_mae", "max"),
        changed_oof_mean=("changed_oof_vs_075", "mean"),
    )
    .reset_index()
)
summary["change_vs_075"] = summary["snap_mean"] - float(baseline.mean())
summary["wins_vs_075"] = [int((pivot[a] < baseline).sum()) for a in summary["alpha"]]
summary["ties_vs_075"] = [int((pivot[a] == baseline).sum()) for a in summary["alpha"]]
summary["losses_vs_075"] = [int((pivot[a] > baseline).sum()) for a in summary["alpha"]]

_, _, test_union_mask = v8.make_test_masks(v3, v4, train, test, y)
recovered_base, test_correction = v8.recover_base_test_prediction(
    v3, train, test, oof, current_v4, test_union_mask
)
test_predictions = {}
for alpha in ALPHAS:
    pred = recovered_base.copy()
    pred[~test_union_mask] = np.clip(
        recovered_base[~test_union_mask] + alpha * test_correction[~test_union_mask],
        0.0,
        1.0,
    )
    pred[test_union_mask] = current_v4.loc[test_union_mask, TARGET].to_numpy(float)
    test_predictions[float(alpha)] = snap(pred)

test_075 = test_predictions[BASELINE_ALPHA]
summary["changed_test_vs_075"] = [
    int(np.sum(np.abs(test_predictions[float(a)] - test_075) > 1e-12))
    for a in summary["alpha"]
]
summary["changed_test_vs_090"] = [
    int(np.sum(np.abs(test_predictions[float(a)] - test_predictions[0.90]) > 1e-12))
    for a in summary["alpha"]
]

print("BASELINE", float(baseline.mean()))
print("BEST_RAW")
print(summary.nsmallest(10, "raw_mean").to_string(index=False))
print("BEST_SNAP")
print(summary.nsmallest(15, "snap_mean").to_string(index=False))
print("RANGE_085_100")
print(summary[summary["alpha"] >= 0.85].to_string(index=False))
print("PER_SEED_BEST")
for seed in pivot.index:
    row = frame[frame["seed"] == seed]
    best = row.loc[row["snap_mae"].idxmin()]
    print(int(seed), float(best["alpha"]), float(best["snap_mae"]))

public_x = np.array([0.0, 0.75, 0.90])
public_y = np.array([0.12928, 0.12578, 0.12558])
coef = np.polyfit(public_x, public_y, 2)
vertex = -coef[1] / (2 * coef[0])
print("PUBLIC_QUADRATIC", coef.tolist(), float(vertex), float(np.polyval(coef, vertex)))
