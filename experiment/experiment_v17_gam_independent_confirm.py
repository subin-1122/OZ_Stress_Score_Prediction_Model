"""
GAM 20% 블렌딩 독립 seed 확인
================================

v16 결과를 본 뒤 후보를 더 고르지 않고, 가장 좋았던 설정 하나만 고정한다.
- ExtraTrees 80% + flexible GAM 20%
- spline knots=7, QuantileRegressor alpha=0.0003

v16에서 사용하지 않은 seed 42, 123, 777의 20-fold OOF로 재검증한다.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (42, 123, 777)
N_SPLITS = 20
N_ESTIMATORS = 300
TRIM_RATIO = 0.10
GAM_WEIGHT = 0.20
GAM_N_KNOTS = 7
GAM_ALPHA = 0.0003
MIN_GAIN = 0.0003


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v16 = load_module(
        "experiment_v16",
        ROOT / "experiment/experiment_v16_gam_quantile.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored_oof = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    full_features = train.drop(columns=[ID_COLUMN, TARGET])
    tree_features = v3.make_model_features(train.drop(columns=[TARGET]))
    learned_mask = stored_oof["linked"].astype(bool).to_numpy()
    numeric = full_features.select_dtypes(include=np.number).columns.tolist()
    categorical = full_features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(full_features, numeric, categorical)

    results: list[dict] = []
    for seed in SEEDS:
        deterministic = v4.aggregate_oof_labels(rule_pairs, y, len(train), seed)
        unlinked = ~(learned_mask | np.isfinite(deterministic))
        tree_oof = np.zeros(len(train))
        gam_oof = np.zeros(len(train))
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        started = time.perf_counter()

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), 1):
            tree_preprocessor = v3.build_preprocessor(tree_features.iloc[train_idx])
            x_tree_train = tree_preprocessor.fit_transform(tree_features.iloc[train_idx])
            x_tree_valid = tree_preprocessor.transform(tree_features.iloc[valid_idx])
            tree = ExtraTreesRegressor(
                n_estimators=N_ESTIMATORS,
                criterion="squared_error",
                max_features=1,
                min_samples_leaf=1,
                bootstrap=False,
                n_jobs=-1,
                random_state=seed * 1000 + fold,
            )
            tree.fit(x_tree_train, y[train_idx])
            tree_oof[valid_idx] = v3.trimmed_tree_prediction(
                tree, x_tree_valid, TRIM_RATIO
            )

            gam_preprocessor = v16.build_gam_preprocessor(
                full_features.iloc[train_idx], GAM_N_KNOTS
            )
            x_gam_train = gam_preprocessor.fit_transform(full_features.iloc[train_idx])
            x_gam_valid = gam_preprocessor.transform(full_features.iloc[valid_idx])
            gam = QuantileRegressor(
                quantile=0.5,
                alpha=GAM_ALPHA,
                fit_intercept=True,
                solver="highs",
            )
            gam.fit(x_gam_train, y[train_idx])
            gam_oof[valid_idx] = np.clip(gam.predict(x_gam_valid), 0.0, 1.0)

            if fold % 5 == 0 or fold == N_SPLITS:
                print(
                    f"[GAM confirm] seed={seed} fold={fold:02d}/{N_SPLITS} "
                    f"elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )

        blend = (1.0 - GAM_WEIGHT) * tree_oof + GAM_WEIGHT * gam_oof
        tree_mae = mean_absolute_error(y[unlinked], tree_oof[unlinked])
        gam_mae = mean_absolute_error(y[unlinked], gam_oof[unlinked])
        blend_mae = mean_absolute_error(y[unlinked], blend[unlinked])
        overall_gain = (tree_mae - blend_mae) * float(unlinked.mean())
        results.append(
            {
                "seed": seed,
                "unlinked_rows": int(unlinked.sum()),
                "tree_unlinked_mae": float(tree_mae),
                "gam_unlinked_mae": float(gam_mae),
                "blend_unlinked_mae": float(blend_mae),
                "overall_equivalent_gain": float(overall_gain),
            }
        )

    gains = np.asarray([row["overall_equivalent_gain"] for row in results])
    metrics = {
        "fixed_candidate": {
            "tree_weight": 1.0 - GAM_WEIGHT,
            "gam_weight": GAM_WEIGHT,
            "gam_n_knots": GAM_N_KNOTS,
            "gam_alpha": GAM_ALPHA,
        },
        "protocol": {
            "seeds": list(SEEDS),
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "minimum_gain": MIN_GAIN,
            "test_used": False,
        },
        "gain_mean": float(gains.mean()),
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "passes_confirm": bool(gains.mean() >= MIN_GAIN and (gains > 0).all()),
        "results": results,
    }
    output_path = ROOT / "outputs/experiment_v17_gam_independent_confirm_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== GAM 독립 seed 확인 =====")
    for row in results:
        print(
            f"seed={row['seed']:3d} tree={row['tree_unlinked_mae']:.9f} "
            f"gam={row['gam_unlinked_mae']:.9f} "
            f"blend={row['blend_unlinked_mae']:.9f} "
            f"overall_gain={row['overall_equivalent_gain']:+.9f}"
        )
    print(
        f"평균 개선={gains.mean():+.9f}, "
        f"승/무/패={metrics['seed_wins']}/"
        f"{metrics['seed_ties']}/{metrics['seed_losses']}, "
        f"통과={metrics['passes_confirm']}"
    )
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
