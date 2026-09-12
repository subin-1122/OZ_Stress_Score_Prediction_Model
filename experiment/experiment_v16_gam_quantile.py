"""
Unlinked 전용 GAM 형태의 스플라인 분위수 회귀 실험
====================================================

트리와 다른 가산형 구조를 검증한다. 숫자형 변수마다 cubic spline을 만들고,
범주형 변수는 one-hot encoding한 뒤 중앙값(quantile=0.5) 회귀를 학습한다.

동일한 10-fold x 5-seed에서 ExtraTrees baseline과 비교하며, 최종 파이프라인에서
연결되지 않은 validation 행의 MAE만 평가한다. GAM 단독과 ExtraTrees 80% + GAM
20%의 사전 고정 블렌딩을 함께 확인한다.

모든 imputer, spline, one-hot 규칙은 fold-train에서만 fit한다.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
SEEDS = (11, 101, 1001, 2026, 31415)
N_SPLITS = 10
N_ESTIMATORS = 250
TRIM_RATIO = 0.10
BLEND_WEIGHT_GAM = 0.20
MIN_SUBMISSION_GAIN = 0.0003

# 서로 다른 복잡도를 사전에 두 개만 고정한다.
GAM_CONFIGS = {
    "smooth": {"n_knots": 5, "alpha": 0.0010},
    "flexible": {"n_knots": 7, "alpha": 0.0003},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_gam_preprocessor(
    fold_train: pd.DataFrame,
    n_knots: int,
) -> ColumnTransformer:
    categorical = fold_train.select_dtypes(exclude=np.number).columns.tolist()
    numeric = [column for column in fold_train.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


def main() -> None:
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
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

    seed_results: list[dict] = []
    for seed in SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            rule_pairs, y, len(train), seed
        )
        union_mask = learned_mask | np.isfinite(deterministic_label)
        unlinked = ~union_mask
        predictions = {"extratrees": np.zeros(len(train))}
        for config_name in GAM_CONFIGS:
            predictions[f"gam_{config_name}"] = np.zeros(len(train))

        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        started = time.perf_counter()
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), 1):
            # ExtraTrees baseline도 같은 fold에서 다시 계산해 공정하게 비교한다.
            tree_preprocessor = v3.build_preprocessor(tree_features.iloc[train_idx])
            x_tree_train = tree_preprocessor.fit_transform(
                tree_features.iloc[train_idx]
            )
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
            predictions["extratrees"][valid_idx] = v3.trimmed_tree_prediction(
                tree, x_tree_valid, TRIM_RATIO
            )

            for config_name, config in GAM_CONFIGS.items():
                preprocessor = build_gam_preprocessor(
                    full_features.iloc[train_idx], int(config["n_knots"])
                )
                x_train = preprocessor.fit_transform(full_features.iloc[train_idx])
                x_valid = preprocessor.transform(full_features.iloc[valid_idx])
                model = QuantileRegressor(
                    quantile=0.5,
                    alpha=float(config["alpha"]),
                    fit_intercept=True,
                    solver="highs",
                )
                model.fit(x_train, y[train_idx])
                predictions[f"gam_{config_name}"][valid_idx] = np.clip(
                    model.predict(x_valid), 0.0, 1.0
                )

            print(
                f"[GAM] seed={seed} fold={fold:02d}/{N_SPLITS} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

        baseline_mae = mean_absolute_error(
            y[unlinked], predictions["extratrees"][unlinked]
        )
        variants: dict[str, dict] = {}
        for config_name in GAM_CONFIGS:
            gam = predictions[f"gam_{config_name}"]
            blend = (
                (1.0 - BLEND_WEIGHT_GAM) * predictions["extratrees"]
                + BLEND_WEIGHT_GAM * gam
            )
            for variant_name, candidate in (
                (f"gam_{config_name}", gam),
                (f"blend20_{config_name}", blend),
            ):
                score = mean_absolute_error(y[unlinked], candidate[unlinked])
                # 전체 3,000행 기준 기대 개선으로 환산한다.
                overall_gain = (baseline_mae - score) * float(unlinked.mean())
                variants[variant_name] = {
                    "unlinked_mae": float(score),
                    "overall_equivalent_gain": float(overall_gain),
                }
        seed_results.append(
            {
                "seed": int(seed),
                "unlinked_rows": int(unlinked.sum()),
                "baseline_unlinked_mae": float(baseline_mae),
                "variants": variants,
            }
        )

    variant_names = list(seed_results[0]["variants"])
    summary = {}
    for name in variant_names:
        gains = np.asarray(
            [row["variants"][name]["overall_equivalent_gain"] for row in seed_results]
        )
        scores = np.asarray(
            [row["variants"][name]["unlinked_mae"] for row in seed_results]
        )
        summary[name] = {
            "unlinked_mae_mean": float(scores.mean()),
            "overall_equivalent_gain_mean": float(gains.mean()),
            "seed_wins": int((gains > 0).sum()),
            "seed_ties": int((gains == 0).sum()),
            "seed_losses": int((gains < 0).sum()),
            "passes_screen": bool(
                gains.mean() >= MIN_SUBMISSION_GAIN and (gains > 0).all()
            ),
        }

    metrics = {
        "protocol": {
            "seeds": list(SEEDS),
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "gam_configs": GAM_CONFIGS,
            "gam_blend_weight": BLEND_WEIGHT_GAM,
            "minimum_submission_gain": MIN_SUBMISSION_GAIN,
            "test_used": False,
        },
        "summary": summary,
        "seed_results": seed_results,
    }
    output_path = ROOT / "outputs/experiment_v16_gam_quantile_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== GAM unlinked OOF 결과 =====")
    for name, result in summary.items():
        print(
            f"{name:20s} unlinked={result['unlinked_mae_mean']:.9f} "
            f"overall_gain={result['overall_equivalent_gain_mean']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"통과={result['passes_screen']}"
        )
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
