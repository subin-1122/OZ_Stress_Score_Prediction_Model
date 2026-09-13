"""
Duplicate confidence 기반 ExtraTrees sample_weight 실험
========================================================

가설
----
고신뢰 duplicate 행은 target 노이즈가 상대적으로 작고, 가까운 duplicate가 없는
행은 target 노이즈가 더 클 수 있다. 모든 학습 행을 같은 비중으로 다루는 대신
duplicate confidence가 높은 행의 가중치를 올리고 나머지는 낮춰 ExtraTrees를
학습하면 noisy unique 행에 덜 끌릴 수 있는지 확인한다.

기존 v14와의 차이
-----------------
v14는 duplicate 그룹이 여러 번 반복되어 모델을 지배하는 것을 막으려고 그룹
크기의 역수로 가중치를 낮췄다. v31은 반대 가설을 검증한다. 고신뢰 duplicate
행을 더 신뢰할 수 있는 관측으로 보고 가중치를 높인다.

검증 설계
---------
1. 5개 seed × 10-fold에서 baseline과 네 개의 사전 고정 가중치 설정을 비교한다.
2. duplicate confidence와 결측치/인코딩 규칙은 매 fold의 train 부분으로만 만든다.
3. 동일 fold·동일 random_state로 baseline과 가중 모델을 학습한다.
4. raw ExtraTrees뿐 아니라 기존 연결 + mean_working 보정 + 0.01 snapping까지
   적용한 최종 OOF에서도 비교한다.
5. test.csv는 읽지 않으며 제출 파일을 만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v31_confidence_sample_weight.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"

SEEDS = (11, 101, 1001, 2026, 31415)
N_SPLITS = 10
N_ESTIMATORS = 250
TRIM_RATIO = 0.10
TARGET_OOF_GAIN = 0.0022

# confidence=0인 행과 confidence=1인 행의 상대 가중치다. 실제 model.fit 전에
# fold별 평균이 1이 되도록 정규화하므로 전체 loss 규모는 동일하게 유지된다.
WEIGHT_VARIANTS = {
    "up_only_1p0_1p5": (1.00, 1.50),
    "gentle_0p8_1p5": (0.80, 1.50),
    "balanced_0p5_2p0": (0.50, 2.00),
    "strong_0p25_3p0": (0.25, 3.00),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair_equal_counts(features: pd.DataFrame, pairs: np.ndarray) -> np.ndarray:
    """두 행에서 값이 같은 원본 feature 수를 센다. target은 보지 않는다."""
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    counts = np.zeros(len(pairs), dtype=np.int16)
    for column in features.columns:
        a = left[column]
        b = right[column]
        counts += ((a == b) | (a.isna() & b.isna())).to_numpy(np.int16)
    return counts


def duplicate_confidence(
    train_indices: np.ndarray,
    pairs: np.ndarray,
    equal_counts: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    fold-train 내부에서만 행별 duplicate confidence를 계산한다.

    10개 이상 feature가 같고 target도 같은 쌍만 실제 duplicate 증거로 쓴다.
    equality 개수별 신뢰도는 Beta(2, 2) 사전분포를 더한 same-target 비율이다.
    한 행에 여러 증거가 있으면 가장 높은 신뢰도와 지지 쌍 수를 함께 반영한다.
    """
    in_train = np.zeros(len(y), dtype=bool)
    in_train[train_indices] = True
    eligible = in_train[pairs[:, 0]] & in_train[pairs[:, 1]]
    fold_pairs = pairs[eligible]
    fold_equal = equal_counts[eligible]
    same = y[fold_pairs[:, 0]] == y[fold_pairs[:, 1]]

    precision_by_count: dict[int, float] = {}
    for count in range(10, 17):
        group = fold_equal == count
        same_count = int(np.sum(group & same))
        total_count = int(group.sum())
        # 작은 bin에서 0%나 100%를 과신하지 않도록 완만하게 수축한다.
        precision_by_count[count] = (same_count + 2.0) / (total_count + 4.0)

    confidence_global = np.zeros(len(y), dtype=float)
    support_global = np.zeros(len(y), dtype=np.int16)
    duplicate_pair_count = 0
    for count in range(10, 17):
        selected = (fold_equal == count) & same
        selected_pairs = fold_pairs[selected]
        if not len(selected_pairs):
            continue
        duplicate_pair_count += len(selected_pairs)
        pair_confidence = precision_by_count[count]
        np.maximum.at(confidence_global, selected_pairs[:, 0], pair_confidence)
        np.maximum.at(confidence_global, selected_pairs[:, 1], pair_confidence)
        np.add.at(support_global, selected_pairs[:, 0], 1)
        np.add.at(support_global, selected_pairs[:, 1], 1)

    # 같은 강도의 증거가 여러 개일수록 조금 더 확실하다고 본다. 지지 수 효과는
    # 최대 25%로 제한해 큰 component 하나가 가중치를 독점하지 않게 한다.
    support_factor = 0.75 + 0.25 * (1.0 - np.exp(-support_global))
    confidence_global *= support_factor
    local = confidence_global[train_indices]
    diagnostics = {
        "duplicate_pairs": int(duplicate_pair_count),
        "rows_with_evidence": int(np.sum(local > 0)),
        "coverage": float(np.mean(local > 0)),
        "confidence_mean": float(local.mean()),
        "confidence_nonzero_mean": (
            float(local[local > 0].mean()) if np.any(local > 0) else 0.0
        ),
        "precision_by_equal_count": precision_by_count,
    }
    return local, diagnostics


def make_sample_weight(
    confidence: np.ndarray,
    low_weight: float,
    high_weight: float,
) -> np.ndarray:
    """confidence를 두 끝점 사이로 선형 변환하고 평균 1로 정규화한다."""
    weight = low_weight + (high_weight - low_weight) * confidence
    return weight / weight.mean()


def fit_predict(
    v3,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    random_state: int,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    """같은 전처리와 ExtraTrees 설정으로 validation 예측을 만든다."""
    preprocessor = v3.build_preprocessor(x_train)
    transformed_train = preprocessor.fit_transform(x_train)
    transformed_valid = preprocessor.transform(x_valid)
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        criterion="squared_error",
        max_features=1,
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(
        transformed_train,
        y_train,
        sample_weight=sample_weight,
    )
    return v3.trimmed_tree_prediction(model, transformed_valid, TRIM_RATIO)


def run_screen(v3, train, y, model_features, pairs, equal_counts):
    """5개 seed에서 baseline과 네 가중치 설정의 paired OOF를 만든다."""
    all_predictions: dict[str, list[np.ndarray]] = {
        "baseline": [],
        **{name: [] for name in WEIGHT_VARIANTS},
    }
    seed_rows = []

    for seed in SEEDS:
        predictions = {
            name: np.zeros(len(train), dtype=float)
            for name in all_predictions
        }
        fold_diagnostics = []
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        started = time.perf_counter()

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), start=1):
            confidence, confidence_diag = duplicate_confidence(
                train_idx,
                pairs,
                equal_counts,
                y,
            )
            x_train = model_features.iloc[train_idx]
            x_valid = model_features.iloc[valid_idx]
            random_state = seed * 1000 + fold

            predictions["baseline"][valid_idx] = fit_predict(
                v3,
                x_train,
                y[train_idx],
                x_valid,
                random_state,
                sample_weight=None,
            )
            weight_stats = {}
            for name, (low_weight, high_weight) in WEIGHT_VARIANTS.items():
                sample_weight = make_sample_weight(
                    confidence,
                    low_weight,
                    high_weight,
                )
                predictions[name][valid_idx] = fit_predict(
                    v3,
                    x_train,
                    y[train_idx],
                    x_valid,
                    random_state,
                    sample_weight=sample_weight,
                )
                weight_stats[name] = {
                    "minimum": float(sample_weight.min()),
                    "maximum": float(sample_weight.max()),
                    "mean": float(sample_weight.mean()),
                }

            fold_diagnostics.append(
                {
                    "fold": fold,
                    **confidence_diag,
                    "weight_stats": weight_stats,
                }
            )
            print(
                f"[weight] seed={seed} fold={fold:02d}/{N_SPLITS} "
                f"duplicate_rows={confidence_diag['rows_with_evidence']} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

        scores = {
            name: float(mean_absolute_error(y, values))
            for name, values in predictions.items()
        }
        baseline_score = scores["baseline"]
        seed_rows.append(
            {
                "seed": int(seed),
                "scores": scores,
                "gains_vs_baseline": {
                    name: float(baseline_score - scores[name])
                    for name in WEIGHT_VARIANTS
                },
                "fold_diagnostics": fold_diagnostics,
            }
        )
        for name, values in predictions.items():
            all_predictions[name].append(values)

    return all_predictions, seed_rows


def raw_summary(all_predictions, seed_rows):
    """raw ExtraTrees 단계의 seed별 승/무/패와 평균 개선을 요약한다."""
    baseline_scores = np.asarray(
        [row["scores"]["baseline"] for row in seed_rows]
    )
    summary = {}
    for name in WEIGHT_VARIANTS:
        scores = np.asarray([row["scores"][name] for row in seed_rows])
        gains = baseline_scores - scores
        summary[name] = {
            "oof_mae_mean": float(scores.mean()),
            "gain_mean": float(gains.mean()),
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
        }
    return float(baseline_scores.mean()), summary


def integrated_summary(v4, v24, train, y, contexts, all_predictions):
    """각 model seed 예측에 기존 최종 레이어를 붙여 paired 비교한다."""
    detail = []
    for model_seed_index, model_seed in enumerate(SEEDS):
        baseline_raw = all_predictions["baseline"][model_seed_index]
        for context in contexts:
            baseline_final = v24.finalize(
                v4,
                train,
                y,
                baseline_raw,
                context["mask"],
                context["label"],
            )
            baseline_mae = mean_absolute_error(y, baseline_final)
            for name in WEIGHT_VARIANTS:
                candidate_final = v24.finalize(
                    v4,
                    train,
                    y,
                    all_predictions[name][model_seed_index],
                    context["mask"],
                    context["label"],
                )
                candidate_mae = mean_absolute_error(y, candidate_final)
                detail.append(
                    {
                        "model_seed": int(model_seed),
                        "link_seed": int(context["seed"]),
                        "variant": name,
                        "baseline_oof_mae": float(baseline_mae),
                        "candidate_oof_mae": float(candidate_mae),
                        "gain_vs_baseline": float(baseline_mae - candidate_mae),
                    }
                )

    # 다섯 model seed 예측을 먼저 평균한 안정적인 ensemble도 별도로 확인한다.
    ensemble_detail = []
    baseline_ensemble = np.mean(
        np.column_stack(all_predictions["baseline"]),
        axis=1,
    )
    candidate_ensembles = {
        name: np.mean(np.column_stack(all_predictions[name]), axis=1)
        for name in WEIGHT_VARIANTS
    }
    for context in contexts:
        baseline_final = v24.finalize(
            v4,
            train,
            y,
            baseline_ensemble,
            context["mask"],
            context["label"],
        )
        baseline_mae = mean_absolute_error(y, baseline_final)
        for name, raw in candidate_ensembles.items():
            candidate_final = v24.finalize(
                v4,
                train,
                y,
                raw,
                context["mask"],
                context["label"],
            )
            candidate_mae = mean_absolute_error(y, candidate_final)
            ensemble_detail.append(
                {
                    "link_seed": int(context["seed"]),
                    "variant": name,
                    "baseline_oof_mae": float(baseline_mae),
                    "candidate_oof_mae": float(candidate_mae),
                    "gain_vs_baseline": float(baseline_mae - candidate_mae),
                }
            )

    summary = {}
    for name in WEIGHT_VARIANTS:
        rows = [row for row in detail if row["variant"] == name]
        ensemble_rows = [
            row for row in ensemble_detail if row["variant"] == name
        ]
        gains = np.asarray([row["gain_vs_baseline"] for row in rows])
        ensemble_gains = np.asarray(
            [row["gain_vs_baseline"] for row in ensemble_rows]
        )
        ensemble_scores = np.asarray(
            [row["candidate_oof_mae"] for row in ensemble_rows]
        )
        summary[name] = {
            "all_25_gain_mean": float(gains.mean()),
            "all_25_gain_min": float(gains.min()),
            "all_25_gain_max": float(gains.max()),
            "all_25_wins": int((gains > 0).sum()),
            "all_25_ties": int((gains == 0).sum()),
            "all_25_losses": int((gains < 0).sum()),
            "ensemble_oof_mae_mean": float(ensemble_scores.mean()),
            "ensemble_gain_mean": float(ensemble_gains.mean()),
            "ensemble_gain_min": float(ensemble_gains.min()),
            "ensemble_gain_max": float(ensemble_gains.max()),
            "ensemble_wins": int((ensemble_gains > 0).sum()),
            "ensemble_ties": int((ensemble_gains == 0).sum()),
            "ensemble_losses": int((ensemble_gains < 0).sum()),
            "reaches_target": bool(
                ensemble_gains.mean() >= TARGET_OOF_GAIN
                and np.all(ensemble_gains > 0)
            ),
        }
    return summary, detail, ensemble_detail


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


def main() -> None:
    v3 = load_module(
        "experiment_v3_for_v31",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4_for_v31",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = load_module(
        "experiment_v24_for_v31",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    # 규칙 준수를 명확히 하기 위해 train과 기존 train OOF만 읽는다.
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
    equal_counts = pair_equal_counts(raw_features, pairs)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    started = time.perf_counter()
    all_predictions, seed_rows = run_screen(
        v3,
        train,
        y,
        model_features,
        pairs,
        equal_counts,
    )
    raw_baseline, raw_variants = raw_summary(all_predictions, seed_rows)
    integrated, integrated_detail, ensemble_detail = integrated_summary(
        v4,
        v24,
        train,
        y,
        contexts,
        all_predictions,
    )

    metrics = {
        "protocol": {
            "test_used": False,
            "seeds": SEEDS,
            "n_splits": N_SPLITS,
            "n_estimators": N_ESTIMATORS,
            "weight_variants": WEIGHT_VARIANTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "candidate_pair_count": int(len(pairs)),
        "raw_baseline_oof_mae_mean": raw_baseline,
        "raw_variants": raw_variants,
        "integrated_variants": integrated,
        "seed_results": seed_rows,
        "integrated_25_details": integrated_detail,
        "integrated_ensemble_details": ensemble_detail,
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v31_confidence_sample_weight_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v31_confidence_sample_weight_oof.npz",
        **{
            f"{name}_seed_{seed}": prediction
            for name, predictions in all_predictions.items()
            for seed, prediction in zip(SEEDS, predictions)
        },
    )

    print("\n===== Confidence sample-weight 결과 =====")
    print(f"raw baseline 평균={raw_baseline:.9f}")
    for name in WEIGHT_VARIANTS:
        raw = raw_variants[name]
        final = integrated[name]
        print(
            f"{name:22s} raw_gain={raw['gain_mean']:+.9f} "
            f"raw 승/무/패={raw['wins']}/{raw['ties']}/{raw['losses']} | "
            f"final OOF={final['ensemble_oof_mae_mean']:.9f} "
            f"final_gain={final['ensemble_gain_mean']:+.9f} "
            f"final 승/무/패={final['ensemble_wins']}/"
            f"{final['ensemble_ties']}/{final['ensemble_losses']} "
            f"목표도달={final['reaches_target']}"
        )
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
