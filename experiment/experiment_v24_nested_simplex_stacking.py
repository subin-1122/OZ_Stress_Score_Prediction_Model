"""
기각된 모델까지 한 번에 합치는 nested simplex stacking
==========================================================

목적
----
개별 모델이 현재 ExtraTrees보다 나쁘더라도 오차 방향이 다르면 앙상블에서
분산이 줄어들 수 있다. 저장된 train OOF만 사용해 모든 가중치를 0 이상,
합계를 1로 제한하고 MAE가 최소가 되도록 선형계획법으로 학습한다.

과적합 방지
-----------
- test.csv와 test 예측은 전혀 읽지 않는다.
- 메타 가중치는 outer fold의 train 행에서만 학습하고 validation에 적용한다.
- 연결된 쉬운 행이 가중치를 지배하지 않도록 미연결 행에서만 가중치를 맞춘다.
- 기존 ExtraTrees 최소 가중치를 0%, 50%, 75%, 90%로 미리 고정해 비교한다.
- raw OOF를 합친 뒤 보정하는 방식과, 후보별 최종 레이어를 먼저 적용한 뒤
  합치는 방식을 모두 확인한다.
- 평균 0.002 이상, 연결 seed 5개 모두 개선하기 전에는 제출 파일을 만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v24_nested_simplex_stacking.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
META_SEED = 42
N_SPLITS = 5
MIN_GAIN = 0.002
MIN_BASE_WEIGHTS = (0.00, 0.50, 0.75, 0.90)


def load_module(name: str, path: Path):
    """기존 연결과 mean_working OOF 함수를 재사용한다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    """최종 예측을 train target의 0.01 격자에 맞춘다."""
    return np.clip(np.rint(values / 0.01) * 0.01, 0.0, 1.0)


def check_ids(frame: pd.DataFrame, expected: pd.Series, name: str) -> None:
    """CSV 기반 OOF가 원본 train과 같은 행 순서인지 검사한다."""
    if ID_COLUMN not in frame or not frame[ID_COLUMN].equals(expected):
        raise ValueError(f"{name}의 ID 또는 행 순서가 train과 다릅니다.")


def load_candidates(train: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """서로 다른 계열의 저장된 OOF를 한 행렬로 모은다."""
    ids = train[ID_COLUMN]
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    baseline = pd.read_csv(ROOT / "outputs/baseline_lightgbm_oof.csv")
    v1 = pd.read_csv(ROOT / "outputs/experiment_v1_oof.csv")
    v2 = pd.read_csv(ROOT / "outputs/experiment_v2_oof.csv")
    for name, frame in (
        ("best_record_linkage_oof", stored),
        ("baseline_lightgbm_oof", baseline),
        ("experiment_v1_oof", v1),
        ("experiment_v2_oof", v2),
    ):
        check_ids(frame, ids, name)

    v18 = np.load(ROOT / "outputs/experiment_v18_gam_full_predictions.npz")
    v21 = np.load(ROOT / "outputs/experiment_v21_distributional_median_oof.npz")
    candidates = {
        # 첫 열은 반드시 현재 기준 ExtraTrees다. 최소 가중치 제약의 대상이다.
        "base_extratrees": stored["base_oof"].to_numpy(float),
        "baseline_lightgbm": baseline["predicted_stress_score"].to_numpy(float),
        "alternative_extratrees": v1["pred_extra_trees"].to_numpy(float),
        "svr": v1["pred_svr"].to_numpy(float),
        "extratrees_v2_a": v2["oof_A_drop_work_age_sleep"].to_numpy(float),
        "extratrees_v2_b": v2["oof_B_drop_work_age_smoke_dia"].to_numpy(float),
        "quantile_gam": v18["gam_oof"],
        "ordinal_median": v21["ordinal_probability_median"],
        "forest_alternative": v21["forest_screen_trimmed_mean"],
        "forest_proximity": v21["forest_proximity_median"],
    }
    matrix = np.column_stack(list(candidates.values()))
    if matrix.shape != (len(train), len(candidates)) or not np.isfinite(matrix).all():
        raise ValueError("OOF 후보 행렬의 크기 또는 값이 올바르지 않습니다.")
    return list(candidates), matrix


def make_link_contexts(v4, train, y, stored) -> list[dict]:
    """연결 split seed 5개의 mask, label과 현재 최종 OOF를 만든다."""
    base = stored["base_oof"].to_numpy(float)
    learned_label = stored["link_label"].to_numpy(float)
    learned_mask = stored["linked"].astype(bool).to_numpy()
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(features, numeric, categorical)
    contexts = []

    for seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            pairs,
            y,
            len(train),
            int(seed),
        )
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]
        current = finalize(v4, train, y, base, union_mask, union_label)
        contexts.append(
            {
                "seed": int(seed),
                "mask": union_mask,
                "label": union_label,
                "current": current,
            }
        )
    return contexts


def finalize(v4, train, y, raw, linked, labels) -> np.ndarray:
    """raw OOF에 기존 mean_working 보정, snap, hard link를 적용한다."""
    correction = v4.crossfit_work_correction(train, y, raw, linked)
    prediction = snap(raw + v4.WORK_CORRECTION_ALPHA * correction)
    prediction[linked] = labels[linked]
    return prediction


def fit_simplex_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    minimum_base_weight: float,
) -> np.ndarray:
    """
    비음수이고 합계가 1인 가중치로 MAE를 정확히 최소화한다.

    각 행의 절대오차를 양수 slack 변수로 바꾸면 선형계획 문제가 된다.
    첫 번째 모델은 현재 ExtraTrees이므로 지정한 최소 가중치를 적용한다.
    """
    n_rows, n_models = prediction.shape
    objective = np.r_[np.zeros(n_models), np.ones(n_rows)]
    identity = np.eye(n_rows)
    upper_matrix = np.r_[
        np.c_[prediction, -identity],
        np.c_[-prediction, -identity],
    ]
    upper_target = np.r_[target, -target]
    equality_matrix = np.c_[
        np.ones((1, n_models)),
        np.zeros((1, n_rows)),
    ]
    bounds = (
        [(minimum_base_weight, None)]
        + [(0.0, None)] * (n_models - 1)
        + [(0.0, None)] * n_rows
    )
    result = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=upper_target,
        A_eq=equality_matrix,
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"simplex 최적화 실패: {result.message}")
    return result.x[:n_models]


def nested_stack(
    matrix: np.ndarray,
    y: np.ndarray,
    usable_mask: np.ndarray,
    minimum_base_weight: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """각 validation fold 밖에서만 가중치를 학습해 meta OOF를 만든다."""
    meta_oof = np.zeros(len(y), dtype=float)
    weights = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=META_SEED)
    for train_idx, valid_idx in splitter.split(y):
        usable = train_idx[usable_mask[train_idx]]
        fold_weights = fit_simplex_mae(
            matrix[usable],
            y[usable],
            minimum_base_weight,
        )
        meta_oof[valid_idx] = matrix[valid_idx] @ fold_weights
        weights.append(fold_weights)
    return meta_oof, weights


def summarize(gains: np.ndarray, scores: np.ndarray) -> dict:
    """seed별 개선 방향과 사전 통과 조건을 요약한다."""
    return {
        "oof_mae_mean": float(scores.mean()),
        "gain_mean_vs_current": float(gains.mean()),
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "passes_gate": bool(gains.mean() >= MIN_GAIN and np.all(gains > 0)),
    }


def main() -> None:
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    names, raw_matrix = load_candidates(train)
    contexts = make_link_contexts(v4, train, y, stored)
    common_unlinked = ~np.logical_or.reduce(
        [context["mask"] for context in contexts]
    )
    results = {"raw_then_finalize": {}, "final_prediction_stack": {}}

    # 방식 1: raw 후보를 합친 뒤 기존 final layer를 적용한다.
    for minimum in MIN_BASE_WEIGHTS:
        meta_raw, weights = nested_stack(
            raw_matrix,
            y,
            common_unlinked,
            minimum,
        )
        gains = []
        scores = []
        repeated = []
        for context in contexts:
            candidate = finalize(
                v4,
                train,
                y,
                meta_raw,
                context["mask"],
                context["label"],
            )
            current_mae = mean_absolute_error(y, context["current"])
            score = mean_absolute_error(y, candidate)
            gains.append(current_mae - score)
            scores.append(score)
            repeated.append(
                {
                    "link_seed": context["seed"],
                    "oof_mae": float(score),
                    "gain_vs_current": float(current_mae - score),
                }
            )
        key = f"minimum_base_{minimum:.2f}"
        results["raw_then_finalize"][key] = {
            **summarize(np.asarray(gains), np.asarray(scores)),
            "mean_fold_weights": dict(
                zip(names, np.mean(np.vstack(weights), axis=0).tolist())
            ),
            "repeated": repeated,
        }

    # 방식 2: 후보마다 final layer를 적용한 완성 예측을 직접 합친다.
    for minimum in MIN_BASE_WEIGHTS:
        gains = []
        scores = []
        repeated = []
        weight_rows = []
        for context in contexts:
            final_matrix = np.column_stack(
                [
                    finalize(
                        v4,
                        train,
                        y,
                        raw_matrix[:, column],
                        context["mask"],
                        context["label"],
                    )
                    for column in range(raw_matrix.shape[1])
                ]
            )
            meta_final, weights = nested_stack(
                final_matrix,
                y,
                ~context["mask"],
                minimum,
            )
            meta_final = snap(meta_final)
            meta_final[context["mask"]] = context["label"][context["mask"]]
            current_mae = mean_absolute_error(y, context["current"])
            score = mean_absolute_error(y, meta_final)
            gains.append(current_mae - score)
            scores.append(score)
            mean_weights = np.mean(np.vstack(weights), axis=0)
            weight_rows.append(mean_weights)
            repeated.append(
                {
                    "link_seed": context["seed"],
                    "oof_mae": float(score),
                    "gain_vs_current": float(current_mae - score),
                    "mean_fold_weights": dict(zip(names, mean_weights.tolist())),
                }
            )
        key = f"minimum_base_{minimum:.2f}"
        results["final_prediction_stack"][key] = {
            **summarize(np.asarray(gains), np.asarray(scores)),
            "mean_weights_across_link_seeds": dict(
                zip(names, np.mean(np.vstack(weight_rows), axis=0).tolist())
            ),
            "repeated": repeated,
        }

    flat = [
        (f"{mode}/{name}", detail)
        for mode, variants in results.items()
        for name, detail in variants.items()
    ]
    ranking = sorted(
        flat,
        key=lambda item: item[1]["gain_mean_vs_current"],
        reverse=True,
    )
    metrics = {
        "protocol": {
            "test_used": False,
            "meta_seed": META_SEED,
            "n_splits": N_SPLITS,
            "minimum_gain": MIN_GAIN,
            "requires_all_five_link_seed_wins": True,
            "candidate_names": names,
            "minimum_base_weights": MIN_BASE_WEIGHTS,
            "common_unlinked_rows": int(common_unlinked.sum()),
        },
        "results": results,
        "best_variant": ranking[0][0],
        "best_result": ranking[0][1],
    }
    output_path = ROOT / "outputs/experiment_v24_nested_simplex_stacking_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Nested simplex stacking =====")
    for name, result in ranking:
        print(
            f"{name:48s} OOF={result['oof_mae_mean']:.9f} "
            f"gain={result['gain_mean_vs_current']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"통과={result['passes_gate']}"
        )
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
