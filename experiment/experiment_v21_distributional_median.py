"""
Ordinal 분포 모델과 ExtraTrees proximity 중앙값의 1차 OOF 비교
=================================================================

목표
----
현재 파이프라인은 ExtraTrees의 여러 트리 예측을 절사평균한다. 하지만 DACON
평가지표인 MAE에서 이상적인 점 예측은 조건부 평균이 아니라 조건부 중앙값이다.
이 파일은 같은 정보를 중앙값으로 예측하는 질적으로 다른 두 방법을 비교한다.

1. LightGBM multiclass가 stress_score의 101개 격자 확률을 예측하고, 누적확률
   50% 지점을 조건부 중앙값으로 사용한다.
2. ExtraTrees에서 validation 행과 같은 leaf에 도달한 train target들을 모아
   forest proximity 가중 중앙값을 계산한다.

누수 방지
---------
- 이 단계에서는 test.csv를 읽지 않는다.
- 결측치 처리와 원-핫 인코딩은 각 fold의 train 부분으로만 fit한다.
- validation 행의 target은 모델, leaf 중앙값, 잔차 보정 계산에 들어가지 않는다.
- 기존 연결 및 mean_working 보정과 같은 OOF 규칙으로 최종 조합을 평가한다.
- 여러 후보 중 평균 OOF 0.002 이상, 연결 seed 5개 모두 개선한 방법만 다음
  독립 검증으로 넘긴다. 이 파일은 제출 CSV를 만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v21_distributional_median.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
GRID_STEP = 0.01
WORK_ALPHA = 0.75
SCREEN_SEED = 73
N_SPLITS = 5
FOREST_TREES = 400
MIN_GAIN = 0.002
BLEND_WEIGHTS = (0.10, 0.20, 0.40, 0.60, 1.00)


def load_module(name: str, path: Path):
    """다른 실험 파일의 검증된 전처리·연결 함수를 안전하게 재사용한다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    """train target에서 확인한 0.01 격자로 이동하고 0~1 범위를 지킨다."""
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def probability_summaries(
    probability: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    클래스 확률로 조건부 중앙값과 조건부 평균을 각각 계산한다.

    fold에 모든 클래스가 존재하더라도 열 순서를 가정하지 않고 classes_ 값을
    이용해 0~100 전체 격자 확률 행렬을 만든다.
    """
    full = np.zeros((len(probability), 101), dtype=np.float64)
    full[:, classes.astype(int)] = probability
    cumulative = np.cumsum(full, axis=1)
    median = np.argmax(cumulative >= 0.5, axis=1) / 100.0
    mean = full @ (np.arange(101, dtype=float) / 100.0)
    return median, mean


def make_ordinal_oof(v3, features: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """5-fold LightGBM multiclass OOF에서 확률 중앙값과 평균을 만든다."""
    labels = np.rint(y * 100).astype(int)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SCREEN_SEED,
    )
    median_oof = np.zeros(len(y), dtype=float)
    mean_oof = np.zeros(len(y), dtype=float)
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        preprocessor = v3.build_preprocessor(features.iloc[train_idx])
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=101,
            n_estimators=80,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            subsample=0.90,
            colsample_bytree=0.80,
            reg_lambda=5.0,
            random_state=SCREEN_SEED * 100 + fold,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, labels[train_idx])
        probability = model.predict_proba(x_valid)
        median, mean = probability_summaries(probability, model.classes_)
        median_oof[valid_idx] = median
        mean_oof[valid_idx] = mean
        print(
            f"[ordinal] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    return {
        "ordinal_probability_median": median_oof,
        "ordinal_probability_mean": mean_oof,
    }


def trimmed_mean(tree_predictions: np.ndarray, ratio: float = 0.10) -> np.ndarray:
    """기존 방식과 같은 위·아래 10% 절사평균을 계산한다."""
    ordered = np.sort(tree_predictions, axis=1)
    cut = int(ordered.shape[1] * ratio)
    return ordered[:, cut:-cut].mean(axis=1)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """각 행의 가중치 누적합이 50%를 처음 넘는 target을 반환한다."""
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[:, order]
    cumulative = np.cumsum(ordered_weights, axis=1)
    threshold = cumulative[:, -1:] * 0.5
    positions = np.argmax(cumulative >= threshold, axis=1)
    return ordered_values[positions]


def proximity_median(
    model: ExtraTreesRegressor,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
) -> np.ndarray:
    """
    같은 leaf에 들어간 train target의 forest-proximity 가중 중앙값을 구한다.

    한 트리가 가진 총 가중치는 항상 1이다. leaf 안에 train 행이 여러 개면
    그 1을 행 수로 나눈다. 여러 트리에서 같은 train 행을 반복해서 만날수록
    validation 행과 가까운 이웃으로 더 큰 가중치를 받는다.
    """
    weights = np.zeros((len(x_valid), len(x_train)), dtype=np.float32)

    for tree in model.estimators_:
        train_leaf = tree.apply(x_train)
        valid_leaf = tree.apply(x_valid)
        members: dict[int, list[int]] = {}
        for row, leaf in enumerate(train_leaf):
            members.setdefault(int(leaf), []).append(row)

        valid_groups: dict[int, list[int]] = {}
        for row, leaf in enumerate(valid_leaf):
            valid_groups.setdefault(int(leaf), []).append(row)

        for leaf, valid_rows in valid_groups.items():
            train_rows = members[leaf]
            contribution = 1.0 / len(train_rows)
            weights[np.ix_(valid_rows, train_rows)] += contribution

    return weighted_median(y_train, weights)


def make_forest_oof(v3, features: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    """같은 ExtraTrees에서 절사평균, 트리 중앙값, proximity 중앙값을 비교한다."""
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SCREEN_SEED)
    outputs = {
        "forest_screen_trimmed_mean": np.zeros(len(y), dtype=float),
        "forest_tree_prediction_median": np.zeros(len(y), dtype=float),
        "forest_proximity_median": np.zeros(len(y), dtype=float),
    }
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(features), start=1):
        preprocessor = v3.build_preprocessor(features.iloc[train_idx])
        x_train = preprocessor.fit_transform(features.iloc[train_idx])
        x_valid = preprocessor.transform(features.iloc[valid_idx])
        model = ExtraTreesRegressor(
            n_estimators=FOREST_TREES,
            criterion="squared_error",
            max_features=1,
            min_samples_leaf=1,
            bootstrap=False,
            random_state=SCREEN_SEED * 100 + fold,
            n_jobs=-1,
        )
        model.fit(x_train, y[train_idx])

        tree_predictions = np.column_stack(
            [tree.predict(x_valid) for tree in model.estimators_]
        )
        outputs["forest_screen_trimmed_mean"][valid_idx] = trimmed_mean(
            tree_predictions
        )
        outputs["forest_tree_prediction_median"][valid_idx] = np.median(
            tree_predictions,
            axis=1,
        )
        outputs["forest_proximity_median"][valid_idx] = proximity_median(
            model,
            x_train,
            y[train_idx],
            x_valid,
        )
        print(
            f"[forest] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    return outputs


def final_prediction(
    v4,
    train: pd.DataFrame,
    y: np.ndarray,
    raw_oof: np.ndarray,
    union_mask: np.ndarray,
    union_label: np.ndarray,
) -> np.ndarray:
    """기존과 동일하게 mean_working 보정, snapping, hard link를 적용한다."""
    correction = v4.crossfit_work_correction(
        train,
        y,
        raw_oof,
        union_mask,
    )
    prediction = snap(raw_oof + WORK_ALPHA * correction)
    prediction[union_mask] = union_label[union_mask]
    return prediction


def evaluate_candidates(
    v4,
    train: pd.DataFrame,
    y: np.ndarray,
    base_oof: np.ndarray,
    learned_label: np.ndarray,
    learned_mask: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[dict, list[dict]]:
    """각 후보를 기존 raw OOF와 섞고 동일한 최종 레이어 아래에서 비교한다."""
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(features, numeric, categorical)
    repeated: list[dict] = []

    for link_seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            pairs,
            y,
            len(train),
            int(link_seed),
        )
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_mask | deterministic_mask
        union_label = learned_label.copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]

        current = final_prediction(
            v4,
            train,
            y,
            base_oof,
            union_mask,
            union_label,
        )
        current_mae = mean_absolute_error(y, current)
        scores: dict[str, dict] = {}

        for name, alternative in candidates.items():
            for weight in BLEND_WEIGHTS:
                raw = (1.0 - weight) * base_oof + weight * alternative
                prediction = final_prediction(
                    v4,
                    train,
                    y,
                    raw,
                    union_mask,
                    union_label,
                )
                score = mean_absolute_error(y, prediction)
                key = f"{name}__weight_{weight:.2f}"
                scores[key] = {
                    "oof_mae": float(score),
                    "gain_vs_current": float(current_mae - score),
                }

        repeated.append(
            {
                "link_seed": int(link_seed),
                "current_oof_mae": float(current_mae),
                "unlinked_rows": int((~union_mask).sum()),
                "scores": scores,
            }
        )

    summary = {}
    keys = repeated[0]["scores"].keys()
    for key in keys:
        gains = np.asarray(
            [row["scores"][key]["gain_vs_current"] for row in repeated]
        )
        scores = np.asarray([row["scores"][key]["oof_mae"] for row in repeated])
        summary[key] = {
            "oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": float(gains.mean()),
            "seed_wins": int((gains > 0).sum()),
            "seed_ties": int((gains == 0).sum()),
            "seed_losses": int((gains < 0).sum()),
            "passes_screen": bool(
                gains.mean() >= MIN_GAIN and np.all(gains > 0)
            ),
        }
    return summary, repeated


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
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    base_oof = stored["base_oof"].to_numpy(float)
    learned_label = stored["link_label"].to_numpy(float)
    learned_mask = stored["linked"].astype(bool).to_numpy()
    features = v3.make_model_features(train.drop(columns=[TARGET]))

    print("===== 1. Ordinal probability 모델 =====", flush=True)
    ordinal = make_ordinal_oof(v3, features, y)
    print("===== 2. Forest proximity 중앙값 =====", flush=True)
    forest = make_forest_oof(v3, features, y)
    candidates = {**ordinal, **forest}

    summary, repeated = evaluate_candidates(
        v4,
        train,
        y,
        base_oof,
        learned_label,
        learned_mask,
        candidates,
    )
    ranking = sorted(
        summary.items(),
        key=lambda item: item[1]["gain_mean_vs_current"],
        reverse=True,
    )

    raw_metrics = {
        name: {
            "mae_all_rows": float(mean_absolute_error(y, prediction)),
            "mae_learned_unlinked_rows": float(
                mean_absolute_error(y[~learned_mask], prediction[~learned_mask])
            ),
        }
        for name, prediction in candidates.items()
    }
    metrics = {
        "protocol": {
            "test_used": False,
            "screen_seed": SCREEN_SEED,
            "n_splits": N_SPLITS,
            "forest_trees": FOREST_TREES,
            "work_alpha": WORK_ALPHA,
            "minimum_gain": MIN_GAIN,
            "requires_all_five_link_seed_wins": True,
            "blend_weights": BLEND_WEIGHTS,
        },
        "raw_candidate_metrics": raw_metrics,
        "summary": summary,
        "repeated": repeated,
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "experiment_v21_distributional_median_metrics.json"
    cache_path = output_dir / "experiment_v21_distributional_median_oof.npz"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(cache_path, **candidates)

    print("\n===== 최종 1차 선별 순위 =====")
    for name, result in ranking[:15]:
        print(
            f"{name:48s} OOF={result['oof_mae_mean']:.9f} "
            f"gain={result['gain_mean_vs_current']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"통과={result['passes_screen']}"
        )
    passed = [name for name, result in ranking if result["passes_screen"]]
    print(f"독립 검증 대상={passed}")
    print(f"결과 요약={metrics_path}")
    print(f"OOF 캐시={cache_path}")


if __name__ == "__main__":
    main()
