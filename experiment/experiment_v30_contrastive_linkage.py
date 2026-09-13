"""
Train-only supervised contrastive embedding을 이용한 새 연결 후보 실험
====================================================================

목적
----
현재 최종 조합(고신뢰 연결 + mean_working 보정 + 0.01 snapping)은 그대로
유지한다. 기존 규칙으로 연결되지 않은 행 가운데, 대조학습 임베딩 공간에서
train 행과 매우 가까운 경우만 새 연결 후보로 추가했을 때 OOF MAE가 낮아지는지
검증한다.

대조학습 쌍
-----------
- 양성: 16개 원본 feature 중 11개 이상이 같고 stress_score도 같은 train 쌍
- 어려운 음성: 8~10개 feature가 같지만 stress_score는 다른 train 쌍

양성 기준 11개는 전체 train 진단에서 688/688쌍의 점수가 같았던 보수적
기준이다. 단, 실제 fold 학습에서는 반드시 fold-train 안에 든 쌍만 사용한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 전처리기, 양성/음성 쌍, 신경망은 outer-train으로만 학습한다.
- 연결 confidence 기준은 outer-train 안의 별도 calibration 행에서 정한다.
- outer-validation의 target은 최종 평가에만 사용한다.
- no-op을 후보에 포함해 calibration에서 이득이 없으면 아무 행도 바꾸지 않는다.
- 목표 개선 0.0022와 모든 반복 개선을 동시에 만족하지 않으면 제출 파일을
  만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v30_contrastive_linkage.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"

# 서로 다른 초기값에서도 같은 결론인지 확인한다.
EMBEDDING_SEEDS = (73, 307, 911)
OUTER_SPLITS = 5
CALIBRATION_RATIO = 0.20
EPOCHS = 120
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 16
NEGATIVES_PER_POSITIVE = 4
TOP_K = 5

# Public 0.12368 목표를 현재 Public-OOF gap으로 환산한 필요 개선량이다.
TARGET_OOF_GAIN = 0.0022


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    """numpy, Python, PyTorch 난수 초기값을 한 번에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_embedding_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """현재 학습 fold만으로 결측치, 표준화, 원-핫 규칙을 학습한다."""
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
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
        remainder="drop",
    )


def pair_equal_counts(features: pd.DataFrame, pairs: np.ndarray) -> np.ndarray:
    """두 행에서 같은 원본 feature의 수를 센다. target은 사용하지 않는다."""
    left = features.iloc[pairs[:, 0]].reset_index(drop=True)
    right = features.iloc[pairs[:, 1]].reset_index(drop=True)
    counts = np.zeros(len(pairs), dtype=np.int16)
    for column in features.columns:
        a = left[column]
        b = right[column]
        counts += ((a == b) | (a.isna() & b.isna())).to_numpy(np.int16)
    return counts


class ContrastiveEncoder(nn.Module):
    """표 형태 feature를 16차원 단위 벡터로 바꾸는 작은 MLP다."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, EMBEDDING_DIM),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.network(values)
        return torch.nn.functional.normalize(encoded, p=2, dim=1)


def select_training_pairs(
    pairs: np.ndarray,
    equal_counts: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """fit fold 내부의 보수적 양성과 어려운 음성 쌍만 반환한다."""
    in_fit = np.zeros(len(y), dtype=bool)
    in_fit[fit_indices] = True
    eligible = in_fit[pairs[:, 0]] & in_fit[pairs[:, 1]]
    same_target = y[pairs[:, 0]] == y[pairs[:, 1]]

    positive_mask = eligible & same_target & (equal_counts >= 11)
    negative_mask = (
        eligible
        & (~same_target)
        & (equal_counts >= 8)
        & (equal_counts <= 10)
    )
    positives = pairs[positive_mask]
    negatives = pairs[negative_mask]

    rng = np.random.default_rng(seed)
    maximum_negatives = NEGATIVES_PER_POSITIVE * len(positives)
    if len(negatives) > maximum_negatives:
        chosen = rng.choice(len(negatives), maximum_negatives, replace=False)
        negatives = negatives[chosen]

    if len(positives) < 50 or len(negatives) < 50:
        raise ValueError(
            f"대조학습 쌍이 부족합니다: positive={len(positives)}, "
            f"negative={len(negatives)}"
        )

    selected = np.vstack([positives, negatives]).astype(np.int32)
    labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))].astype(
        np.float32
    )
    order = rng.permutation(len(selected))
    diagnostics = {
        "positive_pairs": int(len(positives)),
        "negative_pairs": int(len(negatives)),
    }
    return selected[order], labels[order], diagnostics


def fit_encoder(
    features: pd.DataFrame,
    y: np.ndarray,
    pairs: np.ndarray,
    equal_counts: np.ndarray,
    fit_indices: np.ndarray,
    seed: int,
) -> tuple[ColumnTransformer, ContrastiveEncoder, dict]:
    """한 fold의 전처리기와 대조학습 encoder를 fit fold로만 학습한다."""
    set_seed(seed)
    preprocessor = build_embedding_preprocessor(features.iloc[fit_indices])
    transformed = np.asarray(
        preprocessor.fit_transform(features.iloc[fit_indices]),
        dtype=np.float32,
    )
    position = np.full(len(features), -1, dtype=np.int32)
    position[fit_indices] = np.arange(len(fit_indices), dtype=np.int32)

    selected, pair_labels, diagnostics = select_training_pairs(
        pairs,
        equal_counts,
        y,
        fit_indices,
        seed,
    )
    local_left = position[selected[:, 0]]
    local_right = position[selected[:, 1]]
    x_tensor = torch.from_numpy(transformed)
    left_tensor = torch.from_numpy(local_left.astype(np.int64))
    right_tensor = torch.from_numpy(local_right.astype(np.int64))
    label_tensor = torch.from_numpy(pair_labels)

    encoder = ContrastiveEncoder(transformed.shape[1])
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    final_loss = np.nan
    for _ in range(EPOCHS):
        encoder.train()
        embedding = encoder(x_tensor)
        difference = embedding[left_tensor] - embedding[right_tensor]
        distance = torch.sqrt(torch.sum(difference * difference, dim=1) + 1e-8)
        positive_loss = torch.square(distance[label_tensor == 1]).mean()
        negative_loss = torch.square(
            torch.relu(1.0 - distance[label_tensor == 0])
        ).mean()
        loss = positive_loss + negative_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    diagnostics.update(
        {
            "input_dim": int(transformed.shape[1]),
            "final_loss": final_loss,
        }
    )
    return preprocessor, encoder, diagnostics


def encode(
    preprocessor: ColumnTransformer,
    encoder: ContrastiveEncoder,
    frame: pd.DataFrame,
) -> np.ndarray:
    transformed = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    encoder.eval()
    with torch.no_grad():
        return encoder(torch.from_numpy(transformed)).numpy()


def nearest_neighbor_statistics(
    reference_embedding: np.ndarray,
    reference_y: np.ndarray,
    query_embedding: np.ndarray,
) -> dict[str, np.ndarray]:
    """가까운 train 후보, 거리, 1·2위 간격과 top-k 점수 지지도를 만든다."""
    similarity = query_embedding @ reference_embedding.T
    k = min(TOP_K, len(reference_embedding))
    nearest = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    nearest_similarity = np.take_along_axis(similarity, nearest, axis=1)
    order = np.argsort(-nearest_similarity, axis=1)
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_similarity = np.take_along_axis(nearest_similarity, order, axis=1)

    distances = np.sqrt(np.maximum(2.0 - 2.0 * nearest_similarity, 0.0))
    top_labels = reference_y[nearest]
    best_label = top_labels[:, 0]
    support = np.sum(top_labels == best_label[:, None], axis=1)
    margin = distances[:, 1] - distances[:, 0] if k > 1 else np.ones(len(query_embedding))
    return {
        "label": best_label.astype(float),
        "distance": distances[:, 0],
        "margin": margin,
        "support": support.astype(np.int16),
    }


def reference_distance_scale(embedding: np.ndarray) -> float:
    """학습행끼리의 leave-one-out 최근접 거리 중앙값을 scale로 쓴다."""
    similarity = embedding @ embedding.T
    np.fill_diagonal(similarity, -np.inf)
    best = np.max(similarity, axis=1)
    distance = np.sqrt(np.maximum(2.0 - 2.0 * best, 0.0))
    positive = distance[np.isfinite(distance) & (distance > 1e-6)]
    if not len(positive):
        return 1.0
    return max(float(np.median(positive)), 1e-3)


def retrieve(
    features: pd.DataFrame,
    y: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    preprocessor: ColumnTransformer,
    encoder: ContrastiveEncoder,
) -> dict[str, np.ndarray]:
    reference_embedding = encode(preprocessor, encoder, features.iloc[fit_indices])
    query_embedding = encode(preprocessor, encoder, features.iloc[query_indices])
    stats = nearest_neighbor_statistics(
        reference_embedding,
        y[fit_indices],
        query_embedding,
    )
    scale = reference_distance_scale(reference_embedding)
    stats["scaled_distance"] = stats["distance"] / scale
    stats["scaled_margin"] = stats["margin"] / scale
    stats["confidence"] = (
        -stats["scaled_distance"]
        + 0.35 * stats["scaled_margin"]
        + 0.12 * (stats["support"] - 1)
    )
    stats["reference_scale"] = np.full(len(query_indices), scale)
    return stats


def calibration_split(outer_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """outer-validation과 무관한 고정 난수로 inner-fit/calibration을 나눈다."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(outer_train)
    count = max(1, int(round(len(shuffled) * CALIBRATION_RATIO)))
    calibration = np.sort(shuffled[:count])
    inner_fit = np.sort(shuffled[count:])
    return inner_fit, calibration


def make_rules(calibration_confidence: np.ndarray) -> list[dict]:
    """calibration confidence 상위 1~20% 지점을 절대 threshold로 바꾼다."""
    quantiles = (0.80, 0.90, 0.95, 0.975, 0.99)
    thresholds = sorted(
        {float(np.quantile(calibration_confidence, q)) for q in quantiles},
        reverse=True,
    )
    rules = []
    for threshold in thresholds:
        for support in (1, 2, 3):
            rules.append(
                {
                    "threshold": threshold,
                    "minimum_support": support,
                }
            )
    return rules


def rule_mask(stats: dict[str, np.ndarray], rule: dict, already_linked: np.ndarray) -> np.ndarray:
    return (
        (~already_linked)
        & (stats["confidence"] >= rule["threshold"])
        & (stats["support"] >= rule["minimum_support"])
    )


def choose_rule(
    y: np.ndarray,
    current: np.ndarray,
    already_linked: np.ndarray,
    calibration_indices: np.ndarray,
    stats: dict[str, np.ndarray],
) -> dict:
    """calibration에서 MAE가 가장 낮은 규칙을 고르며 동률이면 no-op이다."""
    baseline = mean_absolute_error(y[calibration_indices], current[calibration_indices])
    ranking = [
        {
            "name": "no_op",
            "calibration_mae": float(baseline),
            "calibration_gain": 0.0,
            "threshold": None,
            "minimum_support": None,
            "changed_rows": 0,
            "selected_accuracy": None,
        }
    ]
    local_linked = already_linked[calibration_indices]
    for order, rule in enumerate(make_rules(stats["confidence"])):
        use = rule_mask(stats, rule, local_linked)
        candidate = current[calibration_indices].copy()
        candidate[use] = stats["label"][use]
        score = mean_absolute_error(y[calibration_indices], candidate)
        accuracy = (
            float(np.mean(stats["label"][use] == y[calibration_indices][use]))
            if use.any()
            else None
        )
        ranking.append(
            {
                "name": f"q_rule_{order:02d}",
                "calibration_mae": float(score),
                "calibration_gain": float(baseline - score),
                "threshold": float(rule["threshold"]),
                "minimum_support": int(rule["minimum_support"]),
                "changed_rows": int(use.sum()),
                "selected_accuracy": accuracy,
            }
        )
    return min(
        ranking,
        key=lambda row: (
            row["calibration_mae"],
            0 if row["name"] == "no_op" else 1,
            -(row["minimum_support"] or 0),
        ),
    )


def apply_choice(
    current: np.ndarray,
    already_linked: np.ndarray,
    valid_indices: np.ndarray,
    stats: dict[str, np.ndarray],
    choice: dict,
) -> tuple[np.ndarray, int]:
    result = current[valid_indices].copy()
    if choice["name"] == "no_op":
        return result, 0
    rule = {
        "threshold": choice["threshold"],
        "minimum_support": choice["minimum_support"],
    }
    use = rule_mask(stats, rule, already_linked[valid_indices])
    result[use] = stats["label"][use]
    return result, int(use.sum())


def run_seed(
    features: pd.DataFrame,
    y: np.ndarray,
    pairs: np.ndarray,
    equal_counts: np.ndarray,
    contexts: list[dict],
    embedding_seed: int,
) -> tuple[list[dict], list[dict]]:
    """한 neural seed에서 5-fold nested 후보 생성과 5개 연결 context 평가."""
    candidates = [context["current"].copy() for context in contexts]
    fold_choices: list[list[dict]] = [[] for _ in contexts]
    training_diagnostics = []
    splitter = KFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=embedding_seed,
    )

    for fold, (outer_train, outer_valid) in enumerate(splitter.split(features), start=1):
        inner_fit, calibration = calibration_split(
            outer_train,
            embedding_seed * 1000 + fold,
        )

        inner_preprocessor, inner_encoder, inner_diag = fit_encoder(
            features,
            y,
            pairs,
            equal_counts,
            inner_fit,
            embedding_seed * 100 + fold,
        )
        calibration_stats = retrieve(
            features,
            y,
            inner_fit,
            calibration,
            inner_preprocessor,
            inner_encoder,
        )

        full_preprocessor, full_encoder, full_diag = fit_encoder(
            features,
            y,
            pairs,
            equal_counts,
            outer_train,
            embedding_seed * 100 + fold + 10_000,
        )
        validation_stats = retrieve(
            features,
            y,
            outer_train,
            outer_valid,
            full_preprocessor,
            full_encoder,
        )

        for context_index, context in enumerate(contexts):
            choice = choose_rule(
                y,
                context["current"],
                context["mask"],
                calibration,
                calibration_stats,
            )
            updated, changed = apply_choice(
                context["current"],
                context["mask"],
                outer_valid,
                validation_stats,
                choice,
            )
            candidates[context_index][outer_valid] = updated
            choice.update(
                {
                    "fold": fold,
                    "valid_changed_rows": changed,
                    "valid_selected_accuracy": (
                        float(
                            np.mean(
                                validation_stats["label"][
                                    updated != context["current"][outer_valid]
                                ]
                                == y[outer_valid][
                                    updated != context["current"][outer_valid]
                                ]
                            )
                        )
                        if changed
                        else None
                    ),
                }
            )
            fold_choices[context_index].append(choice)

        training_diagnostics.append(
            {
                "fold": fold,
                "inner": inner_diag,
                "full": full_diag,
                "calibration_rows": int(len(calibration)),
                "validation_rows": int(len(outer_valid)),
            }
        )
        print(
            f"[embedding] seed={embedding_seed} fold={fold}/{OUTER_SPLITS} "
            f"inner_pos={inner_diag['positive_pairs']} "
            f"full_pos={full_diag['positive_pairs']}",
            flush=True,
        )

    rows = []
    for context, candidate, choices in zip(contexts, candidates, fold_choices):
        current_mae = mean_absolute_error(y, context["current"])
        candidate_mae = mean_absolute_error(y, candidate)
        rows.append(
            {
                "embedding_seed": int(embedding_seed),
                "link_seed": int(context["seed"]),
                "current_oof_mae": float(current_mae),
                "candidate_oof_mae": float(candidate_mae),
                "gain_vs_current": float(current_mae - candidate_mae),
                "changed_rows": int(np.sum(candidate != context["current"])),
                "fold_choices": choices,
            }
        )
    return rows, training_diagnostics


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
    torch.set_num_threads(1)
    v3 = load_module(
        "experiment_v3",
        ROOT / "experiment/experiment_v3_best_record_linkage.py",
    )
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = load_module(
        "experiment_v24",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    # 규칙 준수를 쉽게 확인할 수 있도록 train과 기존 train OOF만 읽는다.
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v3.train_candidate_pairs(features, v3.linkage_blocks(categorical))
    equal_counts = pair_equal_counts(features, pairs)
    contexts = v24.make_link_contexts(v4, train, y, stored)

    positive_all = (equal_counts >= 11) & (
        y[pairs[:, 0]] == y[pairs[:, 1]]
    )
    negative_all = (
        (equal_counts >= 8)
        & (equal_counts <= 10)
        & (y[pairs[:, 0]] != y[pairs[:, 1]])
    )
    print(
        f"candidate pairs={len(pairs):,}, "
        f"positive={positive_all.sum():,}, hard_negative={negative_all.sum():,}",
        flush=True,
    )

    started = time.perf_counter()
    results = []
    diagnostics = {}
    for embedding_seed in EMBEDDING_SEEDS:
        seed_rows, seed_diagnostics = run_seed(
            features,
            y,
            pairs,
            equal_counts,
            contexts,
            int(embedding_seed),
        )
        results.extend(seed_rows)
        diagnostics[int(embedding_seed)] = seed_diagnostics

    gains = np.asarray([row["gain_vs_current"] for row in results])
    scores = np.asarray([row["candidate_oof_mae"] for row in results])
    changes = np.asarray([row["changed_rows"] for row in results])
    no_op_count = sum(
        choice["name"] == "no_op"
        for row in results
        for choice in row["fold_choices"]
    )
    total_choices = sum(len(row["fold_choices"]) for row in results)
    all_improve = bool(np.all(gains > 0))
    mean_gain = float(gains.mean())
    reaches_target = bool(mean_gain >= TARGET_OOF_GAIN and all_improve)

    metrics = {
        "protocol": {
            "test_used": False,
            "embedding_seeds": EMBEDDING_SEEDS,
            "outer_splits": OUTER_SPLITS,
            "calibration_ratio": CALIBRATION_RATIO,
            "epochs": EPOCHS,
            "embedding_dim": EMBEDDING_DIM,
            "positive_rule": "same target and at least 11/16 raw features equal",
            "negative_rule": "different target and 8-10/16 raw features equal",
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "pair_diagnostics": {
            "candidate_pairs": int(len(pairs)),
            "positive_pairs_full_train_diagnostic": int(positive_all.sum()),
            "hard_negative_pairs_full_train_diagnostic": int(negative_all.sum()),
        },
        "training_diagnostics": diagnostics,
        "nested_results": results,
        "summary": {
            "candidate_oof_mae_mean": float(scores.mean()),
            "gain_mean_vs_current": mean_gain,
            "gain_min": float(gains.min()),
            "gain_max": float(gains.max()),
            "wins": int((gains > 0).sum()),
            "ties": int((gains == 0).sum()),
            "losses": int((gains < 0).sum()),
            "changed_rows_mean": float(changes.mean()),
            "no_op_choices": int(no_op_count),
            "total_fold_choices": int(total_choices),
            "all_repeats_improve": all_improve,
            "reaches_target_gain": reaches_target,
            "submission_created": False,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    }
    metrics_path = ROOT / "outputs/experiment_v30_contrastive_linkage_metrics.json"
    metrics_path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Supervised contrastive linkage =====")
    for row in results:
        print(
            f"embedding={row['embedding_seed']} link={row['link_seed']} "
            f"OOF={row['candidate_oof_mae']:.9f} "
            f"gain={row['gain_vs_current']:+.9f} "
            f"changed={row['changed_rows']}"
        )
    print(
        f"평균 OOF={scores.mean():.9f} gain={mean_gain:+.9f} "
        f"승/무/패={(gains > 0).sum()}/{(gains == 0).sum()}/"
        f"{(gains < 0).sum()} changed 평균={changes.mean():.1f}"
    )
    print(f"fold no-op={no_op_count}/{total_choices}")
    print(f"목표 개선 {TARGET_OOF_GAIN:.4f} 도달={reaches_target}")
    print("test를 읽지 않았으며 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={metrics_path}")


if __name__ == "__main__":
    main()
