"""
레코드 연결 1위·2위 후보 블렌딩 실험
=====================================

목적
----
고신뢰 연결행 중 1위와 2위 후보 점수가 가까운 행에서만 두 stress_score를
섞으면 드문 오연결 피해를 줄일 수 있는지 확인한다.

검증 원칙
---------
- 연결 분류기와 후보 확률은 기존 20-fold OOF 방식으로 다시 만든다.
- 2위 후보를 사용할 margin 범위와 혼합 비율은 meta-train에서만 선택한다.
- 선택한 규칙은 meta-validation에 적용하므로 같은 정답으로 규칙을 고르고
  평가하는 double-dipping을 피한다.
- deterministic 연결이 있는 행은 기존 확정 규칙을 유지하고 건드리지 않는다.
- test 데이터와 DACON Public 점수는 이 실험의 채택 판단에 사용하지 않는다.

사전 채택 기준
--------------
- 전체 OOF MAE가 0.0001 이상 개선
- deterministic split seed 5개가 모두 개선
- linked 그룹 MAE도 동시에 개선

이 기준을 통과하기 전에는 제출 CSV를 만들지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
ALPHA = 0.75
GRID_STEP = 0.01

# 결과를 본 뒤 후보를 늘리지 않도록 미리 고정한 작은 탐색 범위다.
MARGIN_QUANTILES = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
SECOND_WEIGHTS = (0.20, 0.35, 0.50)
META_N_SPLITS = 10
MIN_SELECTION_ROWS = 10


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values / GRID_STEP) * GRID_STEP, 0.0, 1.0)


def aggregate_top2_candidates(
    query_pairs: np.ndarray,
    query_probability: np.ndarray,
    train_target: np.ndarray,
    n_train: int,
    n_output: int,
) -> dict[str, np.ndarray]:
    """기존 집계식을 유지하면서 행별 1위와 2위 target 정보를 모두 보존한다."""
    output = {
        "label": np.full(n_output, np.nan),
        "probability": np.zeros(n_output),
        "margin": np.zeros(n_output),
        "support": np.zeros(n_output, dtype=np.int16),
        "first_score": np.zeros(n_output),
        "second_label": np.full(n_output, np.nan),
        "second_probability": np.zeros(n_output),
        "second_score": np.zeros(n_output),
        "second_support": np.zeros(n_output, dtype=np.int16),
    }
    if len(query_pairs) == 0:
        return output

    offset = n_train if query_pairs[:, 0].min() >= n_train else 0
    output_row = query_pairs[:, 0] - offset
    order = np.argsort(output_row, kind="stable")
    output_row = output_row[order]
    ordered_pairs = query_pairs[order]
    ordered_probability = query_probability[order]
    rows, starts = np.unique(output_row, return_index=True)
    ends = np.r_[starts[1:], len(output_row)]

    for row, start, end in zip(rows, starts, ends):
        candidates = pd.DataFrame(
            {
                "label": train_target[ordered_pairs[start:end, 1]],
                "probability": ordered_probability[start:end],
            }
        )
        grouped = candidates.groupby("label").agg(
            max_probability=("probability", "max"),
            sum_probability=("probability", "sum"),
            count=("probability", "size"),
        )
        # 기존 v3와 완전히 같은 점수식을 사용한다.
        grouped["score"] = (
            grouped["max_probability"]
            + 0.15 * np.log1p(grouped["count"])
            + 0.03
            * np.maximum(
                grouped["sum_probability"] - grouped["max_probability"], 0
            )
        )
        grouped = grouped.sort_values("score", ascending=False)

        first = grouped.iloc[0]
        output["label"][row] = float(grouped.index[0])
        output["probability"][row] = float(first["max_probability"])
        output["support"][row] = int(first["count"])
        output["first_score"][row] = float(first["score"])

        if len(grouped) > 1:
            second = grouped.iloc[1]
            output["second_label"][row] = float(grouped.index[1])
            output["second_probability"][row] = float(second["max_probability"])
            output["second_score"][row] = float(second["score"])
            output["second_support"][row] = int(second["count"])

        output["margin"][row] = (
            output["first_score"][row] - output["second_score"][row]
        )
    return output


def make_oof_top2(
    v3,
    train_features: pd.DataFrame,
    y: np.ndarray,
    all_pairs: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, np.ndarray]:
    """기존 연결 OOF와 같은 fold·seed·분류기로 top2 정보를 다시 계산한다."""
    n_rows = len(train_features)
    output = aggregate_top2_candidates(
        np.empty((0, 2), dtype=np.int32),
        np.empty(0),
        y,
        n_train=n_rows,
        n_output=n_rows,
    )
    rng = np.random.default_rng(v3.LINKAGE_SEED)
    splitter = KFold(
        n_splits=v3.LINKAGE_N_SPLITS,
        shuffle=True,
        random_state=v3.LINKAGE_SEED,
    )
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_features), 1):
        in_train = np.zeros(n_rows, dtype=bool)
        in_valid = np.zeros(n_rows, dtype=bool)
        in_train[train_idx] = True
        in_valid[valid_idx] = True
        train_pairs = all_pairs[
            in_train[all_pairs[:, 0]] & in_train[all_pairs[:, 1]]
        ]
        cross = (
            (in_valid[all_pairs[:, 0]] & in_train[all_pairs[:, 1]])
            | (in_valid[all_pairs[:, 1]] & in_train[all_pairs[:, 0]])
        )
        query_pairs = all_pairs[cross].copy()
        reverse = in_train[query_pairs[:, 0]]
        query_pairs[reverse] = query_pairs[reverse][:, ::-1]

        pair_target = (
            y[train_pairs[:, 0]] == y[train_pairs[:, 1]]
        ).astype(np.int8)
        keep = v3.balanced_pair_sample(pair_target, rng)
        scales = v3.robust_scales(train_features, train_idx, numeric)
        classifier = v3.build_link_classifier(v3.LINKAGE_SEED + fold)
        classifier.fit(
            v3.make_pair_features(
                train_features,
                train_pairs[keep],
                numeric,
                categorical,
                scales,
            ),
            pair_target[keep],
        )
        query_probability = classifier.predict_proba(
            v3.make_pair_features(
                train_features,
                query_pairs,
                numeric,
                categorical,
                scales,
            )
        )[:, 1]
        fold_result = aggregate_top2_candidates(
            query_pairs,
            query_probability,
            y,
            n_train=n_rows,
            n_output=n_rows,
        )
        for name in output:
            output[name][valid_idx] = fold_result[name][valid_idx]

        print(
            f"[top2 OOF] fold={fold:02d}/{v3.LINKAGE_N_SPLITS} "
            f"query_pairs={len(query_pairs):,} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return output


def apply_blend(
    prediction: np.ndarray,
    eligible: np.ndarray,
    margin: np.ndarray,
    first_label: np.ndarray,
    second_label: np.ndarray,
    threshold: float,
    second_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """margin 기준을 만족하는 연결행만 top1/top2 혼합값으로 교체한다."""
    selected = eligible & (margin <= threshold)
    result = prediction.copy()
    result[selected] = snap(
        (1.0 - second_weight) * first_label[selected]
        + second_weight * second_label[selected]
    )
    return result, selected


def nested_meta_validation(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    eligible: np.ndarray,
    top2: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """각 meta-validation fold 밖에서 margin 범위와 혼합 비율을 선택한다."""
    nested_prediction = baseline_prediction.copy()
    selections: list[dict] = []
    splitter = KFold(n_splits=META_N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(y), 1):
        selection_pool = eligible[train_idx]
        pool_margin = top2["margin"][train_idx][selection_pool]
        baseline_train_mae = mean_absolute_error(
            y[train_idx], baseline_prediction[train_idx]
        )

        best = {
            "quantile": None,
            "threshold": None,
            "second_weight": 0.0,
            "train_mae": baseline_train_mae,
            "train_affected": 0,
        }
        if len(pool_margin) >= MIN_SELECTION_ROWS:
            for quantile in MARGIN_QUANTILES:
                threshold = float(np.quantile(pool_margin, quantile))
                for second_weight in SECOND_WEIGHTS:
                    candidate, affected = apply_blend(
                        baseline_prediction,
                        eligible,
                        top2["margin"],
                        top2["label"],
                        top2["second_label"],
                        threshold,
                        second_weight,
                    )
                    train_affected = int(affected[train_idx].sum())
                    if train_affected < MIN_SELECTION_ROWS:
                        continue
                    candidate_mae = mean_absolute_error(
                        y[train_idx], candidate[train_idx]
                    )
                    if candidate_mae < best["train_mae"] - 1e-15:
                        best = {
                            "quantile": float(quantile),
                            "threshold": threshold,
                            "second_weight": float(second_weight),
                            "train_mae": float(candidate_mae),
                            "train_affected": train_affected,
                        }

        if best["quantile"] is None:
            valid_affected = np.zeros(len(y), dtype=bool)
        else:
            candidate, valid_affected = apply_blend(
                baseline_prediction,
                eligible,
                top2["margin"],
                top2["label"],
                top2["second_label"],
                float(best["threshold"]),
                float(best["second_weight"]),
            )
            nested_prediction[valid_idx] = candidate[valid_idx]

        selections.append(
            {
                "fold": fold,
                **best,
                "valid_affected": int(valid_affected[valid_idx].sum()),
            }
        )
    return nested_prediction, selections


def margin_diagnostics(
    y: np.ndarray,
    learned_linked: np.ndarray,
    top2: dict[str, np.ndarray],
) -> list[dict]:
    """낮은 margin 구간에 top1 오연결이 실제로 집중되는지 요약한다."""
    available = learned_linked & np.isfinite(top2["second_label"])
    diagnostics: list[dict] = []
    for quantile in (0.02, 0.05, 0.10, 0.20, 0.50, 1.00):
        threshold = float(np.quantile(top2["margin"][available], quantile))
        selected = available & (top2["margin"] <= threshold)
        diagnostics.append(
            {
                "lowest_margin_fraction": quantile,
                "threshold": threshold,
                "rows": int(selected.sum()),
                "top1_exact_accuracy": float(
                    np.mean(top2["label"][selected] == y[selected])
                ),
                "top1_mae": float(
                    mean_absolute_error(y[selected], top2["label"][selected])
                ),
                "top2_exact_accuracy": float(
                    np.mean(top2["second_label"][selected] == y[selected])
                ),
            }
        )
    return diagnostics


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
    base_oof = stored_oof["base_oof"].to_numpy(float)

    train_link = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = train_link.select_dtypes(include=np.number).columns.tolist()
    categorical = train_link.select_dtypes(exclude=np.number).columns.tolist()
    blocks = v3.linkage_blocks(categorical)
    train_pairs = v3.train_candidate_pairs(train_link, blocks)
    top2 = make_oof_top2(
        v3, train_link, y, train_pairs, numeric, categorical
    )

    learned_linked = v3.high_confidence_mask(top2)
    stored_linked = stored_oof["linked"].astype(bool).to_numpy()
    stored_label = stored_oof["link_label"].to_numpy(float)
    linkage_reproduction = {
        "stored_linked_rows": int(stored_linked.sum()),
        "rerun_linked_rows": int(learned_linked.sum()),
        "linked_mask_equal": bool(np.array_equal(stored_linked, learned_linked)),
        "top1_label_equal_on_linked": bool(
            np.allclose(
                stored_label[stored_linked],
                top2["label"][stored_linked],
                equal_nan=True,
            )
        ),
    }
    if not linkage_reproduction["linked_mask_equal"]:
        raise RuntimeError("기존 OOF 연결 mask를 동일하게 재현하지 못했습니다.")
    if not linkage_reproduction["top1_label_equal_on_linked"]:
        raise RuntimeError("기존 OOF top1 연결값을 동일하게 재현하지 못했습니다.")

    diagnostics = margin_diagnostics(y, learned_linked, top2)
    rule_pairs = v4.make_rule_pairs(
        train_link, numeric, categorical
    )
    seed_results: list[dict] = []

    for seed in v4.LINK_SPLIT_SEEDS:
        deterministic_label = v4.aggregate_oof_labels(
            rule_pairs, y, len(train), seed
        )
        deterministic_mask = np.isfinite(deterministic_label)
        union_mask = learned_linked | deterministic_mask
        union_label = top2["label"].copy()
        union_label[deterministic_mask] = deterministic_label[deterministic_mask]

        correction = v4.crossfit_work_correction(
            train, y, base_oof, union_mask
        )
        baseline = snap(np.clip(base_oof + ALPHA * correction, 0.0, 1.0))
        baseline[union_mask] = union_label[union_mask]

        # deterministic 규칙과 충돌하지 않는 learned 연결행만 변경 후보로 둔다.
        eligible = (
            learned_linked
            & ~deterministic_mask
            & np.isfinite(top2["second_label"])
        )
        nested_prediction, selections = nested_meta_validation(
            y, baseline, eligible, top2, int(seed)
        )
        baseline_mae = mean_absolute_error(y, baseline)
        nested_mae = mean_absolute_error(y, nested_prediction)
        baseline_linked_mae = mean_absolute_error(
            y[union_mask], baseline[union_mask]
        )
        nested_linked_mae = mean_absolute_error(
            y[union_mask], nested_prediction[union_mask]
        )
        seed_results.append(
            {
                "seed": int(seed),
                "eligible_rows": int(eligible.sum()),
                "baseline_oof_mae": float(baseline_mae),
                "nested_oof_mae": float(nested_mae),
                "overall_gain": float(baseline_mae - nested_mae),
                "baseline_linked_mae": float(baseline_linked_mae),
                "nested_linked_mae": float(nested_linked_mae),
                "linked_gain": float(baseline_linked_mae - nested_linked_mae),
                "nested_changed_rows": int(
                    np.sum(np.abs(nested_prediction - baseline) > 1e-12)
                ),
                "selections": selections,
            }
        )

    gains = np.array([item["overall_gain"] for item in seed_results])
    linked_gains = np.array([item["linked_gain"] for item in seed_results])
    mean_gain = float(gains.mean())
    passes = bool(
        mean_gain >= 0.0001
        and (gains > 0).all()
        and (linked_gains > 0).all()
    )
    metrics = {
        "protocol": {
            "alpha": ALPHA,
            "margin_quantiles": list(MARGIN_QUANTILES),
            "second_weights": list(SECOND_WEIGHTS),
            "meta_n_splits": META_N_SPLITS,
            "minimum_selection_rows": MIN_SELECTION_ROWS,
            "minimum_overall_gain": 0.0001,
            "requires_all_five_seed_wins": True,
            "requires_all_five_linked_group_wins": True,
        },
        "linkage_reproduction": linkage_reproduction,
        "learned_linked_rows": int(learned_linked.sum()),
        "learned_linked_with_second_candidate": int(
            (learned_linked & np.isfinite(top2["second_label"])).sum()
        ),
        "margin_diagnostics": diagnostics,
        "mean_overall_gain": mean_gain,
        "seed_wins": int((gains > 0).sum()),
        "seed_ties": int((gains == 0).sum()),
        "seed_losses": int((gains < 0).sum()),
        "linked_seed_wins": int((linked_gains > 0).sum()),
        "passes_submission_gate": passes,
        "seed_results": seed_results,
    }
    output_path = ROOT / "outputs/experiment_v13_link_top2_blend_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== margin 진단 =====")
    for row in diagnostics:
        print(
            f"lowest {row['lowest_margin_fraction']:.0%}: "
            f"rows={row['rows']:4d}, top1_acc={row['top1_exact_accuracy']:.4f}, "
            f"top1_mae={row['top1_mae']:.6f}, "
            f"top2_acc={row['top2_exact_accuracy']:.4f}"
        )
    print("\n===== nested top2 블렌딩 결과 =====")
    for row in seed_results:
        print(
            f"seed={row['seed']:5d} eligible={row['eligible_rows']:4d} "
            f"changed={row['nested_changed_rows']:4d} "
            f"gain={row['overall_gain']:+.9f} "
            f"linked_gain={row['linked_gain']:+.9f}"
        )
    print(
        f"평균 개선={mean_gain:+.9f}, "
        f"seed 승/무/패={metrics['seed_wins']}/"
        f"{metrics['seed_ties']}/{metrics['seed_losses']}"
    )
    print(f"제출 기준 통과={passes}")
    print(f"결과 요약={output_path}")


if __name__ == "__main__":
    main()
