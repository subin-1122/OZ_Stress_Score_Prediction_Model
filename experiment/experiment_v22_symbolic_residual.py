"""
gplearn을 이용한 미연결 행 symbolic 잔차 보정
================================================

목적
----
기존의 고신뢰 연결, mean_working 보정, 0.01 snapping까지 모두 적용한 뒤에도
남은 미연결 행의 잔차에 단순한 수식 구조가 있는지 확인한다. Symbolic
Regressor를 독립적인 최종 모델로 쓰지 않고 현재 예측에 작은 잔차 보정으로만
더한다.

검증 원칙
---------
- test.csv는 읽지 않는다.
- 숫자 결측치 중앙값과 표준화는 각 outer fold의 train 부분으로만 fit한다.
- symbolic 식은 outer fold train의 미연결 행 잔차로만 학습한다.
- validation target은 식 탐색이나 보정 강도 학습에 들어가지 않는다.
- 보정 강도는 실행 전에 0.25, 0.50, 1.00으로 고정한다.
- 전체 OOF 0.002 이상 및 연결 seed 5개 전부 개선해야만 독립 재검증한다.
- 이 파일은 제출 CSV를 만들지 않는다.

실행
----
    .venv/bin/python experiment/experiment_v22_symbolic_residual.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
OUTER_SEED = 909
N_SPLITS = 5
MIN_GAIN = 0.002
ALPHAS = (0.25, 0.50, 1.00)


def load_module(name: str, path: Path):
    """기존 파이프라인의 연결·보정 함수를 그대로 불러온다."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snap(values: np.ndarray) -> np.ndarray:
    """예측을 target의 0.01 격자와 0~1 범위에 맞춘다."""
    return np.clip(np.rint(values / 0.01) * 0.01, 0.0, 1.0)


def make_current_predictions(v4, train, y, stored):
    """연결 seed 5개 각각의 현재 최종 OOF 예측과 마스크를 재현한다."""
    base_oof = stored["base_oof"].to_numpy(float)
    learned_label = stored["link_label"].to_numpy(float)
    learned_mask = stored["linked"].astype(bool).to_numpy()
    features = train.drop(columns=[ID_COLUMN, TARGET])
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    pairs = v4.make_rule_pairs(features, numeric, categorical)

    predictions = []
    masks = []
    labels = []
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
        correction = v4.crossfit_work_correction(
            train,
            y,
            base_oof,
            union_mask,
        )
        prediction = snap(base_oof + 0.75 * correction)
        prediction[union_mask] = union_label[union_mask]
        predictions.append(prediction)
        masks.append(union_mask)
        labels.append(union_label)

    return predictions, masks, labels


def numeric_input(train: pd.DataFrame) -> pd.DataFrame:
    """
    symbolic search에는 숫자형 원본과 설명 가능한 기본 파생변수만 사용한다.

    범주형을 임의 숫자로 바꾸면 수식의 순서 관계가 거짓이 될 수 있어 제외한다.
    """
    x = train.drop(columns=[ID_COLUMN, TARGET]).select_dtypes(
        include=np.number
    ).copy()
    height_m = x["height"] / 100.0
    x["bmi"] = x["weight"] / height_m.where(height_m > 0).pow(2)
    x["pulse_pressure"] = (
        x["systolic_blood_pressure"] - x["diastolic_blood_pressure"]
    )
    x["mean_pressure"] = (
        x["systolic_blood_pressure"]
        + 2.0 * x["diastolic_blood_pressure"]
    ) / 3.0
    x["chol_div_glucose"] = x["cholesterol"] / x["glucose"]
    x["bone_div_height"] = x["bone_density"] / x["height"]
    return x.replace([np.inf, -np.inf], np.nan)


def make_symbolic_oof(
    x: pd.DataFrame,
    residual: np.ndarray,
    usable_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """outer 5-fold에서 매번 새 symbolic 식을 탐색해 잔차 OOF를 만든다."""
    correction = np.zeros(len(x), dtype=float)
    expressions: list[dict] = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=OUTER_SEED)
    started = time.perf_counter()

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        usable = train_idx[usable_mask[train_idx]]
        preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        x_train = preprocessor.fit_transform(x.iloc[usable])
        x_valid = preprocessor.transform(x.iloc[valid_idx])

        model = SymbolicRegressor(
            population_size=800,
            generations=10,
            tournament_size=20,
            stopping_criteria=0.001,
            const_range=(-2.0, 2.0),
            init_depth=(2, 5),
            function_set=(
                "add",
                "sub",
                "mul",
                "div",
                "sqrt",
                "log",
                "abs",
                "neg",
                "min",
                "max",
            ),
            metric="mean absolute error",
            parsimony_coefficient=0.01,
            p_crossover=0.80,
            p_subtree_mutation=0.05,
            p_hoist_mutation=0.05,
            p_point_mutation=0.05,
            max_samples=0.85,
            n_jobs=-1,
            verbose=0,
            random_state=OUTER_SEED * 100 + fold,
        )
        model.fit(x_train, residual[usable])
        correction[valid_idx] = model.predict(x_valid)
        expressions.append(
            {
                "fold": fold,
                "train_unlinked_rows": int(len(usable)),
                "expression": str(model._program),
                "training_fitness": float(model._program.raw_fitness_),
                "program_length": int(model._program.length_),
                "program_depth": int(model._program.depth_),
            }
        )
        print(
            f"[symbolic] fold={fold}/{N_SPLITS} "
            f"length={model._program.length_} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    return correction, expressions


def main() -> None:
    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    current_predictions, masks, _ = make_current_predictions(v4, train, y, stored)

    # 모든 연결 seed에서 미연결인 행만 symbolic 식의 학습 대상으로 사용한다.
    common_unlinked = ~np.logical_or.reduce(masks)
    current_average = np.mean(np.column_stack(current_predictions), axis=1)
    residual = y - current_average
    x = numeric_input(train)
    correction, expressions = make_symbolic_oof(x, residual, common_unlinked)

    repeated = []
    for seed, current, linked in zip(
        v4.LINK_SPLIT_SEEDS,
        current_predictions,
        masks,
    ):
        current_mae = mean_absolute_error(y, current)
        scores = {}
        for alpha in ALPHAS:
            candidate = current.copy()
            unlinked = ~linked
            candidate[unlinked] = snap(
                candidate[unlinked] + alpha * correction[unlinked]
            )
            score = mean_absolute_error(y, candidate)
            scores[f"alpha_{alpha:.2f}"] = {
                "oof_mae": float(score),
                "gain_vs_current": float(current_mae - score),
            }
        repeated.append(
            {
                "link_seed": int(seed),
                "current_oof_mae": float(current_mae),
                "scores": scores,
            }
        )

    summary = {}
    for key in repeated[0]["scores"]:
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

    metrics = {
        "protocol": {
            "test_used": False,
            "outer_seed": OUTER_SEED,
            "n_splits": N_SPLITS,
            "minimum_gain": MIN_GAIN,
            "requires_all_five_link_seed_wins": True,
            "alphas": ALPHAS,
            "common_unlinked_rows": int(common_unlinked.sum()),
        },
        "expressions": expressions,
        "summary": summary,
        "repeated": repeated,
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "experiment_v22_symbolic_residual_metrics.json"
    cache_path = output_dir / "experiment_v22_symbolic_residual_oof.npz"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        cache_path,
        symbolic_correction=correction,
        common_unlinked=common_unlinked,
    )

    print("\n===== Symbolic residual 1차 선별 =====")
    for key, result in sorted(
        summary.items(),
        key=lambda item: item[1]["gain_mean_vs_current"],
        reverse=True,
    ):
        print(
            f"{key:12s} OOF={result['oof_mae_mean']:.9f} "
            f"gain={result['gain_mean_vs_current']:+.9f} "
            f"승/무/패={result['seed_wins']}/"
            f"{result['seed_ties']}/{result['seed_losses']} "
            f"통과={result['passes_screen']}"
        )
    print(f"결과 요약={metrics_path}")
    print(f"OOF 캐시={cache_path}")


if __name__ == "__main__":
    main()
