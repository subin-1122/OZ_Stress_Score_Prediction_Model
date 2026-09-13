"""
FLAML AutoML unlinked 전용 nested 5-fold 광역 탐색
==================================================

각 outer fold의 미연결 학습행 안에서 FLAML이 120회씩 모델과 하이퍼파라미터를
탐색한다. LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees를 합쳐 전체
최대 600회 후보를 평가한다.

누수 방지
---------
- test.csv를 읽지 않는다.
- 전처리기는 outer-train의 미연결 행으로만 fit한다.
- FLAML의 모델 선택은 outer-train 안의 3-fold CV로만 수행한다.
- outer-validation target은 AutoML이 전혀 보지 못하며 최종 OOF 평가에만 쓴다.
- 기존 연결, mean_working 보정, snapping을 동일하게 적용해 최종 성능을 비교한다.
- 평균 개선 0.001 이상일 때만 제출 후보로 인정한다.

실행
----
    .venv/bin/python experiment/experiment_v43_flaml_automl_unlinked.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
OUTER_SEED = 5252
OUTER_SPLITS = 5
INNER_SPLITS = 3
MAX_ITER_PER_FOLD = 120
TRAIN_TIME_LIMIT = 5
ESTIMATORS = ("lgbm", "xgboost", "catboost", "rf", "extra_tree")
BLEND_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 1.00)
TARGET_OOF_GAIN = 0.001


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def automl_history(automl):
    """FLAML 버전별 객체 차이를 안전하게 JSON 형태로 바꾼다."""
    history = getattr(automl, "config_history", {})
    rows = []
    for iteration, record in history.items():
        if isinstance(record, (list, tuple)) and len(record) >= 2:
            estimator = record[0]
            config = record[1]
            elapsed = record[2] if len(record) >= 3 else None
        else:
            estimator, config, elapsed = None, record, None
        rows.append(
            {
                "iteration": int(iteration),
                "estimator": str(estimator),
                "config": to_builtin(config),
                "elapsed": to_builtin(elapsed),
            }
        )
    return rows


def main():
    v31 = __import__("experiment_v31_confidence_sample_weight")
    v1 = v31.load_module(
        "experiment_v1_for_v43", ROOT / "experiment/experiment_v1_models.py"
    )
    v4 = v31.load_module(
        "experiment_v4_for_v43",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    v24 = v31.load_module(
        "experiment_v24_for_v43",
        ROOT / "experiment/experiment_v24_nested_simplex_stacking.py",
    )

    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    v24.check_ids(stored, train[ID_COLUMN], "best_record_linkage_oof")
    y = train[TARGET].to_numpy(float)
    features = v1.create_domain_features(
        train.drop(columns=[ID_COLUMN, TARGET])
    ).replace([np.inf, -np.inf], np.nan)
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    contexts = v24.make_link_contexts(v4, train, y, stored)
    canonical_unlinked = ~contexts[0]["mask"]

    automl_oof = np.zeros(len(train), dtype=float)
    uncertainty_proxy = np.zeros(len(train), dtype=float)
    fold_results = []
    splitter = KFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=OUTER_SEED,
    )
    started = time.perf_counter()
    for fold, (outer_train, outer_valid) in enumerate(splitter.split(train), start=1):
        fit_idx = outer_train[canonical_unlinked[outer_train]]
        preprocessor = v1.make_preprocessor(
            numeric,
            categorical,
            scale_numeric=True,
        )
        x_fit = preprocessor.fit_transform(features.iloc[fit_idx])
        x_valid = preprocessor.transform(features.iloc[outer_valid])
        log_path = ROOT / f"outputs/experiment_v43_flaml_fold{fold}.log"
        automl = AutoML()
        fold_started = time.perf_counter()
        automl.fit(
            X_train=x_fit,
            y_train=y[fit_idx],
            task="regression",
            metric="mae",
            estimator_list=list(ESTIMATORS),
            eval_method="cv",
            split_type="uniform",
            n_splits=INNER_SPLITS,
            max_iter=MAX_ITER_PER_FOLD,
            time_budget=-1,
            train_time_limit=TRAIN_TIME_LIMIT,
            sample=True,
            ensemble=False,
            n_jobs=-1,
            seed=OUTER_SEED * 100 + fold,
            early_stop=False,
            retrain_full=True,
            log_file_name=str(log_path),
            log_type="all",
            model_history=True,
            verbose=0,
        )
        prediction = np.clip(automl.predict(x_valid), 0.0, 1.0)
        automl_oof[outer_valid] = prediction
        # 절대적인 불확실성은 아니며 outer-valid에서 평균 쪽으로 얼마나 크게
        # 벗어났는지를 진단용으로만 저장한다.
        uncertainty_proxy[outer_valid] = np.abs(prediction - np.mean(y[fit_idx]))
        history = automl_history(automl)
        fold_results.append(
            {
                "fold": fold,
                "fit_unlinked_rows": int(len(fit_idx)),
                "outer_valid_rows": int(len(outer_valid)),
                "transformed_features": int(x_fit.shape[1]),
                "search_iterations_requested": MAX_ITER_PER_FOLD,
                "config_history_rows": len(history),
                "best_estimator": str(automl.best_estimator),
                "best_config": to_builtin(automl.best_config),
                "best_inner_cv_loss": float(automl.best_loss),
                "best_config_per_estimator": to_builtin(
                    automl.best_config_per_estimator
                ),
                "best_loss_per_estimator": to_builtin(
                    automl.best_loss_per_estimator
                ),
                "config_history": history,
                "elapsed_seconds": float(time.perf_counter() - fold_started),
                "log_file": str(log_path),
            }
        )
        print(
            f"[FLAML] fold={fold}/{OUTER_SPLITS} fit={len(fit_idx)} "
            f"trials={len(history)} best={automl.best_estimator} "
            f"inner_MAE={automl.best_loss:.6f} "
            f"elapsed={time.perf_counter()-fold_started:.1f}s",
            flush=True,
        )

    baseline = stored["base_oof"].to_numpy(float)
    results = []
    for blend_weight in BLEND_WEIGHTS:
        raw = baseline.copy()
        raw[canonical_unlinked] = (
            (1.0 - blend_weight) * baseline[canonical_unlinked]
            + blend_weight * automl_oof[canonical_unlinked]
        )
        rows = []
        for context in contexts:
            candidate = v24.finalize(
                v4,
                train,
                y,
                raw,
                context["mask"],
                context["label"],
            )
            current_mae = mean_absolute_error(y, context["current"])
            score = mean_absolute_error(y, candidate)
            rows.append(
                {
                    "link_seed": int(context["seed"]),
                    "candidate_oof_mae": float(score),
                    "gain_vs_current": float(current_mae - score),
                }
            )
        gains = np.asarray([row["gain_vs_current"] for row in rows])
        scores = np.asarray([row["candidate_oof_mae"] for row in rows])
        results.append(
            {
                "blend_weight": blend_weight,
                "candidate_oof_mae_mean": float(scores.mean()),
                "gain_mean": float(gains.mean()),
                "gain_min": float(gains.min()),
                "gain_max": float(gains.max()),
                "wins": int((gains > 0).sum()),
                "ties": int((gains == 0).sum()),
                "losses": int((gains < 0).sum()),
                "passes_screen": bool(gains.mean() >= TARGET_OOF_GAIN),
                "context_results": rows,
            }
        )
    results.sort(key=lambda row: row["gain_mean"], reverse=True)
    total_history = sum(row["config_history_rows"] for row in fold_results)
    metrics = {
        "protocol": {
            "test_used": False,
            "outer_seed": OUTER_SEED,
            "outer_splits": OUTER_SPLITS,
            "inner_splits": INNER_SPLITS,
            "max_iter_per_fold": MAX_ITER_PER_FOLD,
            "requested_total_iterations": MAX_ITER_PER_FOLD * OUTER_SPLITS,
            "estimators": ESTIMATORS,
            "blend_weights": BLEND_WEIGHTS,
            "target_oof_gain": TARGET_OOF_GAIN,
        },
        "canonical_unlinked_rows": int(canonical_unlinked.sum()),
        "actual_config_history_rows": int(total_history),
        "fold_results": fold_results,
        "results": results,
        "best": results[0],
        "submission_created": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    path = ROOT / "outputs/experiment_v43_flaml_automl_metrics.json"
    path.write_text(
        json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        ROOT / "outputs/experiment_v43_flaml_automl_oof.npz",
        automl_oof=automl_oof,
        uncertainty_proxy=uncertainty_proxy,
        baseline=baseline,
    )

    print("\n===== FLAML AutoML nested OOF =====")
    print(
        f"요청 탐색={MAX_ITER_PER_FOLD * OUTER_SPLITS}, "
        f"best-history 기록={total_history}"
    )
    print(
        "fold winners="
        + str([row["best_estimator"] for row in fold_results])
    )
    for row in results:
        print(
            f"blend={row['blend_weight']:.2f} "
            f"OOF={row['candidate_oof_mae_mean']:.9f} "
            f"gain={row['gain_mean']:+.9f} "
            f"승/무/패={row['wins']}/{row['ties']}/{row['losses']} "
            f"통과={row['passes_screen']}"
        )
    print("test를 읽지 않았으며 현재 단계에서는 제출 파일을 생성하지 않았습니다.")
    print(f"결과 요약={path}")


if __name__ == "__main__":
    main()
