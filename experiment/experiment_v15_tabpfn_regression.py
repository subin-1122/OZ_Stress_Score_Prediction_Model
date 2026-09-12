"""
TabPFNRegressor 선택 실험
=========================

실행 전 준비
------------
1. https://ux.priorlabs.ai 에서 로그인하고 Licenses 탭에서 라이선스를 승인한다.
2. API key는 파일에 적지 말고 현재 터미널 환경변수로만 설정한다.

   export TABPFN_TOKEN="발급받은_API_KEY"

3. 아래 명령으로 실행한다.

   .venv/bin/python experiment/experiment_v15_tabpfn_regression.py

보안 및 대회 규칙
-----------------
- token은 출력하거나 JSON/코드/Git에 저장하지 않는다.
- 로컬 checkpoint만 사용하며 train/test를 외부 API로 보내지 않는다.
- 범주 매핑과 숫자 결측치 중앙값은 각 fold-train에서만 만든다.
- test는 이 OOF 스크리닝에 사용하지 않는다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "stress_score"
ID_COLUMN = "ID"
N_SPLITS = 5
SEED = 11
N_ESTIMATORS = 4
MIN_GAIN = 0.0003


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fold_encode(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """fold-train만으로 결측치와 범주 매핑을 정한다."""
    train_encoded = pd.DataFrame(index=train_frame.index)
    valid_encoded = pd.DataFrame(index=valid_frame.index)
    categorical_indices: list[int] = []

    for position, column in enumerate(train_frame.columns):
        if pd.api.types.is_numeric_dtype(train_frame[column]):
            train_numeric = pd.to_numeric(train_frame[column], errors="coerce")
            valid_numeric = pd.to_numeric(valid_frame[column], errors="coerce")
            median = float(train_numeric.median())
            train_encoded[column] = train_numeric.fillna(median).astype(float)
            valid_encoded[column] = valid_numeric.fillna(median).astype(float)
        else:
            categorical_indices.append(position)
            train_values = train_frame[column].astype("string").fillna("MISSING")
            valid_values = valid_frame[column].astype("string").fillna("MISSING")
            categories = sorted(train_values.unique().tolist())
            mapping = {value: index for index, value in enumerate(categories)}
            train_encoded[column] = train_values.map(mapping).astype(float)
            valid_encoded[column] = valid_values.map(mapping).fillna(-1).astype(float)
    return (
        train_encoded.to_numpy(float),
        valid_encoded.to_numpy(float),
        categorical_indices,
    )


def main() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "TABPFN_TOKEN이 없습니다. API key를 코드에 넣지 말고 현재 터미널의 "
            "환경변수로만 설정한 뒤 다시 실행하세요."
        )

    try:
        from tabpfn import TabPFNRegressor
    except ImportError as error:
        raise ImportError(
            "TabPFN이 없습니다. python -m pip install -r requirements-tabpfn.txt를 "
            "먼저 실행하세요."
        ) from error

    v4 = load_module(
        "experiment_v4",
        ROOT / "experiment/experiment_v4_deterministic_union.py",
    )
    train = pd.read_csv(ROOT / "open (3)/train.csv")
    stored = pd.read_csv(ROOT / "outputs/best_record_linkage_oof.csv")
    y = train[TARGET].to_numpy(float)
    features = train.drop(columns=[ID_COLUMN, TARGET])
    learned_mask = stored["linked"].astype(bool).to_numpy()
    baseline_oof = stored["base_oof"].to_numpy(float)
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    rule_pairs = v4.make_rule_pairs(features, numeric, categorical)
    deterministic = v4.aggregate_oof_labels(rule_pairs, y, len(train), SEED)
    unlinked = ~(learned_mask | np.isfinite(deterministic))

    tabpfn_oof = np.zeros(len(train))
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    started = time.perf_counter()
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train), 1):
        x_train, x_valid, categorical_indices = fold_encode(
            features.iloc[train_idx], features.iloc[valid_idx]
        )
        model = TabPFNRegressor(
            n_estimators=N_ESTIMATORS,
            auto_scale_n_estimators=False,
            categorical_features_indices=categorical_indices,
            device="cpu",
            fit_mode="low_memory",
            random_state=SEED * 1000 + fold,
            show_progress_bar=False,
        )
        model.fit(x_train, y[train_idx])
        tabpfn_oof[valid_idx] = np.clip(
            model.predict(x_valid, output_type="median"), 0.0, 1.0
        )
        print(
            f"[TabPFN] fold={fold}/{N_SPLITS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )

    baseline_mae = mean_absolute_error(y[unlinked], baseline_oof[unlinked])
    tabpfn_mae = mean_absolute_error(y[unlinked], tabpfn_oof[unlinked])
    overall_gain = (baseline_mae - tabpfn_mae) * float(unlinked.mean())
    metrics = {
        "protocol": {
            "n_splits": N_SPLITS,
            "seed": SEED,
            "n_estimators": N_ESTIMATORS,
            "output_type": "median",
            "device": "cpu",
            "minimum_gain": MIN_GAIN,
            "test_used": False,
            "token_stored": False,
        },
        "unlinked_rows": int(unlinked.sum()),
        "baseline_unlinked_mae": float(baseline_mae),
        "tabpfn_unlinked_mae": float(tabpfn_mae),
        "overall_equivalent_gain": float(overall_gain),
        "passes_first_screen": bool(overall_gain >= MIN_GAIN),
    }
    output_path = ROOT / "outputs/experiment_v15_tabpfn_regression_metrics.json"
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== TabPFN OOF 결과 =====")
    print(f"baseline unlinked={baseline_mae:.9f}")
    print(f"TabPFN unlinked  ={tabpfn_mae:.9f}")
    print(f"전체 환산 개선   ={overall_gain:+.9f}")
    print(f"1차 기준 통과    ={metrics['passes_first_screen']}")
    print(f"결과 요약        ={output_path}")


if __name__ == "__main__":
    main()
