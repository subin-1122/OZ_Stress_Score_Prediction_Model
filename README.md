# 스트레스 점수 예측 AI 해커톤

신체 정보, 생활 습관, 수면 패턴 등의 데이터를 이용해 `stress_score`를 예측하는 DACON 해커톤 프로젝트입니다.

- 평가 지표: MAE
- 학습 데이터: 3,000행
- 외부 데이터: 사용하지 않음
- 현재 최고 Public MAE: **0.1255266667**
- 현재 최고 제출 코드: `experiment/experiment_v11_alpha0930_public_probe.py`

## 폴더 구조

```text
stress_predic_DACON/
├── baseline_lightgbm.py
├── experiment/
│   ├── experiment_v1_models.py
│   ├── experiment_v2_extratrees.py
│   ├── experiment_v3_best_record_linkage.py
│   ├── experiment_v4_deterministic_union.py
│   ├── experiment_v5_grid_snap.py
│   ├── experiment_v6_unlinked_blend_grid_snap.py
│   ├── experiment_v8_no_work_correction_diagnostic.py
│   ├── experiment_v9_alpha090_public_check.py
│   ├── experiment_v10_alpha_fine_compare.py
│   ├── experiment_v11_alpha0930_public_probe.py
│   ├── experiment_v12_alpha_ensemble.py
│   ├── experiment_v13_link_top2_blend.py
│   ├── experiment_v14_component_dedup.py
│   ├── experiment_v15_tabpfn_regression.py
│   ├── experiment_v16_gam_quantile.py
│   ├── experiment_v17_gam_independent_confirm.py
│   ├── experiment_v18_gam_full_pipeline_eval.py
│   ├── experiment_v18b_gam_layer_diagnostic.py
│   ├── experiment_v19_deterministic_union_audit.py
│   ├── experiment_v20_gam_work_alpha.py
│   ├── experiment_v21_distributional_median.py
│   ├── experiment_v22_symbolic_residual.py
│   ├── experiment_v23_pure_extratrees_snap.py
│   ├── experiment_v24_nested_simplex_stacking.py
│   └── experiment_v25_mae_aware_linkage.py
├── open (3)/                  # DACON 데이터, Git 제외
├── outputs/                   # 예측값과 제출 파일, Git 제외
├── requirements.txt
├── requirements-tabpfn.txt    # TabPFN 선택 설치 환경
├── requirements-symbolic.txt  # gplearn 선택 설치 환경
└── .vscode/settings.json
```

## 실행 환경 만들기

프로젝트 폴더를 VS Code로 연 뒤 터미널에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

VS Code는 `.vscode/settings.json`에 따라 프로젝트의 `.venv`를 기본 Python 환경으로 사용합니다.

## 데이터 배치

DACON에서 받은 파일을 다음 위치에 둡니다.

```text
open (3)/
├── train.csv
├── test.csv
└── sample_submission.csv
```

대회 원본 데이터와 제출 CSV는 GitHub에 올리지 않습니다. 각 코드가 실행 위치를 기준으로 프로젝트 루트를 자동으로 찾습니다.

## 현재 최고 제출 재현

```bash
.venv/bin/python experiment/experiment_v11_alpha0930_public_probe.py
```

실행이 끝나면 다음 파일이 생성됩니다.

```text
outputs/experimental_alpha093_link_snap_submission.csv
outputs/experimental_alpha093_link_snap_metrics.json
```

## 현재 모델 구성

1. ExtraTrees 3개 seed의 예측을 평균합니다.
2. train에서 확인된 고신뢰 유사 레코드를 연결합니다.
3. 연결되지 않은 행에 `mean_working` 구간별 OOF 잔차 보정을 적용합니다.
4. 현재 탐색 강도 `alpha=0.930`을 사용합니다.
5. train target에서 확인한 0.01 간격으로 예측값을 맞춥니다.

`mean_working` 보정값은 각 검증 행의 정답을 제외한 cross-fitting 방식으로 계산합니다. test 데이터는 학습 통계, 인코딩 기준, 결측치 통계 계산에 포함하지 않습니다.

## 주요 실험 결과

| 단계 | 핵심 변경 | OOF MAE | Public MAE | 판단 |
|---|---|---:|---:|---|
| Baseline | LightGBM | - | 0.17982 | 기준 모델 |
| v3 | ExtraTrees + 고신뢰 연결 | 약 0.11853 | 0.1258158348 | 큰 폭 개선 |
| v4 | deterministic union | 약 0.11844 | 0.1257867444 | 채택 |
| v5 | 0.01 grid snapping | 약 0.11840 | 0.12578 | 소폭 개선 |
| v8 | mean_working 보정 제거 | 약 0.12082 | 0.12928 | 보정 효과 확인 |
| v9 | mean_working alpha=0.900 | 약 0.11843 | 0.12558 | 채택 |
| v11 | mean_working alpha=0.930 | 약 0.11847 | **0.1255266667** | 현재 최고 Public |
| v12 | alpha 0.75~0.90 평균 | 약 0.11847 | 미제출 | 5/5 seed 악화로 기각 |
| v13 | 연결 후보 top1/top2 블렌딩 | 개선 없음 | 미제출 | 5/5 seed 무승부로 기각 |
| v14 | 중복 그룹 가중치/대표행 학습 | 약 0.13559~0.13566 | 미제출 | 개선이 작고 seed별 불안정 |
| v16~v17 | Quantile GAM, 독립 seed 재검증 | 최대 약 0.00120 개선 | 미제출 | 예비 검증 통과 |
| v18~v20 | GAM을 전체 파이프라인에 결합 | 최선 약 0.11850 | 미제출 | 기존 0.11840보다 악화 |
| v19 | deterministic union 보수적 규칙 | 최대 약 0.000085 개선 | 미제출 | 사전 기준 0.0001 미달 |
| v15 | TabPFN 회귀 | 미실행 | 미제출 | 사용자 결정으로 중단 |
| v21 | Ordinal 중앙값·forest proximity | 최선 약 0.000080 개선 | 미제출 | 기준 미달 및 seed 불안정 |
| v22 | gplearn symbolic 잔차 보정 | 최소 0.000231 악화 | 미제출 | 5/5 seed 악화 |
| v23 | 순수 3-seed ExtraTrees + snap | 0.1221033 | 0.12956 | 연결 제거 시 실제 성능 악화 확인 |
| v24 | 10개 OOF 후보 nested simplex stacking | 최선 0.1184800 | 미제출 | 5/5 seed 악화로 기각 |
| v25 | MAE-aware pair ranking + nested gate | 평균 0.1184536 | 미제출 | 4승 2무 19패로 기각 |

### OOF와 Public의 차이

`alpha=0.930`은 `alpha=0.900`보다 OOF MAE가 약 0.000046 나빴지만, Public MAE는 약 0.0000533 좋아졌습니다. 따라서 v11은 OOF 최저 모델이 아니라 Public 보정 강도를 확인하기 위한 탐색 모델이며, 추가 alpha 탐색 결과에 따라 최종값이 바뀔 수 있습니다.

## 실험 코드 안내

- `baseline_lightgbm.py`: 전처리와 LightGBM 기준 모델
- `experiment_v1_models.py`: 여러 기본 모델 비교
- `experiment_v2_extratrees.py`: ExtraTrees 세부 설정 실험
- `experiment_v3_best_record_linkage.py`: 고신뢰 레코드 연결 도입
- `experiment_v4_deterministic_union.py`: 결정 규칙 연결 추가
- `experiment_v5_grid_snap.py`: target의 0.01 간격 반영
- `experiment_v6_unlinked_blend_grid_snap.py`: 미연결 행 혼합 실험
- `experiment_v8_no_work_correction_diagnostic.py`: mean_working 보정 제거 진단
- `experiment_v9_alpha090_public_check.py`: alpha=0.900 제출
- `experiment_v10_alpha_fine_compare.py`: alpha=0.750~1.000 세밀 비교
- `experiment_v11_alpha0930_public_probe.py`: 현재 최고 Public 제출
- `experiment_v12_alpha_ensemble.py`: alpha 0.75/0.80/0.85/0.90 예측 평균 검증
- `experiment_v13_link_top2_blend.py`: 애매한 연결행의 1위·2위 후보 블렌딩 검증
- `experiment_v14_component_dedup.py`: 유사 레코드 그룹의 역가중치 및 대표행 학습 검증
- `experiment_v15_tabpfn_regression.py`: train-only 인코딩을 사용하는 TabPFN 5-fold OOF 실험
- `experiment_v16_gam_quantile.py`: Quantile GAM과 ExtraTrees 블렌딩 예비 스크리닝
- `experiment_v17_gam_independent_confirm.py`: 고정한 GAM 조합을 새 seed와 20-fold로 재검증
- `experiment_v18_gam_full_pipeline_eval.py`: GAM을 기존 100-fold 연결·보정·snapping 파이프라인에 결합
- `experiment_v18b_gam_layer_diagnostic.py`: GAM 효과가 어느 보정 단계에서 사라지는지 분리 진단
- `experiment_v19_deterministic_union_audit.py`: deterministic 연결 규칙을 더 엄격하게 재검증
- `experiment_v20_gam_work_alpha.py`: GAM 결합 후 mean_working alpha를 0.00~1.00으로 재탐색
- `experiment_v21_distributional_median.py`: 101-class 조건부 중앙값과 forest-proximity 중앙값 선별
- `experiment_v22_symbolic_residual.py`: 미연결 행의 post-mean_working 잔차를 symbolic 식으로 보정
- `experiment_v23_pure_extratrees_snap.py`: 연결·잔차 보정을 제거한 순수 ExtraTrees 진단 제출 재현
- `experiment_v24_nested_simplex_stacking.py`: 기존 10개 OOF 예측을 train-only nested CV에서 비음수·합 1 가중치로 스태킹
- `experiment_v25_mae_aware_linkage.py`: 확장 후보의 복사 MAE를 직접 예측하고 별도 nested gate로 적용 여부를 검증

## 누수 방지 원칙

- test 정답을 사용하지 않습니다.
- 전처리, 결측치 통계, 인코딩 기준과 잔차 보정값은 train에서만 학습합니다.
- OOF에서는 검증 행의 정답이 해당 행의 연결값이나 보정값 계산에 들어가지 않게 합니다.
- test는 학습이 끝난 변환과 예측을 적용할 때만 사용합니다.
- 외부 데이터를 사용하지 않습니다.

레코드 연결 방식은 대회 운영진 문의 후 사용 가능한 방법이라는 답변을 확인하고 실험했습니다.

## 최신 추가 실험 결론

Public 결과만 보고 `alpha=0.950`, `0.970`, `1.000`을 순차 탐색하는 계획은 중단했습니다. 내부 검증에서 0.92 이상은 deterministic split seed 5개가 모두 악화했기 때문에, 추가 leaderboard 탐색보다 OOF에서 독립적으로 재현되는 새 신호만 검토합니다.

이번 라운드의 새 모델 후보 중 기존 파이프라인을 안정적으로 이긴 방법은 없었습니다. 레코드 연결 효과를 분리한 순수 ExtraTrees 진단 제출은 Public 0.12956으로, 현재 최종 파이프라인보다 약 0.00403 나빴습니다.

### 상세 판단

- alpha 0.75/0.80/0.85/0.90 평균은 alpha=0.75보다 OOF MAE가 0.0000667 나빴고 5개 seed가 모두 악화했습니다.
- 고신뢰 learned 연결행은 OOF에서 top1 정답 일치율이 100%였습니다. 2위 후보는 정답인 경우가 없어서 nested 검증이 모든 fold에서 변경하지 않음을 선택했습니다.
- 중복 그룹 역가중치는 평균 0.0000117만 개선했고 3승 2패였습니다. 대표행만 남긴 학습은 평균 0.0000573 악화했습니다.
- Quantile GAM은 작은 10/20-fold 검증에서는 모든 seed가 개선했지만, 실제 기준과 같은 100-fold 파이프라인에서는 평균 0.0001367 악화했습니다.
- GAM은 raw unlinked 예측에서 0.002029 개선했지만, `mean_working` 보정 뒤에는 0.0001929 악화했습니다. 두 방법이 같은 잔차 신호를 겹쳐 고친다는 근거입니다.
- GAM 결합 상태에서 `mean_working` alpha를 다시 찾은 최선은 0.65였지만 OOF 0.1185007로, 기존 0.1183987보다 0.000102 나빴습니다.
- 더 엄격한 deterministic 연결은 learned-only 기준 최대 약 0.0000847 개선했지만 사전 채택 기준 0.0001에 못 미쳤고, 현재 규칙보다 실질 개선은 약 0.0000033뿐이었습니다.
- ID 인접성은 바로 옆 ID의 target 차이가 0.3349로 무작위 쌍의 0.3324보다 낫지 않았고, train-test 동일 번호의 feature 일치 수도 무작위와 차이가 없어서 blocking에 사용하지 않았습니다.
- 101-class 조건부 중앙값의 최선은 평균 0.000080 개선에 그쳤고 4승 1패였습니다. forest-proximity 중앙값은 최선 조합도 평균 0.000124 악화했습니다.
- gplearn 잔차 보정은 가장 약한 alpha=0.25에서도 평균 0.000231 악화했고 5개 연결 seed가 모두 악화했습니다. fold별 수식도 대부분 상수 또는 단일 항으로 수렴했습니다.
- 순수 3-seed ExtraTrees의 raw OOF는 0.1221729이고 0.01 snapping 후에는 0.1221033입니다. 실제 Public은 0.12956으로 gap은 약 0.0074567이었으며, 연결·보정 파이프라인이 Public에서도 약 0.00403 우수했습니다.
- 연결 threshold 0.98은 0.97에서 정확했던 7행을 제외했고, 전체 파이프라인 OOF를 모든 seed에서 0.0001167 악화시켜 기각했습니다.
- 기존 10개 모델 OOF를 한꺼번에 결합한 nested simplex stacking은 base 모델 비중을 최소 90%로 제한한 가장 보수적 조합도 OOF 0.1184800으로 기존보다 0.0000813 나빴고, 5개 연결 seed가 모두 악화해 기각했습니다.
- MAE-aware 연결은 후보 쌍을 318,810개에서 995,536개로 넓혔지만, 학습된 top-1 후보를 그대로 사용하면 평균 OOF가 0.04177 악화했습니다. 별도 nested gate도 평균 0.0000549 악화했고 25회 반복에서 4승 2무 19패여서 기각했습니다.
- v25의 oracle 상한이 0.11823으로 크게 나온 것은 행당 평균 약 536개 후보와 101개뿐인 target 격자 때문에 같은 점수가 우연히 후보군에 포함된 결과입니다. 이 값은 실제로 식별 가능한 연결 신호가 아니므로 모델 채택 근거로 사용하지 않습니다.
- 따라서 현재 유지할 조합은 **고신뢰 레코드 연결 + mean_working 보정 + 0.01 snapping**이며, 최고 Public 제출은 alpha=0.930의 0.1255266667입니다.

### 선택 실험 의존성

TabPFN은 더 진행하지 않기로 결정해 실행 대상에서 제외했습니다. symbolic 실험을 재현할 때만 별도 의존성을 설치합니다.

```bash
python -m pip install -r requirements-symbolic.txt
.venv/bin/python experiment/experiment_v22_symbolic_residual.py
```

v21과 v22 모두 test.csv를 읽지 않고 train-only OOF로만 판단하며 제출 파일을 만들지 않습니다.
