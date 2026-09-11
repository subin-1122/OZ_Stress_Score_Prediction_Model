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
│   └── experiment_v13_link_top2_blend.py
├── open (3)/                  # DACON 데이터, Git 제외
├── outputs/                   # 예측값과 제출 파일, Git 제외
├── requirements.txt
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

## 누수 방지 원칙

- test 정답을 사용하지 않습니다.
- 전처리, 결측치 통계, 인코딩 기준과 잔차 보정값은 train에서만 학습합니다.
- OOF에서는 검증 행의 정답이 해당 행의 연결값이나 보정값 계산에 들어가지 않게 합니다.
- test는 학습이 끝난 변환과 예측을 적용할 때만 사용합니다.
- 외부 데이터를 사용하지 않습니다.

레코드 연결 방식은 대회 운영진 문의 후 사용 가능한 방법이라는 답변을 확인하고 실험했습니다.

## 다음 실험

Public 결과만 보고 `alpha=0.950`, `0.970`, `1.000`을 순차 탐색하는 계획은 중단했습니다. 내부 검증에서 0.92 이상은 deterministic split seed 5개가 모두 악화했기 때문에, 추가 leaderboard 탐색보다 OOF에서 독립적으로 재현되는 새 신호만 검토합니다.

### 종료한 방향

- alpha 0.75/0.80/0.85/0.90 평균은 alpha=0.75보다 OOF MAE가 0.0000667 나빴고 5개 seed가 모두 악화했습니다.
- 고신뢰 learned 연결행은 OOF에서 top1 정답 일치율이 100%였습니다. 2위 후보는 정답인 경우가 없어서 nested 검증이 모든 fold에서 변경하지 않음을 선택했습니다.
- 두 실험 모두 사전 기준인 전체 OOF 0.0001 이상 개선과 5/5 seed 개선을 통과하지 못했으므로 제출 파일을 만들지 않았습니다.
