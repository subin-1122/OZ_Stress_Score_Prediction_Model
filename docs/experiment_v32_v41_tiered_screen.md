# v32~v41 단계별 신규 모델·전처리 스크리닝

## 공통 판단 기준

요청한 순서대로 후보를 진행했다. 각 후보의 train-only 5-fold OOF를 기존
100-fold raw 예측과 정해진 비율로 합친 뒤, 고신뢰 연결 + mean_working 보정 +
0.01 snapping을 동일하게 적용했다.

성공 기준은 실험 전에 다음과 같이 고정했다.

- 최종 파이프라인 평균 OOF MAE 개선이 0.0022 이상
- 연결 split seed 5개가 모두 개선
- 두 조건을 통과할 때만 100-fold 반복 확인
- 100-fold에서도 통과할 때만 제출 파일 생성

모든 실험은 `train.csv`와 기존 train OOF만 사용했다. `test.csv`와 외부 데이터는
읽지 않았다.

## Tier 1 결과

| 순서 | 실험 | 최선 개선 | 승/무/패 | 판단 |
|---:|---|---:|---:|---|
| 1 | SVR | 기존 단독 OOF 0.191106 | - | 기존 실험에서 기각 |
| 2 | Gaussian Process | +0.000523 | 5/0/0 | 기준 미달 |
| 3 | v2 A/B unlinked 전용 | +0.000281 | 5/0/0 | 기준 미달 |
| 4 | Fold-safe target encoding | +0.000001 | 3/0/2 | 실질 무효 |
| 5 | 대사 위험 복합 플래그 | +0.000481 | 5/0/0 | 기준 미달 |

### SVR 기존 근거

SVR은 새로운 실험이 아니었다. v1에서 단독 OOF MAE가 `0.191106416`으로 당시
ExtraTrees `0.177397700`보다 나빴다. v24의 10개 모델 nested simplex stacking도
모든 fold에서 SVR 가중치를 0으로 선택했다. 같은 실험을 다시 실행하지 않고 기존
실패로 처리했다.

### Gaussian Process

현재 미연결 행만 학습한 RBF/Matern GPR을 비교했다. 최선은
`Matern(nu=2.5, length_scale=8, noise=0.10)` 예측을 25% 혼합한 조합이었다.

- 최종 OOF MAE: `0.117875333`
- 기존 대비 개선: `0.000523333`
- 연결 seed: 5/5 개선

방향은 좋았지만 목표 개선의 약 24%라 100-fold로 올리지 않았다.

### v2 변수 조합의 unlinked 전용 재평가

v2에 실제 남아 있는 조합은 A와 B뿐이므로 없는 C 조합을 새로 만들지 않았다.
A/B를 현재 미연결 행만으로 학습했으며, 최선은 A/B 50:50 예측을 현재 raw와
50% 혼합한 조합이었다.

- 최종 OOF MAE: `0.118118000`
- 개선: `0.000280667`
- 연결 seed: 5/5 개선

### Fold-safe target encoding

7개 범주형 변수를 smoothed target 평균으로 바꿔 기존 입력에 추가했다. 학습행
encoding도 inner 5-fold OOF로 만들어 자기 target이 자기 입력에 들어가지 않게
했다. smoothing 100, 혼합 50%가 `+0.000000667`로 사실상 무승부였다.

### 대사 위험 복합 플래그

외부 임상 경계 대신 fold-train의 60/70/80% 분위수로 고혈당·고콜레스테롤·
고혈압·고BMI를 정의했다. 위험 개수와 2-way/3-way/4-way 동시 플래그를
추가했다. 최선은 60% 경계 모델의 75% 혼합이었다.

- 최종 OOF MAE: `0.117917333`
- 개선: `0.000481333`
- 연결 seed: 5/5 개선

## Tier 2 결과

| 순서 | 실험 | 최선 개선 | 승/무/패 | 판단 |
|---:|---|---:|---:|---|
| 6 | RandomForest | +0.000367 | 4/0/1 | 기준·일관성 미달 |
| 7 | K-means/GMM 잔차 | 0.000000 | 0/5/0 | 변화 없음 |
| 8 | Bayesian Ridge/Lasso/Ridge | +0.000129 | 4/0/1 | 기준·일관성 미달 |
| 9 | Winsorizing | -0.000073 | 0/0/5 | 악화 |

RandomForest 최선은 `sqrt, leaf=1` 모델의 25% 혼합이었다. 클러스터별 잔차
중앙값은 0.01 snapping을 넘길 만큼 크지 않아 모든 설정이 no-op과 같았다.
정규화 선형모델 최선은 Ridge(alpha=100) 50% 혼합이었다. Winsorizing은
fold-train 상하 1/2/5% 경계를 모두 확인했지만 최선 설정도 모든 seed에서
악화했다.

## Tier 3 결과

### 중복 그룹 전체 제외 CV

target을 사용하지 않는 feature 규칙으로 중복 connected component를 만들고,
그룹 전체가 같은 fold로 가도록 GroupKFold를 적용했다.

| 검증 | ExtraTrees OOF MAE |
|---|---:|
| 일반 5-fold | 0.146372250 |
| 중복 그룹 GroupKFold | 0.249145742 |
| 차이 | +0.102773492 |

1,489개 행이 731개의 비단일 그룹에 포함됐고 가장 큰 그룹은 3행이었다. 중복
그룹 전체를 학습에서 제거하면 오차가 매우 커지므로, 행 단위 CV가 근접 중복
구조에 의해 일반화 성능을 크게 낙관한다는 진단이다. 점수를 올리는 모델은
아니지만 이후 작은 OOF 개선을 더 보수적으로 판단해야 한다는 근거가 된다.

### KNN/MICE 결측치 처리

원본 확인 결과 숫자형 결측은 `mean_working`과 여기서 파생된
`work_sleep_strain`에 각각 1,032개 있었다. 중앙값, KNN 5/10 이웃, MICE를
fold-safe하게 비교했다.

최선은 MICE 모델의 10% 혼합이었지만 최종 OOF가 `0.118588667`, 기존 대비
`0.000190000` 악화했고 연결 seed 5개가 모두 악화했다.

## 최종 결론

성공 기준 0.0022를 통과한 후보가 없어 100-fold 추가 검증과 제출 파일을 만들지
않았다. 현재 최종 조합과 최고 Public 점수 `0.1255266667`은 그대로 유지한다.

이번 묶음에서 상대적으로 방향이 좋았던 GPR과 대사 복합 플래그도 각각 목표의
약 24%, 22% 수준이다. 특히 GroupKFold 진단에서 큰 낙관 편향이 확인됐으므로,
이 정도의 작은 5-fold 개선만으로 제출 후보를 만드는 것은 안전하지 않다.

## 재현 코드

- `experiment_v32_gaussian_process.py`
- `experiment_v33_unlinked_v2_feature_sets.py`
- `experiment_v34_fold_safe_target_encoding.py`
- `experiment_v35_metabolic_flags.py`
- `experiment_v36_random_forest.py`
- `experiment_v37_cluster_residual.py`
- `experiment_v38_regularized_linear_unlinked.py`
- `experiment_v39_fold_winsorizing.py`
- `experiment_v40_grouped_duplicate_cv.py`
- `experiment_v41_knn_mice_imputation.py`
