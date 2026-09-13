# v30 supervised contrastive linkage 실험

## 실험 목적

기존의 고신뢰 레코드 연결 규칙이 놓친 미연결 행을 찾기 위해, 서로 닮은
train 행을 가까운 위치에 배치하는 16차원 대조학습 임베딩을 학습했다.
현재 최종 조합인 **고신뢰 연결 + mean_working 보정 + 0.01 snapping**은
변경하지 않고, 임베딩이 고른 새 후보를 일부 미연결 행에만 적용했다.

## 누수 방지

- `test.csv`를 읽지 않았다.
- 숫자 결측치 중앙값, 표준화 값, 원-핫 인코딩은 각 fold의 train 부분에서만
  학습했다.
- 양성·음성 쌍과 신경망도 각 fold의 train 행으로만 만들었다.
- outer-validation 행의 `stress_score`는 최종 평가에만 사용했다.
- 적용 기준은 outer-train 안의 별도 calibration 행에서 골랐고 no-op도 항상
  후보에 포함했다.

## 쌍 정의

- 양성 쌍: 원본 16개 feature 중 11개 이상이 같고 `stress_score`도 같은 쌍
- 어려운 음성 쌍: 8~10개 feature가 같지만 `stress_score`는 다른 쌍

전체 train 진단에서는 양성 후보가 688쌍이었고 688쌍 모두 점수가 같았다.
어려운 음성은 2,453쌍이었다. 실제 학습에서는 해당 fold의 학습 부분 안에
동시에 들어온 쌍만 사용했다.

## 모델과 검증

- MLP 구조: 입력 → 64 → 32 → 16차원 L2 정규화 임베딩
- contrastive loss로 양성 거리는 줄이고 어려운 음성 거리는 벌렸다.
- 3개 신경망 seed × 5-fold로 OOF 후보를 만들었다.
- 가장 가까운 train 행의 점수, 1·2위 거리 차이, top-5의 같은 점수 지지 수로
  confidence를 계산했다.
- 각 outer fold 안에서 다시 inner-fit/calibration으로 나눠 적용 기준을 골랐다.

## 결과

Nested 결과는 기존 최종 OOF와 완전히 같았다.

| 지표 | 결과 |
|---|---:|
| 기존/후보 평균 OOF MAE | 0.118398667 |
| 평균 개선 | 0.000000000 |
| 15회 반복 승/무/패 | 0 / 15 / 0 |
| no-op 선택 | 68 / 75 fold |
| 목표 개선 0.0022 도달 | 실패 |

Nested 방식이 지나치게 보수적이었는지 확인하기 위해 confidence 상위 비율을
사후 진단했다. 이 값은 여러 비율을 모두 본 결과이므로 제출 모델 선택에는
사용하지 않는다.

| 적용 범위 | 평균 개선 | 후보 exact 정확도 | 15회 승/패 |
|---|---:|---:|---:|
| 상위 1% | +0.000044444 | 54.67% | 8 / 7 |
| 상위 2% | -0.000542667 | 30.00% | 0 / 15 |
| 상위 5% | -0.001958222 | 12.88% | 0 / 15 |
| 상위 10% | -0.004447333 | 7.74% | 0 / 15 |
| 상위 20% | -0.009394889 | 4.56% | 0 / 15 |
| 전체 | -0.045269556 | 1.76% | 0 / 15 |

## 결론

상위 1%의 극소수 후보에서도 개선이 seed별로 안정적이지 않았고 효과가 목표의
약 2%에 불과했다. 범위를 조금만 넓혀도 모든 반복에서 악화했다. 따라서 실패
원인은 calibration 문턱만이 아니라 임베딩 최근접 후보의 정밀도 부족이다.

이 실험은 기각하며 제출 파일을 만들지 않는다. 현재 최종 조합과 최고 Public
점수 0.1255266667은 그대로 유지한다.

## 재현 파일

- `experiment/experiment_v30_contrastive_linkage.py`: nested OOF 본 실험
- `experiment/experiment_v30b_contrastive_retrieval_diagnostic.py`: 후보 품질 사후 진단
- `outputs/experiment_v30_contrastive_linkage_metrics.json`: nested 상세 결과
- `outputs/experiment_v30b_contrastive_retrieval_metrics.json`: 사후 진단 상세 결과
