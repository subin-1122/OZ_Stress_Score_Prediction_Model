# v31 duplicate confidence sample-weight 실험

## 가설

확실한 duplicate 행은 target 노이즈가 작고, 가까운 duplicate가 없는 unique 행은
target 노이즈가 더 클 수 있다고 가정했다. ExtraTrees 학습 시 모든 행의 비중을
같게 두지 않고 duplicate confidence가 높은 행의 `sample_weight`를 올렸을 때
성능이 개선되는지 확인했다.

이 실험은 duplicate 그룹의 반복 영향력을 줄이려고 역가중했던 v14와 반대
방향이다.

## Confidence 계산

- 원본 feature 16개 중 10개 이상이 같은 후보 쌍만 사용했다.
- 해당 fold의 train 안에서 target까지 같은 쌍을 duplicate 증거로 사용했다.
- feature 일치 개수별 same-target 비율을 Beta(2, 2)로 완만하게 수축했다.
- 행별 최고 쌍 신뢰도와 duplicate 지지 쌍 수를 결합했다.
- validation 행은 confidence 계산과 모델 학습에서 완전히 제외했다.

10-fold의 학습행 2,700개 중 약 1,170~1,250개 행에 duplicate 증거가 있었다.

## 1차 스크리닝

동일한 5개 seed × 10-fold × 250 trees에서 무가중 baseline과 네 설정을 paired
비교했다. 모든 가중치는 평균 1로 정규화했다.

| 설정: unique → duplicate | Raw 개선 | Raw 승/패 | 최종 개선 | 최종 연결 seed 승/패 |
|---|---:|---:|---:|---:|
| 1.00 → 1.50 | -0.000011 | 2 / 3 | -0.000045 | 1 / 4 |
| 0.80 → 1.50 | -0.000026 | 3 / 2 | +0.000071 | 4 / 1 |
| **0.50 → 2.00** | **+0.000235** | **4 / 1** | **+0.000383** | **5 / 0** |
| 0.25 → 3.00 | -0.000084 | 2 / 3 | -0.000051 | 2 / 3 |

`0.50 → 2.00`만 일관된 가능성을 보였지만, 개선폭은 목표 0.0022의 약 17%에
불과했다. 이 설정 하나만 현재 기준과 같은 3-seed × 100-fold로 확인했다.

## 100-fold 독립 확인

| 지표 | 기존 | Confidence 가중 | 개선 |
|---|---:|---:|---:|
| Raw ExtraTrees OOF MAE | 0.122172869 | 0.122380367 | **-0.000207498** |
| 최종 파이프라인 OOF MAE | 0.118398667 | 0.118613333 | **-0.000214667** |

최종 연결 seed 5개가 모두 악화해 승/무/패는 `0 / 0 / 5`였다. 10-fold에서
보였던 개선이 100-fold에서 방향까지 반대로 바뀌었으므로 일반화되는 신호가
아니라 fold 구성과 후보 선택에 의한 변동으로 판단한다.

## 결론

Confidence 기반 sample-weight는 기각한다. 제출 파일은 생성하지 않았으며 현재
최종 조합인 **고신뢰 레코드 연결 + mean_working 보정 + 0.01 snapping**과 최고
Public 점수 `0.1255266667`은 그대로 유지한다.

## 규칙 준수

- `test.csv`를 읽지 않았다.
- 외부 데이터와 사전학습 모델을 사용하지 않았다.
- 결측치 처리, 인코딩, confidence와 sample weight는 fold-train에서만 만들었다.
- validation target은 OOF 평가에만 사용했다.

## 재현 파일

- `experiment/experiment_v31_confidence_sample_weight.py`: 5-seed × 10-fold 스크리닝
- `experiment/experiment_v31b_confidence_weight_100fold_confirm.py`: 3-seed × 100-fold 확인
- `outputs/experiment_v31_confidence_sample_weight_metrics.json`: 1차 상세 결과
- `outputs/experiment_v31b_confidence_weight_100fold_metrics.json`: 확인 결과
