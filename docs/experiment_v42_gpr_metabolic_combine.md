# v42 GPR + 대사 복합 플래그 결합

## 변경된 성공 기준

이번 실험부터 평균 OOF MAE 개선 `0.001 이상`을 제출 후보 기준으로 사용한다.
5-fold 스크리닝에서 이 기준을 통과하면 100-fold 반복 확인 후 제출 파일을
생성한다.

## 결합 대상

- v32 최선: Matern GPR 예측을 현재 raw 예측에 25% 혼합
- v35 최선: 60% 분위 대사 복합 플래그 ExtraTrees를 75% 혼합

저장된 train-only OOF만 사용했고 두 가지 결합법을 사전에 고정했다.

- `mean_best`: 두 최선 raw 예측의 단순 평균
- `additive_best`: 두 후보가 baseline에서 이동시킨 보정량을 모두 더함

## 결과

| 결합 | 최종 OOF MAE | 평균 개선 | 승/무/패 | 0.001 통과 |
|---|---:|---:|---:|---|
| additive_best | 0.117920000 | +0.000478667 | 5/0/0 | 실패 |
| mean_best | 0.117970000 | +0.000428667 | 5/0/0 | 실패 |

GPR과 대사 모델의 raw 이동량 상관은 `+0.1604`로 높지 않았지만, 최종 연결·
mean_working·snapping을 적용하면 결합 성능이 GPR 단독 개선 `+0.000523333`보다
작아졌다. 두 신호가 최종 MAE에서 독립적으로 누적되지 않고 일부 상쇄됐다.

## 결론

새 기준 0.001을 넘지 못했으므로 100-fold 재학습과 제출 파일 생성을 하지
않는다. 현재 최고 Public 제출 `0.1255266667`을 유지한다.

- 코드: `experiment/experiment_v42_gpr_metabolic_combine.py`
- 결과: `outputs/experiment_v42_gpr_metabolic_combine_metrics.json`
