# 스트레스 점수 예측 주요 실험 요약 (~9/14)

이 문서는 `subin_experiment` 브랜치에 있는 전체 실험 중 성능 변화가 컸거나 다음 의사결정에 영향을 준 실험만 정리한 문서입니다. 세부 구현은 `experiment/`에 보관하고, 최종 재현 코드(9/14)는 [`09_14_final/09_14_best_model.py`](../09_14_final/09_14_best_model.py)에서 확인할 수 있습니다.

## 최종 결과

| 항목 | 결과 |
|---|---:|
| 평가 지표 | MAE |
| Baseline Public MAE | 0.17982 |
| 최종 Public MAE | **0.12404** |
| Baseline 대비 개선 | **0.05578** |
| 최종 모델 | ExtraTrees + 레코드 연결 + `mean_working` 보정 + snapping + 전역 774쌍 평균 |
| 최종 보정 강도 | `alpha=1.30` |

> **검증값 해석:** 아래 표의 `전체 OOF MAE`, `pair MAE 개선`, `Group CV 개선`은 평가 대상과 분할 방식이 서로 다릅니다. 숫자 크기를 직접 비교하지 않고 각 실험 안에서 기준 모델 대비 좋아졌는지를 판단했습니다. Public MAE는 실제 DACON 제출 결과가 확인된 경우에만 기록했습니다.

## Public 제출 성능 변화

| 단계 | 핵심 변경 | 내부 검증 | Public MAE | 판단 |
|---|---|---:|---:|---|
| Baseline | LightGBM | - | 0.17982 | 기준 모델 |
| v3 | ExtraTrees + 학습형 고신뢰 연결 | 전체 OOF 0.118534 | 0.1258158348 | 큰 폭 개선 |
| v4 | 설명 가능한 deterministic 연결 추가 | 전체 OOF 약 0.11844 | 0.1257867444 | 채택 |
| v5 | target의 0.01 간격으로 snapping | 전체 OOF 약 0.11840 | 0.12578 | 소폭 개선 |
| v8 | `mean_working` 보정 제거 | 전체 OOF 약 0.12082 | 0.12928 | 보정의 실전 효과 확인 |
| v9 | `mean_working alpha=0.90` | 전체 OOF 0.118427 | 0.12558 | 채택 |
| v11 | `mean_working alpha=0.93` | 전체 OOF 0.118473 | 0.1255266667 | 당시 최고 |
| v23 | 연결 없는 순수 ExtraTrees + snapping | 전체 OOF 0.122103 | 0.12956 | 연결 효과 재확인 |
| v44-1 | Gaussian Process 25% 혼합 | OOF 평균 +0.000556 개선 | 0.1258466667 | Public 악화 |
| v44-2 | 대사 복합 플래그 모델 75% 혼합 | OOF 평균 +0.000607 개선 | 0.1255733333 | 미세 악화 |
| v44-3 | GPR + 대사 신호 결합 | OOF 평균 +0.000385 개선 | 0.12564 | Public 악화 |
| v46 | test-test 쌍 통계 모델 | Group CV에서는 개선 | 0.1285133333 | 큰 폭 악화 |
| v50 | 동일인 entity 전용 boosting + 근로시간 보정 | entity MAE +0.001036 개선 | 약 0.1293 | 큰 폭 악화 |
| v55 | 기존 모델의 test-test 751쌍 예측 평균 | pair MAE 평균 +0.005069 개선 | 0.1247666667 | **새로운 큰 개선** |
| v63 | `alpha=1.30` + 전역 test-test 774쌍 평균 | Group CV 25/25 개선 | **0.12404** | **최종 채택** |
| v64b | pair 예측 6구간 잔차 후처리 | 100-fold 내부 검증 +0.001581 | 0.12568 | Public 역전·기각 |

## 주요 오프라인 실험

| 실험 | 접근 방법 | 핵심 결과 | 최종 판단 |
|---|---|---|---|
| v1-v2 | LightGBM, RandomForest, ExtraTrees 등 기본 모델 비교 | ExtraTrees 계열이 가장 유리 | ExtraTrees를 base로 채택 |
| v3-v5 | 고신뢰 연결, deterministic union, 0.01 snapping | 세 요소 모두 Public 개선에 기여 | 채택 |
| v6 | 미연결 행 모델 혼합 | 개선폭이 작고 재검증에서 불안정 | 기각 |
| v10-v12 | `mean_working` alpha 세밀 탐색과 alpha 앙상블 | OOF 최적과 Public 최적이 다름. alpha 앙상블은 5/5 악화 | 단일 alpha 유지 |
| v13 | 연결 후보 1위·2위 블렌딩 | 2위 후보가 추가 정보를 제공하지 못함 | 기각 |
| v14 | 중복 그룹 역가중치·대표행 학습 | 효과가 작고 seed별 불안정 | 기각 |
| v16-v20 | Quantile GAM과 기존 파이프라인 결합 | raw 단계에서는 개선됐지만 `mean_working` 보정 뒤 신호가 중복 | 기각 |
| v21 | Ordinal 조건부 중앙값·forest proximity | 최대 개선 약 0.000080, seed 불안정 | 기각 |
| v22 | gplearn symbolic residual | 가장 약한 설정도 평균 0.000231 악화 | 기각 |
| v24 | 10개 OOF 후보 nested simplex stacking | 가장 보수적인 조합도 기존보다 악화 | 기각 |
| v25 | MAE-aware 후보 ranking과 nested gate | 4승 2무 19패 | 기각 |
| v26-v27 | 근로시간 계층 보정·결측 signature 보정 | nested 검증에서 개선이 사라지거나 악화 | 기각 |
| v28 | corruption-pattern likelihood 연결 | 변경 행이 너무 적고 평균 OOF 악화 | 기각 |
| v29 | 트리 분포·불확실성 후처리 | nested 0승 22무 103패 | 기각 |
| v30 | supervised contrastive linkage | 평균 개선 0, 확장 후보 불안정 | 기각 |
| v31 | duplicate confidence sample weight | 5-fold 신호가 100-fold에서 반대로 전환 | 기각 |
| v32-v41 | GPR, target encoding, 대사 플래그, RF, clustering, 선형모델, winsorizing, MICE | 어느 후보도 당시 제출 기준 개선폭을 통과하지 못함 | 기각 |
| v42 | GPR + 대사 플래그 결합 | 최선 평균 개선 +0.000429 | 기준 미달 |
| v43 | FLAML AutoML 광역 탐색 | 600개 조합 탐색 후에도 평균 개선 0.001 미달 | 기각 |
| v45-v50 | 동일인 쌍 통계·entity 전용 모델 | Group CV는 크게 개선됐지만 Public에서 반복 악화 | 재학습 방식 기각 |
| v54-v57 | 기존 예측의 쌍 평균·재스냅·raw source 비교 | 재학습 없는 50:50 평균만 안정적으로 개선 | **쌍 평균 채택** |
| v58 | 두 행의 트리 예측 1,800개를 합쳐 한 번 절사 | 평균 약 0.000005 악화 | 기각 |
| v59-v60 | 전역 3,000쌍 구조와 누락 pair 점검 | train-train 774쌍 target 일치율 100%, test-test 774쌍 확인 | 전역 쌍 확장 |
| v61-v63 | pair averaging 이후 alpha 재탐색 | `alpha=1.30`이 25/25 model-link 조합 개선, Public도 개선 | **최종 채택** |
| v64-v64b | pair 평균 후 예측 구간별 잔차 보정 | 내부 검증은 강했지만 Public 0.12568로 악화 | **분포 보정 금지** |

## 최종 모델 구성

최종 Public `0.12404` 모델은 다음 순서로 동작합니다.

1. 세 개 seed의 ExtraTrees를 100-fold로 학습합니다.
2. 각 트리 예측에서 상·하위 10%를 제외한 절사평균을 사용합니다.
3. test 행과 고신뢰 train 행이 연결되면 해당 train의 `stress_score`를 사용합니다.
4. 설명 가능한 완전일치 결정 규칙으로 연결 범위를 보완합니다.
5. 미연결 행에는 train OOF 잔차로 만든 `mean_working` 보정을 `alpha=1.30`만큼 적용합니다.
6. 행별 예측을 0.01 간격으로 snapping합니다.
7. target을 사용하지 않는 전역 완전매칭에서 확인한 test-test 774쌍은 두 예측을 50:50으로 평균합니다.
8. 쌍 평균 뒤에는 다시 snapping하지 않습니다.

최종 코드는 다음 명령으로 재현합니다.

```bash
.venv/bin/python 09_14_final/09_14_best_model.py
```

## 가장 중요했던 결론

- 일반 회귀 모델 튜닝보다 **고신뢰 레코드 연결**이 훨씬 큰 개선을 만들었습니다.
- 연결되지 않은 행에는 `mean_working` 잔차 보정이 반복적으로 유효했습니다.
- target 값이 0.01 격자라는 구조를 반영한 snapping이 소폭 개선됐습니다.
- 동일인 두 행으로 새 모델을 재학습하는 방식은 Public에서 크게 악화했습니다.
- 동일인 두 행의 **기존 예측값만 평균**하는 방식은 재학습 없이 분산을 줄여 큰 개선을 만들었습니다.
- train-train 쌍에서 검증된 분포 수준 잔차 보정은 test-test 집단으로 전이되지 않았습니다. v64b 이후에는 다수 test 행을 한 방향으로 이동시키는 보정을 사용하지 않습니다.
- 100-fold OOF는 동일인 상대 행이 학습에 남을 수 있어 낙관적입니다. pair 관련 실험은 가능하면 두 행을 같은 fold에 넣는 Group CV로 확인해야 합니다.

## 규칙 및 저장 원칙

- 외부 데이터와 test 정답을 사용하지 않았습니다.
- 결측치 처리, one-hot encoding, 모델 학습과 보정표 학습에는 test 통계를 사용하지 않았습니다.
- test 입력 행 간 구조 매칭은 대회 운영진이 허용한 범위에서 사용했습니다.
- `train.csv`, `test.csv`, 제출 CSV, OOF 배열과 모델 산출물은 GitHub에 올리지 않습니다.
- `experiment/`에는 성공·실패 코드를 모두 보관하고, 실제 최종 제출 재현은 `09_14_final/`만 사용합니다.

## 관련 문서

- [`experiment_v30_contrastive_linkage.md`](experiment_v30_contrastive_linkage.md)
- [`experiment_v31_confidence_sample_weight.md`](experiment_v31_confidence_sample_weight.md)
- [`experiment_v32_v41_tiered_screen.md`](experiment_v32_v41_tiered_screen.md)
- [`experiment_v42_gpr_metabolic_combine.md`](experiment_v42_gpr_metabolic_combine.md)
- [`experiment_v43_flaml_automl.md`](experiment_v43_flaml_automl.md)
