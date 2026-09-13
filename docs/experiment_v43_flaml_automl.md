# v43 FLAML AutoML 광역 탐색

## 목적

사람이 지정한 소수 모델 대신 FLAML로 미연결 서브셋의 넓은 모델·
하이퍼파라미터 공간을 탐색했다.

## 누수 방지 구조

- `test.csv`를 읽지 않았다.
- 바깥 5-fold의 validation 행은 전처리와 AutoML에서 완전히 제외했다.
- 각 outer-train의 미연결 행에서만 결측치·스케일·원-핫 전처리를 fit했다.
- FLAML은 outer-train 내부 3-fold CV의 MAE로 모델을 선택했다.
- 최종 OOF는 AutoML이 보지 않은 outer-validation 예측으로만 계산했다.
- 현재 연결 + mean_working + 0.01 snapping을 똑같이 적용했다.

## 탐색 범위

- Outer fold: 5개
- 각 fold trial: 120개
- 총 trial: 600개
- 후보 모델: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
- 새 제출 기준: 평균 OOF 개선 0.001 이상

FLAML의 `config_history`는 모든 trial이 아니라 최고 기록이 갱신된 시점만
보존한다. 본 실험 콘솔의 `trials=2~8` 표시는 이 갱신 횟수다. 실제 로그의
`record_id`를 다시 세어 각 fold 120개, 총 600개가 실행됐음을 확인했다.

### 모델별 실제 trial 수

| 모델 | trial 수 |
|---|---:|
| LightGBM | 206 |
| XGBoost | 105 |
| CatBoost | 90 |
| RandomForest | 78 |
| ExtraTrees | 121 |
| 합계 | 600 |

Outer fold별 최선 모델은 LightGBM 4회, XGBoost 1회였다.

## Nested OOF 결과

| AutoML 혼합 비율 | 최종 OOF MAE | 평균 개선 | 승/무/패 | 0.001 통과 |
|---:|---:|---:|---:|---|
| 0.10 | 0.118295333 | +0.000103333 | 5/0/0 | 실패 |
| 0.25 | 0.118372000 | +0.000026667 | 3/0/2 | 실패 |
| 0.50 | 0.118388667 | +0.000010000 | 3/0/2 | 실패 |
| 0.75 | 0.118907333 | -0.000508667 | 0/0/5 | 실패 |
| 1.00 | 0.119366667 | -0.000968000 | 0/0/5 | 실패 |

AutoML 단독 예측보다 현재 ExtraTrees에 10%만 혼합한 결과가 가장 좋았다.
방향은 연결 seed 5개에서 모두 같았지만 개선은 기준의 약 10%에 불과했다.

## 결론

600개 조합을 nested 방식으로 탐색했지만 평균 개선 0.001을 넘지 못했다.
100-fold 추가 확인과 제출 파일을 생성하지 않으며 현재 최고 Public
`0.1255266667`을 유지한다.

- 코드: `experiment/experiment_v43_flaml_automl_unlinked.py`
- 의존성: `requirements-automl.txt`
- 결과: `outputs/experiment_v43_flaml_automl_metrics.json`
- trial 로그: `outputs/experiment_v43_flaml_fold1.log` ~ `fold5.log`
