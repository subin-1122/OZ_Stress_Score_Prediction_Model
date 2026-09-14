# 09/14 스트레스 점수 예측 최종 모델

Public MAE **0.12404**를 기록한 최종 파이프라인입니다. 과거의 64개 실험 파일과 중간 예측 CSV 없이, 이 폴더의 Python 파일 하나로 학습부터 제출 파일 생성까지 실행할 수 있습니다.

## 폴더 구성

```text
09_14_final/
├── 09_14_best_model.py
└── 09_14_README.md
```

대회 데이터와 생성된 제출 CSV는 GitHub에 올리지 않습니다.

## 사용한 방법

1. **ExtraTrees 회귀**
   - seed 3개: `11`, `101`, `1001`
   - 100-fold OOF
   - seed별 트리 수: `500`, `300`, `300`
   - 각 모델의 트리 예측에서 위·아래 10%를 제외한 절사평균 사용
2. **고신뢰 레코드 연결**
   - test 행과 신체 정보가 매우 비슷한 train 행을 찾습니다.
   - 연결 확률이 `0.97` 이상일 때만 train의 `stress_score`를 사용합니다.
3. **설명 가능한 결정 규칙 연결**
   - 범주형 7개가 모두 같고, 숫자형 2개 이상과 핵심 숫자형 1개 이상이 같은 경우만 추가 연결합니다.
   - 후보 train 점수가 모두 같을 때만 사용합니다.
4. **`mean_working` 잔차 보정**
   - train OOF 잔차의 그룹별 중앙값으로 미연결 행을 보정합니다.
   - Public에서 확인된 최종 강도는 `alpha=1.30`입니다.
5. **0.01 단위 snapping**
   - 각 행 예측값을 실제 target 간격인 0.01 단위로 맞춥니다.
6. **전역 test-test 774쌍 평균**
   - target을 사용하지 않는 최소거리 완전매칭으로 같은 사람의 두 기록을 찾습니다.
   - 두 기록의 예측값을 50:50으로 평균하며, 평균 뒤에는 다시 snapping하지 않습니다.

## 데이터 배치

기본적으로 저장소 루트의 `open (3)` 폴더를 자동으로 찾습니다.

```text
프로젝트 루트/
├── open (3)/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── 09_14_final/
    ├── 09_14_best_model.py
    └── 09_14_README.md
```

## 환경 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn networkx
```

macOS에서 기존 Conda 환경이 함께 표시되더라도, VS Code의 Python 인터프리터는 프로젝트의 `.venv/bin/python`을 선택하면 됩니다.

## 실행 방법

프로젝트 루트에서 실행합니다.

```bash
.venv/bin/python 09_14_final/09_14_best_model.py
```

데이터와 출력 위치를 직접 지정할 수도 있습니다.

```bash
.venv/bin/python 09_14_final/09_14_best_model.py \
  --data-dir "open (3)" \
  --output-dir outputs
```

전체 학습 전에 코드 동작만 빠르게 확인하려면 다음 명령을 사용합니다.

```bash
.venv/bin/python 09_14_final/09_14_best_model.py --quick
```

`--quick`으로 생성된 파일은 트리와 fold 수를 줄인 결과이므로 **제출하면 안 됩니다.**

## 생성 파일

기본 출력 폴더는 저장소 루트의 `outputs/`입니다.

```text
outputs/
├── 09_14_best_submission.csv
└── 09_14_best_run_metrics.json
```

- `09_14_best_submission.csv`: DACON 제출 파일
- `09_14_best_run_metrics.json`: 실행 설정, 연결 행 수, 파일 해시와 검증 결과

## 누수 방지 원칙

- 숫자 결측치 중앙값과 one-hot encoder는 각 fold의 train 부분으로만 학습합니다.
- 레코드 연결 분류기와 거리 크기, `mean_working` 보정표는 train 데이터만으로 학습합니다.
- test 정답, test 전체 평균·중앙값, test 기준 결측치 통계를 사용하지 않습니다.
- test 입력 행 간 구조 매칭은 대회 운영진이 허용한 범위에서만 사용합니다.

## 최종 결과와 주의사항

- 최종 Public MAE: **0.12404**
- `alpha=1.30 + 전역 774쌍 평균` 파일의 점수입니다.
