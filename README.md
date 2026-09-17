# 데이콘 스트레스 지수 예측 대회

- **Public MAE 1위: 0.12264**
- 대회: DACON [초격차] AI 헬스케어 6기 해커톤
- 팀: `프린트("4조")`
- 저장소: [OZ_Stress_Score_Prediction_Model](https://github.com/subin-1122/OZ_Stress_Score_Prediction_Model)
- 작성일: 2026-09-17

최종 제출 모델은 [`code/reproduce_v5.py`](code/reproduce_v5.py)이며,
`train.csv`, `test.csv`, `sample_submission.csv`로 `v5.csv`를 재현할 수 있습니다.

상세한 모델 설명과 실행 결과는 [`code/README.md`](code/README.md)에서 확인할 수 있습니다.

## 1. 환경 설정

필요한 패키지를 설치합니다.

```bash
pip install numpy pandas scikit-learn
```

저장소 루트에서 다음과 같이 실행합니다.

```bash
python code/reproduce_v5.py --data ./open --out ./submissions
```

`--data`로 지정하는 폴더에는 다음 파일이 있어야 합니다.

- `train.csv`
- `test.csv`
- `sample_submission.csv`

실행이 완료되면 `submissions/v5.csv`와 `submissions/all_matched.csv`가 생성됩니다.

## 2. 폴더 구조

```text
OZ_Stress_Score_Prediction_Model/
├── README.md
├── code/
│   ├── README.md
│   └── reproduce_v5.py
├── submissions/
│   ├── README.md
│   ├── v5.csv             # 실행 후 생성
│   └── all_matched.csv    # 실행 후 생성
├── verify.py
├── v5.csv                 # 최종 제출 파일
└── 스트레스 지수 예측 AI 해커톤 발표.pptx
```

대회 데이터는 저장소에 포함하지 않습니다. 로컬의 `open/` 폴더 등에 별도로 배치해 실행합니다.
