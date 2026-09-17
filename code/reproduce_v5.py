# -*- coding: utf-8 -*-
"""
=============================================================================
 v5.csv 전 과정 재현 스크립트  —  DACON [초격차] AI 헬스케어 6기 해커톤
 팀 프린트("4조")
=============================================================================
 train.csv / test.csv / sample_submission.csv 만 있으면
 개체 복원(all_matched.csv) → 우도·사전분포 추정 → 사후 중앙값 → v5.csv
 까지 한 번에 수행합니다.

   python reproduce_v5.py
   python reproduce_v5.py --data ../open --out ../submissions

 의존성 : numpy, pandas, scikit-learn
 산출물 :
   <out>/all_matched.csv   개체 복원 결과 (train 3000 + test 3000 = 6000행)
   <out>/v5.csv            제출 파일 (3000행)

-----------------------------------------------------------------------------
 v5 가 무엇인가
-----------------------------------------------------------------------------
 이 해법에는 흔히 말하는 머신러닝 모델이 없습니다. 학습되는 것은 도수표 3종뿐입니다.

   ① 우도표  P(mean_working | y)        101 × 13   (커널 가중 도수표)
   ② 사전분포 P(y | Z=0)                101칸
   ③ 부호 우도표 P(sign(Δdbp) | y)      101 × 3

 추론은 이 표들을 로그로 더하고, 정규화하고, 누적확률 0.5 지점(사후 중앙값)을
 고르는 것이 전부입니다. MAE 를 최소화하는 점추정량이 중앙값이기 때문입니다.

 핵심 구조 : train 3,000행 + test 3,000행 = 6,000행은 실제로는 3,000명이
 두 번씩 기록된 자료입니다. 같은 사람의 두 기록을 미세오차 허용범위로 묶으면
   · Z=2 (둘 다 train)  774명 → 우도·사전분포 학습 재료
   · Z=1 (train 1 + test 1) 1,452명 → test 행의 정답을 그대로 복사 (오차 0)
   · Z=0 (둘 다 test)   774명 = 1,548행 → 실제 예측 대상
 이 되어, 3,000행 중 1,548행만 모형으로 추정하면 됩니다.

 v5 설정 (build_all.py 의 VERSIONS['v5'] 와 동일)
   fit   = 'train_only'  우도 적합에 test 레코드를 일절 사용하지 않음 (규정 안전)
   prior = 1.0           분할편향 사전분포 P(y|Z=0) 사용 (모수형 + 미러 KDE 평균)
   tau_l = 1.00          mean_working 우도 가중
   tau_s = 0.60          sign(Δ이완기혈압) 우도 가중
   tau_m = 0.00          sign(Δmean_working) 우도 미사용  (← v6 에서 추가되는 항)
   mode  = 'safe'        추론 시 test 내 같은 개체의 두 기록을 결합
=============================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# 0. 하이퍼파라미터 — 전부 사람이 정한 값 (실질 자유도 7개)
# =============================================================================
NUM = ['age', 'height', 'weight', 'cholesterol', 'systolic_blood_pressure',
       'diastolic_blood_pressure', 'glucose', 'bone_density']
CATS = ['gender', 'activity', 'smoke_status', 'medical_history',
        'family_medical_history', 'sleep_pattern', 'edu_level']

# 같은 사람의 두 기록 사이에 들어간 미세오차(jitter)의 실측 폭.
# train 내부 중복쌍 774개에서 직접 측정한 값이며, 매칭 허용오차의 근거가 된다.
TOL = {'age': 1.0, 'height': 0.2, 'weight': 0.2, 'cholesterol': 0.5,
       'systolic_blood_pressure': 2.0, 'diastolic_blood_pressure': 2.0,
       'glucose': 0.4, 'bone_density': 0.02}
MULT = 8                      # 후보 탐색 반경 = TOL × MULT (넉넉히 잡고 뒤에서 거른다)

GRID = np.round(np.arange(0, 1.001, 0.01), 2)   # y 후보 격자 101개
NG = len(GRID)

H_LIST = [0.04, 0.06, 0.09]   # 우도 커널폭 (3종 앙상블)
PK_LIST = [0.2, 0.5]          # 라플라스 평활 상수 (2종 앙상블) → 총 6개 조합 평균
KDE_H = 0.25                  # 사전분포용 커널폭
SIGN_H, SIGN_PK = 0.25, 2.0   # 부호 우도표의 커널폭 / 평활 상수

TAU_L = 1.00                  # mean_working 우도 가중
TAU_S = 0.60                  # sign(Δdbp) 우도 가중
PRIOR_W = 1.0                 # 사전분포 가중
SHRINK = 1.00                 # 분할편향 계수 c 의 축소율


# =============================================================================
# 1. 개체 복원 (Record Linkage)
# =============================================================================
def find_pairs(df):
    """미세오차만 다른 두 행을 같은 사람으로 묶는다.

    · 수치형 8개를 TOL×MULT 로 나눠 스케일링한 뒤 체비셰프 거리 1.0 이내를 후보로 수집
      (좌표별로 |Δ| ≤ TOL×MULT 를 모두 만족한다는 뜻)
    · 범주형 7개는 100% 일치해야 함 (같은 사람이면 범주값은 복제되므로 동일)
    · 거리가 가까운 쌍부터 1:1 로 확정 (greedy) — 이미 짝이 있는 행은 건너뛴다

    반환 : partner[i] = i 의 짝 행 인덱스 (짝이 없으면 -1)
    """
    S = np.column_stack([df[c].values / (TOL[c] * MULT) for c in NUM])
    cv = df[CATS].astype(object).where(df[CATS].notna(), 'NA').values

    cand = {a: [] for a in range(len(df))}
    for g in df.gender.unique():                       # 성별로 나눠 탐색 (연산량 절감)
        idx = np.where(df.gender.values == g)[0]
        nn = NearestNeighbors(radius=1.0, metric='chebyshev').fit(S[idx])
        dist, ind = nn.radius_neighbors(S[idx], return_distance=True)
        for a, (dd, jj) in enumerate(zip(dist, ind)):
            A = idx[a]
            for k, b in enumerate(jj):
                B = idx[b]
                if B != A and (cv[A] == cv[B]).all():
                    cand[A].append((dd[k], B))

    par = np.full(len(df), -1)
    uniq = {(min(a, b), max(a, b), d) for a, v in cand.items() for d, b in v}
    for a, b, d in sorted(uniq, key=lambda e: e[2]):
        if par[a] == -1 and par[b] == -1:
            par[a], par[b] = b, a
    return par


# =============================================================================
# 2. 학습되는 표 3종
# =============================================================================
def likelihood(Y, LI, Ia, Ib, Ka, Kb, SRCA, SRCB, train_only, h, pk):
    """우도표 P(mean_working | y) — 101 × 13 도수표.

    정답을 아는 개체(LI)의 (y, mean_working) 쌍을 모아,
    y 축으로 가우시안 커널 가중을 준 도수표를 만들고 열 방향으로 정규화한다.
    분포의 함수 형태를 가정하지 않는 비모수(nonparametric) 추정이다.

    train_only=True 이면 train 레코드에서 나온 관측만 사용한다 (규정 안전).
    """
    oy, ok = [], []
    for I, M, S in ((Ia, Ka, SRCA), (Ib, Kb, SRCB)):
        sel = LI[M[LI] & ((S[LI] == 0) if train_only else np.ones(len(LI), bool))]
        oy.append(Y[sel])
        ok.append(I[sel])
    oy = np.concatenate(oy)
    ok = np.concatenate(ok)

    W = np.exp(-0.5 * ((GRID[:, None] - oy[None, :]) / h) ** 2)
    P = np.zeros((NG, 13))
    for k in range(13):
        P[:, k] = W[:, ok == k].sum(1)
    P += pk                                   # 라플라스 평활 (도수 0 인 칸 보호)
    P /= P.sum(1, keepdims=True)
    return np.log(P)


def prior_Z0(Y, LI, Zc, shrink, _KC):
    """사전분포 P(y | Z=0) — 분할 편향 보정.

    관측 : E[y|Z=2] = 0.4617, E[y|Z=1] = 0.5040.
    스트레스가 낮은 개체일수록 train 으로 배정될 확률이 높았다는 뜻이다.
    한 기록이 train 으로 갈 확률을 p(y) = 0.5 + c(y − 0.5) 로 두면
    두 기록이 모두 test 에 남을 확률은 (1 − p(y))^2 에 비례한다.

    모수형 (1−p)^2 과, Z=2 표본을 미러링한 KDE 를 평균해 최종 사전분포로 쓴다.
    """
    y2 = Y[LI][Zc[LI] == 2]
    c = 3.0 * (y2.mean() - 0.5) * shrink
    pg = np.clip(0.5 + c * (GRID - 0.5), 1e-6, 1 - 1e-6)

    pa = (1 - pg) ** 2
    pa /= pa.sum()                                     # 모수형

    W = np.exp(-0.5 * ((GRID[:, None] - y2[None, :]) / KDE_H) ** 2)
    d = W.sum(1) / _KC
    pb = (d / d.sum())[::-1]                           # 미러 KDE

    pr = (pa + pb) / 2
    return pr / pr.sum()


def cat_lik(code, Y, LI, Zc, K=3):
    """부호 우도표 P(sign(Δx) | y) — 101 × 3.

    같은 사람의 두 기록에서 (뒤 값 − 앞 값)의 부호는 −1 / 0 / +1 세 범주다.
    학습 재료는 Z=2 (두 기록이 모두 train) 인 774개 쌍만 사용한다.
    """
    sel = LI[Zc[LI] == 2]
    W = np.exp(-0.5 * ((GRID[:, None] - Y[sel][None, :]) / SIGN_H) ** 2)
    P = np.zeros((NG, K))
    for k in range(K):
        P[:, k] = W[:, code[sel] == k].sum(1)
    P += SIGN_PK
    P /= P.sum(1, keepdims=True)
    return np.log(P)


# =============================================================================
# 3. 메인
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description='v5.csv 전 과정 재현')
    ap.add_argument('--data', default=None,
                    help='train.csv / test.csv / sample_submission.csv 가 있는 폴더')
    ap.add_argument('--out', default='../submissions', help='출력 폴더')
    args = ap.parse_args()

    data = args.data or os.environ.get('DATA_DIR')
    if data is None:
        for d in ('../data', '../open', './data', './open', '.'):
            if os.path.exists(os.path.join(d, 'train.csv')):
                data = d
                break
    if data is None or not os.path.exists(os.path.join(data, 'train.csv')):
        raise SystemExit('train.csv 를 찾을 수 없습니다. --data 로 폴더를 지정하세요.')
    os.makedirs(args.out, exist_ok=True)

    # ---- 3-1. 적재 ----------------------------------------------------------
    # 전처리는 하지 않는다. 인코딩·스케일링·결측치 대치 모두 미수행이며 원시값을 그대로 쓴다.
    # 확률 모형에서 결측은 "해당 우도항을 생략"하는 것으로 정확히 처리되기 때문이다.
    train = pd.read_csv(f'{data}/train.csv')
    test = pd.read_csv(f'{data}/test.csv')
    samp = pd.read_csv(f'{data}/sample_submission.csv')
    print(f'train {len(train)}행 · test {len(test)}행 로드')

    train['src'] = 0
    test['src'] = 1
    test['stress_score'] = np.nan
    alld = pd.concat([train, test], ignore_index=True)
    N = len(alld)

    # ---- 3-2. 개체 복원 + all_matched.csv 저장 -------------------------------
    partner = find_pairs(alld)
    tr_partner = find_pairs(train)          # train 내부 쌍 (매칭 품질 검증용)

    matched = alld[['ID']].copy()
    matched['src'] = alld.src.values                       # 0=train, 1=test
    matched['partner'] = partner                           # 짝 행 인덱스 (-1 = 없음)
    matched['partner_ID'] = np.where(partner >= 0, alld.ID.values[partner], None)
    matched.to_csv(f'{args.out}/all_matched.csv', index=False)
    print(f'all_matched.csv 저장 · 짝을 찾은 행 {(partner >= 0).sum()} / {N}')

    srcv = alld.src.values
    yall = alld.stress_score.values
    pairs = np.array([(a, partner[a]) for a in range(N) if partner[a] > a])
    A_, B_ = pairs[:, 0], pairs[:, 1]

    Y = np.where(~np.isnan(yall[A_]), yall[A_], yall[B_])   # 개체의 정답 (있으면)
    LAB = ~np.isnan(Y)                                      # 정답을 아는 개체
    Zc = (srcv[A_] == 0).astype(int) + (srcv[B_] == 0).astype(int)   # Z = 0 / 1 / 2

    agree = np.mean([train.stress_score.values[a] == train.stress_score.values[tr_partner[a]]
                     for a in np.where(tr_partner >= 0)[0]])
    print(f'개체 {len(pairs)}개 | Z=0 {np.sum(Zc == 0)} / Z=1 {np.sum(Zc == 1)} / Z=2 {np.sum(Zc == 2)}')
    print(f'train 내부 쌍 {int((tr_partner >= 0).sum() // 2)}쌍 · 타깃 일치율 {agree:.4f}')

    # ---- 3-3. 파생값 --------------------------------------------------------
    mwA, mwB = alld.mean_working.values[A_], alld.mean_working.values[B_]
    Ka, Kb = np.isfinite(mwA), np.isfinite(mwB)             # 결측 여부
    Ia = np.clip(np.nan_to_num(mwA, nan=4), 4, 16).astype(int) - 4   # 4~16 → 0~12
    Ib = np.clip(np.nan_to_num(mwB, nan=4), 4, 16).astype(int) - 4
    SRCA, SRCB = srcv[A_], srcv[B_]

    dbA = alld.diastolic_blood_pressure.values[A_]
    dbB = alld.diastolic_blood_pressure.values[B_]
    SG = (np.sign(dbB - dbA) + 1).astype(int)               # 0 / 1 / 2 코드

    LI = np.where(LAB)[0]        # 학습에 쓰는 개체
    UI = np.where(~LAB)[0]       # 예측 대상 개체 (Z=0, 774명 = 1,548행)
    _KC = np.exp(-0.5 * ((GRID[:, None] - GRID[None, :]) / KDE_H) ** 2).sum(1)

    # ---- 3-4. 표 3종 학습 (test 미사용) --------------------------------------
    lps = [likelihood(Y, LI, Ia, Ib, Ka, Kb, SRCA, SRCB, True, h, pk)
           for h in H_LIST for pk in PK_LIST]              # 커널폭 × 평활 6조합
    q = np.log(np.clip(prior_Z0(Y, LI, Zc, SHRINK, _KC), 1e-12, None))
    lpri = PRIOR_W * (q - q.mean())
    LS = cat_lik(SG, Y, LI, Zc)                            # sign(Δdbp) 우도
    print(f'우도표 {len(lps)}종 · 사전분포 1종 · 부호 우도표 1종 학습 완료')

    # ---- 3-5. 사후분포 → 사후 중앙값 ----------------------------------------
    # 로그로 더하고(언더플로 방지) → 정규화 → 누적확률 0.5 지점 선택
    S = np.zeros((len(UI), NG))
    for lp0 in lps:
        lp = np.zeros((len(UI), NG))
        for I, M in ((Ia, Ka), (Ib, Kb)):                  # 같은 개체의 두 관측을 결합
            s = M[UI]
            lp[s] += TAU_L * lp0[:, I[UI][s]].T
        lp = lp + lpri[None, :]                            # × 사전분포
        lp = lp + TAU_S * LS[:, SG[UI]].T                  # × sign(Δdbp) 우도
        w = np.exp(lp - lp.max(1, keepdims=True))
        S += w / w.sum(1, keepdims=True)                   # 6조합 사후분포 평균
    S /= S.sum(1, keepdims=True)
    med = GRID[(np.cumsum(S, 1) < 0.5).sum(1)]             # 사후 중앙값 = MAE 최적

    # ---- 3-6. 후처리 : 정답을 아는 1,452행은 그대로 복사 ----------------------
    rec = np.empty(N)
    rec[A_] = np.where(LAB, Y, np.nan)
    rec[B_] = rec[A_]
    rec[A_[~LAB]] = med
    rec[B_[~LAB]] = med

    pred = pd.DataFrame({'ID': alld.ID.values[srcv == 1], 'stress_score': rec[srcv == 1]})
    out = samp[['ID']].merge(pred, on='ID', how='left')    # sample_submission 순서 고정
    assert out.stress_score.notna().all(), '결측 발생'
    assert out.stress_score.between(0, 1).all(), '[0,1] 범위 이탈'
    out['stress_score'] = out.stress_score.round(2)        # 0.01 격자로 반올림

    fn = f'{args.out}/v5.csv'
    out.to_csv(fn, index=False)
    print(f'{fn} 저장 · {len(out)}행 · 평균 {out.stress_score.mean():.4f} '
          f'· 고유값 {out.stress_score.nunique()}개')


if __name__ == '__main__':
    main()
