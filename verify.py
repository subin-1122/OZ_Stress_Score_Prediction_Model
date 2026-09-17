import pandas as pd, numpy as np
sub=pd.read_csv('out/submission.csv'); samp=pd.read_csv('data/sample_submission.csv')
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
print("1) columns match:", list(sub.columns)==list(samp.columns))
print("2) ID order identical:", (sub.ID.values==samp.ID.values).all(), "| rows:",len(sub))
print("3) no NaN:", sub.stress_score.notna().all(), "| range:", sub.stress_score.min(), sub.stress_score.max())
print("4) on 0.01 grid:", np.allclose(sub.stress_score.values*100, np.round(sub.stress_score.values*100)))
print("5) dtype:", sub.stress_score.dtype)
# independent re-derivation of the linkage with a DIFFERENT algorithm (brute force)
num=['age','height','weight','cholesterol','systolic_blood_pressure','diastolic_blood_pressure','glucose','bone_density']
cats=['gender','activity','smoke_status','medical_history','family_medical_history','sleep_pattern','edu_level']
TOL=np.array([8.0,1.6,1.6,4.0,16.0,16.0,3.2,0.16])
Tn=tr[num].values.astype(float); En=te[num].values.astype(float)
Tc=tr[cats].astype(object).where(tr[cats].notna(),'NA').values
Ec=te[cats].astype(object).where(te[cats].notna(),'NA').values
hit=0; agree=0
key={}
for i in range(len(tr)): key.setdefault(tuple(Tc[i]),[]).append(i)
exact=np.zeros(len(te),bool); exval=np.full(len(te),np.nan)
for j in range(len(te)):
    for i in key.get(tuple(Ec[j]),[]):
        if (np.abs(Tn[i]-En[j])<=TOL).all():
            exact[j]=True; exval[j]=tr.stress_score.values[i]; break
print("6) brute-force linkage found:",exact.sum(),"test rows (expect 1452)")
m=exact
print("7) submission matches brute-force exact values on those rows:",
      np.mean(np.isclose(sub.stress_score.values[m],exval[m])))
# test-test pairs share identical prediction?
al=pd.read_csv('all_matched.csv'); p=al.partner.values; n=3000
tp=[(a-n,p[a]-n) for a in range(n,2*n) if p[a]>=n and p[a]>a]
d=np.array([abs(sub.stress_score.values[i]-sub.stress_score.values[j]) for i,j in tp])
print("8) test-test pairs (%d) identical prediction: %.4f"%(len(tp),(d==0).mean()))
print("\ndistribution of predictions on the 1548 model-predicted rows:")
mp=~exact
print(pd.Series(sub.stress_score.values[mp]).describe().round(3).to_string())
print("\npredicted-value histogram:", np.histogram(sub.stress_score.values[mp],bins=10,range=(0,1))[0])
