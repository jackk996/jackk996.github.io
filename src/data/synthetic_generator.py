import numpy as np, pandas as pd
from .schema import INDICATOR_COLUMNS

def generate_synthetic_risk_data(projects=2, subsystems=3, months=18, seed=1, start='2020-01'):
    rng=np.random.default_rng(seed); dates=pd.period_range(start, periods=months, freq='M').astype(str); rows=[]
    for p in range(projects):
      for s in range(subsystems):
        latent=0.2+0.05*p+0.03*s
        for i,m in enumerate(dates):
          vals=rng.beta(2,5,len(INDICATOR_COLUMNS)); vals[6]=rng.uniform(0.65,1.0); vals[13]=rng.normal(0,0.2)
          risk=float(np.clip(0.55*latent+0.35*np.nanmean(np.r_[vals[:6],1-vals[6],abs(vals[13]),vals[16:]])+rng.normal(0,0.03),0,1))
          rows.append(dict(project_id=f'P{p+1:02d}', subsystem_id=f'S{s+1:02d}', month=m, risk_score=risk, dataset_label='synthetic_demo', **dict(zip(INDICATOR_COLUMNS, vals))))
    return pd.DataFrame(rows)
