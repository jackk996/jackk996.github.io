from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class WindowDataset:
    X_seq: np.ndarray; y: np.ndarray; metadata: pd.DataFrame

def build_subsystem_windows(df, feature_cols, target_col='risk_score', window_size=6):
    X=[]; y=[]; rows=[]; work=df.copy(); work['month']=pd.to_datetime(work['month'])
    for (p,s), g in work.sort_values('month').groupby(['project_id','subsystem_id']):
        g=g.reset_index(drop=True)
        for end in range(window_size-1, len(g)-1):
            months=g.loc[end-window_size+1:end,'month']
            if not (months.diff().dropna() == pd.DateOffset(months=1)).all(): continue
            if g.loc[end+1,'month'] != g.loc[end,'month'] + pd.DateOffset(months=1): continue
            X.append(g.loc[end-window_size+1:end, feature_cols].to_numpy(float)); y.append([float(g.loc[end+1,target_col])])
            rows.append({'project_id':p,'subsystem_id':s,'input_months':[m.strftime('%Y-%m') for m in months], 'target_month':g.loc[end+1,'month'].strftime('%Y-%m')})
    return WindowDataset(np.asarray(X), np.asarray(y), pd.DataFrame(rows))
