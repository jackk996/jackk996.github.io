from __future__ import annotations
import json
import numpy as np, pandas as pd
from .schema import INDICATOR_COLUMNS, default_indicator_metadata

class RiskPreprocessor:
    def __init__(self, indicator_cols=None, workload_mode="absolute", clip_quantiles=(0.01,0.99)):
        self.indicator_cols = indicator_cols or INDICATOR_COLUMNS
        self.metadata = default_indicator_metadata(); self.workload_mode=workload_mode; self.clip_quantiles=clip_quantiles; self.fitted=False
    def _risk_transform(self, df):
        x = df[self.indicator_cols].astype(float).copy()
        if "functional_compliance_rate" in x: x["functional_compliance_rate"] = 1.0 - x["functional_compliance_rate"]
        if "workload_saturation_deviation" in x and self.workload_mode == "absolute": x["workload_saturation_deviation"] = x["workload_saturation_deviation"].abs()
        return x
    def fit(self, df):
        x=self._risk_transform(df); self.medians_=x.median(numeric_only=True); filled=x.fillna(self.medians_)
        self.lower_=filled.quantile(self.clip_quantiles[0]); self.upper_=filled.quantile(self.clip_quantiles[1])
        clipped=filled.clip(self.lower_, self.upper_, axis=1); self.min_=clipped.min(); self.max_=clipped.max(); self.fitted=True; return self
    def transform(self, df):
        if not self.fitted: raise RuntimeError("RiskPreprocessor must be fit before transform")
        x=self._risk_transform(df).fillna(self.medians_).clip(self.lower_, self.upper_, axis=1)
        return (x-self.min_)/(self.max_-self.min_).replace(0,1)
    def fit_transform(self, df): return self.fit(df).transform(df)
    def save(self, path):
        data={k:getattr(self,k).to_dict() if hasattr(getattr(self,k), 'to_dict') else getattr(self,k) for k in ['indicator_cols','workload_mode','medians_','lower_','upper_','min_','max_']}
        open(path,'w',encoding='utf-8').write(json.dumps(data,ensure_ascii=False,indent=2,default=str))
