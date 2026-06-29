#!/usr/bin/env python
import argparse, json, pandas as pd
from src.data.validation import validate_risk_dataframe
from src.data.synthetic_generator import generate_synthetic_risk_data
p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--output', default='data/processed/risk_records.csv'); p.add_argument('--synthetic-demo', action='store_true'); a=p.parse_args()
df=generate_synthetic_risk_data() if a.synthetic_demo else pd.read_csv(a.input)
rep=validate_risk_dataframe(df); df.to_csv(a.output,index=False); open(a.output+'.validation.json','w').write(json.dumps(rep.to_dict(),indent=2,ensure_ascii=False))
