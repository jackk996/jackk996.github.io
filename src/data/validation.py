from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd
from .schema import REQUIRED_COLUMNS, ID_COLUMNS, TARGET_COLUMN, INDICATOR_COLUMNS

@dataclass
class ValidationReport:
    missing_columns: list[str]
    duplicate_records: int
    risk_score_out_of_range: int
    missing_rate: dict[str, float]
    time_gaps: list[dict]
    entity_month_counts: dict[str, int]
    is_valid: bool
    def to_dict(self): return asdict(self)

def validate_risk_dataframe(df: pd.DataFrame) -> ValidationReport:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return ValidationReport(missing_cols, 0, 0, {}, [], {}, False)
    work = df.copy(); work["month"] = pd.to_datetime(work["month"]).dt.to_period("M").dt.to_timestamp()
    dup = int(work.duplicated(ID_COLUMNS).sum())
    oor = int(((work[TARGET_COLUMN] < 0) | (work[TARGET_COLUMN] > 1)).sum())
    gaps=[]; counts={}
    for (p,s), g in work.sort_values("month").groupby(["project_id","subsystem_id"]):
        months = g["month"].to_list(); counts[f"{p}/{s}"] = len(months)
        diffs = pd.Series(months).diff().dropna()
        for idx, d in diffs.items():
            if d != pd.DateOffset(months=1):
                gaps.append({"project_id":p,"subsystem_id":s,"after":str(months[idx-1])[:10],"before":str(months[idx])[:10]})
    miss = {c: float(df[c].isna().mean()) for c in REQUIRED_COLUMNS}
    valid = not missing_cols and dup == 0 and oor == 0
    return ValidationReport(missing_cols, dup, oor, miss, gaps, counts, valid)

def assert_indicator_columns(df: pd.DataFrame) -> None:
    missing = [c for c in INDICATOR_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing risk indicator columns: {missing}")
