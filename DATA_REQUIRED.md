# Data Required for Numerical Reproduction

Precise numerical reproduction requires a monthly subsystem-level table with one row per `(project_id, subsystem_id, month)` and columns `risk_score` plus all 20 configured indicators in `src/data/schema.py`.

Expected complete scale: 10 projects × 12 subsystems × 72 months from 2020-01 to 2025-12, up to 8,640 records. `risk_score` must be in `[0, 1]` and represent the monthly ground-truth risk score. Without this data, generated outputs are pipeline demonstrations only and must be labelled `synthetic_demo`.
