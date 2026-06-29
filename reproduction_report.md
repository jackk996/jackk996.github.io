# Reproduction Report

模型与实验流程复现完成，论文数值结果尚不能验证。

## Paper target conflict

- 表5.3“全部20项”: RMSE=0.0462, MAE=0.0361, MAPE=9.5.
- 表5.3下方正文及表5.4至表5.9: RMSE=0.0564, MAE=0.0441, MAPE=11.6.

Both references are stored in `src/evaluation/paper_targets.py` for reporting differences only; they are not used for training or hard-coded outputs.

## Current data status

No real project-risk dataset was found during repository audit. Any generated data or results are `synthetic_demo` only.
