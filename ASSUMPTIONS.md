# Engineering Assumptions

- Temporal PCA decay `alpha` is not specified by the paper; it is configurable and should be selected via validation or sensitivity analysis.
- Dropout is placed after LSTM sequence outputs and before attention for LSTM-TDA.
- `functional_compliance_rate` is transformed to risk as `1 - value`.
- `workload_saturation_deviation` uses a configurable bidirectional absolute-deviation transform by default.
- LSTM-SA uses single-head scaled dot-product self-attention and mean pooling over time.
- Default MAPE is safe MAPE with denominator `max(abs(y_true), epsilon)`.
- Synthetic generated records are only for tests and pipeline demonstrations, never paper-result reproduction.
