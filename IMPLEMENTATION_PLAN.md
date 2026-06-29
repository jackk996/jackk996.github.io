# Implementation Plan

## File structure

- `configs/strict_paper.yaml`: paper-style defaults, including fixed four-component mode and no effective early stopping.
- `configs/corrected.yaml`: leakage-safe defaults with practical early stopping and data-driven PCA dimensionality.
- `data/template_risk_records.csv`: data template with identifiers, target, and 20 indicators.
- `src/data/`: schema, validation, preprocessing, sequence construction, and optional synthetic data generator.
- `src/features/`: entropy weighting and temporal weighted PCA.
- `src/models/`: TensorFlow/Keras TDA, LSTM-TDA, and baseline architectures.
- `src/training/`: seeding, deterministic setup, model training, and grid search.
- `src/evaluation/`: metrics, paper targets, and experiment orchestration.
- `src/explainability/`: risk traceability and four-level dynamic control rules.
- `scripts/`: CLI entry points for data preparation, training, grid search, full experiments, and report generation.
- `tests/`: unit tests for schema, leakage control, PCA, sequence construction, TDA, risk controls, tracing, seeds, and synthetic labelling.

## Module interfaces

- `validate_risk_dataframe(df) -> ValidationReport`: validates uniqueness, ordering, ranges, missingness, gaps, and entity month counts.
- `RiskPreprocessor.fit(df)`, `transform(df)`: fits only on training data, applies missing handling, clipping/winsorization, risk-direction transforms, and min-max scaling.
- `TemporalWeightedPCA.fit(X, timestamps)`, `transform(X)`, `fit_transform(X, timestamps)`, `inverse_transform(Z)`, `save(path)`, `load(path)`.
- `build_subsystem_windows(df, feature_cols, target_col, window_size) -> WindowDataset`.
- `TemporalDecayAttention`: returns `(context, attention_weights, beta)`.
- `build_lstm_tda_model(...)` and baseline builders return compiled Keras models.
- `run_experiment_suite(config)`: runs input, attention, baseline, window, units, and layer sensitivity experiments while saving seed-level and aggregate outputs.
- `explain_prediction(sample_id, artifacts, prediction)`: returns structured JSON-compatible trace and Chinese Markdown.

## Data flow

1. Load raw CSV from configured path.
2. Validate schema and emit audit artifacts.
3. Split by target month according to config.
4. Fit preprocessing, entropy weights, and PCA on training-period records only.
5. Transform all periods using frozen training artifacts.
6. Build subsystem-local sliding windows by target month without crossing entity boundaries or time gaps.
7. Train models with configured seeds and hyperparameters.
8. Evaluate validation/test metrics, save predictions, figures, model weights, configs, environment metadata, and data hashes.
9. Generate paper comparison report; if data is synthetic or missing, label conclusions as `synthetic_demo` and do not claim paper-value reproduction.

## Ambiguities to confirm

- The paper does not specify `alpha` for temporal PCA decay; it is configurable and searched/reported.
- Dropout placement is not specified; default is between LSTM outputs and attention.
- Workload saturation deviation is bidirectional; transformation policy is configurable.
- LSTM-SA aggregation is unspecified; this implementation averages attention outputs over time.
- MAPE zero handling is unspecified; default is `safe_mape` with configurable epsilon.
- Paper tables contain conflicting LSTM-TDA reference values; both are recorded and compared.

## Acceptance criteria

- Real data, if supplied, validates against identifiers, 20 indicators, `risk_score`, monotonic months, ranges, gaps, and approximate expected scale.
- No preprocessing/PCA fitting uses validation or test target months.
- Windows never cross project/subsystem boundaries or time gaps.
- TDA attention weights sum to one, beta is positive, and gradients reach all trainable parameters.
- Five-seed final results are retained as raw rows and aggregate means/stds.
- Synthetic data outputs are explicitly marked `synthetic_demo`.
- If no real data is supplied, final report states: “模型与实验流程复现完成，论文数值结果尚不能验证”.
