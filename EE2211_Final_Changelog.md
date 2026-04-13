# EE2211 Final Toolkit Changelog

## Added

- `23. Metrics / Cross-Validation`
  - binary confusion-matrix metric formulas from `TP/FN/FP/TN`
  - k-fold fit counting
  - validation-based hyperparameter selection with simple-model tie-break
- `24. K-Means Clustering`
  - iterative clustering output
  - final centroids, assignments, `WCSS`
  - optional post-hoc clustering accuracy against known labels
- `25. Decision Tree Metrics`
  - root/child `Gini`, `entropy`, `misclassification`
  - weighted child impurity
  - regression-tree threshold split means and `MSE`
- `26. Neural Network Forward`
  - 1-3 layer dense-network forward pass
  - ReLU / sigmoid / linear activations
  - auto-bias support and parameter counting
- integrated midterm-style conceptual quick answers into the concept / MCQ guide
  - parameter-count traps
  - bias-role statements
  - ridge-lambda effects
  - inverse-type rules
  - one-hot / argmax / clustering / encoding archetypes
- `16b. Midterm Coverage / Open-Ended`
  - direct coverage of structured story questions from the repo midterms
  - learning-paradigm helper
  - variable-type / encoding helper
  - preprocessing / data-cleaning helper
  - explicit mappings for the factory-defect, AutoDrive, student-data, and promotion-rate question styles
- external automated tests in [`tests/test_exam_toolkit.py`](/C:/EE2211%20calculator%20resources/tests/test_exam_toolkit.py)

## Improved

- expanded built-in self-test with new post-midterm checks
- preserved existing midterm-style regression / ridge validation checks
- kept the calculator fully offline and centered on exam-speed outputs

## Validated against repo-style examples

- midterm-style linear / polynomial ridge regression numbers
- midterm-style multiclass polynomial ridge numbers
- Tutorial 9 decision-tree impurity and regression-split calculations
- Tutorial 10 cross-validation counting / selection logic
- Tutorial 11 k-means convergence pattern
- Tutorial 12 neural-network forward-pass logic
