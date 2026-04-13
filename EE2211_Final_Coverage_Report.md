# EE2211 Final Coverage Report

## Audit basis

This coverage map was built from the repository materials:

- Lecture notes `L1` to `L12`
- Tutorials and tutorial Q&A
- Past papers and trial solutions
- Optional Python session materials where they reinforced examinable workflows
- The current calculator code

The final exam announcement says the final follows the same style as the midterm: `TF`, `MCQ`, `MRQ`, and `FITB`, with extra focus on the post-midterm material. The toolkit therefore prioritizes:

- fast offline numerical answers for `FITB`
- compact conceptual checks for `TF/MCQ/MRQ`
- explicit in-calculator answer patterns for structured and open-ended midterm story questions
- parameter counting and shape tracking
- deterministic outputs with explicit formulas, dimensions, and trap warnings

## Examiner patterns found in the repo

Recurring patterns across the past papers, tutorials, and lecture notes:

- conceptual trap statements around `AI vs ML`, `supervised vs unsupervised`, `classification vs regression`, `inductive vs deductive`, `correlation vs causation`, `interpolation vs extrapolation`
- repeated midterm answer archetypes around `parameter totals`, `W` dimensions, `bias role`, `ridge-lambda effects`, `inverse types`, `argmax`, and `clustering as unsupervised learning`
- `NOIR`, missing-data, categorical-encoding, and preprocessing questions
- parameter-counting and system-type questions for linear / polynomial / multiclass models
- bias handling as a column of ones
- left inverse / right inverse / ordinary inverse questions
- `ridge` interpretation questions, especially what changes as `lambda` increases
- one-hot multiclass regression/classification with `argmax`
- train prediction, residual, `MSE`, and model-order / overfitting comparisons
- `PMF`, Bayes, and normal-distribution probability calculations
- derivative shape and gradient-descent update questions
- post-midterm evaluation questions: confusion matrices, precision/recall, train/validation/test, k-fold CV
- post-midterm algorithm questions: decision-tree impurity / MSE split, k-means, and basic neural-network forward-pass / activation concepts
- structured story questions asking you to identify learning paradigm, task family, NOIR type, preprocessing choice, or the correct interpretation of a real-world scenario

## Coverage map after this patch

### L1-L3

- Covered: core ML concepts, task/performance/experience, supervised vs unsupervised, classification vs regression, inductive vs deductive, NOIR, descriptive stats, Bayes, PMF unknown solving, normal CDF, correlation, Simpson-style grouped-record helpers, and built-in midterm-style conceptual quick answers
- Tool support:
  - concept / MCQ guide with integrated midterm-style quick answers
  - midterm coverage / open-ended page for direct story-question patterns from the repo papers
  - stats & quartiles
  - NOIR helper
  - Bayes / PMF / normal
  - Pearson correlation / feature selection

### L4-L6

- Covered: systems of linear equations, even / over / under-determined status, left/right/full inverse, least squares, bias handling, multi-output regression, binary sign classification, multiclass one-hot classification, polynomial feature expansion, polynomial model size, ridge regression
- Tool support:
  - inverse / determinant / adjoint
  - transpose & rank
  - linear systems
  - linear regression (single and multi-output)
  - binary classification
  - multiclass classification
  - polynomial regression / classification
  - ridge regression
  - exam-style solver

### L7-L8

- Covered: overfitting / underfitting, bias-variance trade-off, model-order comparison, gradient formulas, scalar GD, exponential GD, derivative-shape logic
- Tool support:
  - tutorial concepts / MCQ guide
  - model order / overfitting
  - gradient descent (scalar)
  - exponential GD regression
  - gradient / loss builder
  - derivative shape checker

### L9

- Covered: decision-tree classification impurity and regression-tree split metrics
- Tool support:
  - `25. Decision Tree Metrics`
  - supports root/child `Gini`, `entropy`, `misclassification`, weighted child impurity, threshold split means, root/child `MSE`, and improvement

### L10

- Covered: confusion-matrix arithmetic, per-class precision/recall, binary metric formulas from `TP/FN/FP/TN`, cross-validation fit counting, validation-based candidate selection
- Covered: open-ended model-evaluation and validation-selection reasoning as used in midterm-style wording
- Tool support:
  - `23. Metrics / Cross-Validation`
  - `27. Classification Metrics`
  - `28. Regression Metrics`

### L11

- Covered: k-means iteration, final centroids, assignments, `WCSS`, optional post-hoc clustering accuracy against known labels
- Covered: open-ended clustering identification questions phrased through unlabeled grouping scenarios
- Tool support:
  - `24. K-Means Clustering`

### L12

- Covered: forward pass through 1-3 dense layers, ReLU / sigmoid / linear activations, layer shapes, parameter counts, final output interpretation
- Covered: open-ended neural-network concept prompts at the introductory level through built-in concept references
- Tool support:
  - `26. Neural Network Forward`

## Automated validation added

External test suite added in [`tests/test_exam_toolkit.py`](/C:/EE2211%20calculator%20resources/tests/test_exam_toolkit.py) covering:

- midterm-style regression values from the repo (`w0`, training MSE, polynomial ridge MSE)
- midterm-style multiclass polynomial ridge values (`total parameters`, bias coefficient, predicted class)
- binary confusion metrics
- cross-validation fit counting and tie-breaking
- decision-tree impurity values from Tutorial 9
- regression-tree split metrics from Tutorial 9
- k-means convergence on a deterministic example
- permutation-based clustering accuracy
- neural-network forward pass
- probability-constraint solving
- midterm-style quick-answer guide presence for the highest-yield conceptual archetypes

The main calculator file also still compiles cleanly, and the built-in GUI self-test was expanded with the new helper logic.

## Current strengths

- Strong alignment with the repo’s `FITB` style for parameter totals, bias extraction, ridge, MSE, argmax class prediction, and now also post-midterm metrics / clustering / tree / NN numerics
- Good support for `TF/MCQ/MRQ` trap statements through concept helpers plus metric/tree/NN pages
- Fully offline and deterministic
- Uses direct computation helpers that can be tested outside the GUI

## Remaining weak spots

- `Random forest` remains conceptual only; there is no dedicated calculator page for ensemble voting or feature-subsampling logic
- `Fuzzy C-means` is not implemented as a solver; only conceptual coverage is present
- `Neural-network` support is intentionally focused on forward-pass and activation/parameter questions, not full training/backprop derivations beyond existing gradient helpers
- Some Yueming-part tutorial coding tasks are intentionally not turned into full coding workflows because the lecture notes explicitly say the final should not test those coding questions directly

## Practical recommendation

For exam use, the highest-value pages now are:

1. `22. Exam-Style Solver`
2. `23. Metrics / Cross-Validation`
3. `24. K-Means Clustering`
4. `25. Decision Tree Metrics`
5. `26. Neural Network Forward`
6. `29. Toolkit Self-Test`

Running the self-test once before the exam is recommended.
