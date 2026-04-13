# EE2211 Calculator

Offline exam toolkit for `EE2211 Introduction to Machine Learning`.

This repo is built for the actual exam constraints:

- fully offline
- fast to use under time pressure
- deterministic outputs
- focused on `TF / MCQ / MRQ / FITB` question styles
- grounded in the lecture notes, tutorials, and past/midterm papers stored in this repo

## What this repo contains

- [EE2211_Exam_Toolkit_GUI_v16_midterm_patched.py](</C:/EE2211 calculator resources/EE2211_Exam_Toolkit_GUI_v16_midterm_patched.py:1>)
  - main offline GUI toolkit
- [ee2211_calc](</C:/EE2211 calculator resources/ee2211_calc>)
  - lightweight CLI/core helpers
- [tests](</C:/EE2211 calculator resources/tests>)
  - automated tests for the calculator logic
- [Lecture Notes](</C:/EE2211 calculator resources/Lecture Notes>)
- [Tutorials](</C:/EE2211 calculator resources/Tutorials>)
- [Past Papers & Trial Solutions](</C:/EE2211 calculator resources/Past Papers & Trial Solutions>)
- [EE2211_Final_Coverage_Report.md](</C:/EE2211 calculator resources/EE2211_Final_Coverage_Report.md:1>)
- [EE2211_Final_Changelog.md](</C:/EE2211 calculator resources/EE2211_Final_Changelog.md:1>)

## Quick start

### Run the GUI

```bash
python EE2211_Exam_Toolkit_GUI_v16_midterm_patched.py
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

### Optional editable install for the CLI

```bash
python -m pip install -e .
```

## Main coverage

The calculator now covers the high-yield exam patterns across `L1-L12`, including:

- linear systems, rank, inverse, left/right inverse
- linear regression, multi-output regression, residuals, `MSE`
- binary classification and multiclass one-hot classification
- polynomial feature expansion and parameter counting
- ridge regularization and lambda-effect questions
- bias / offset as a column of ones
- system type: under / even / over-determined
- NOIR, preprocessing, one-hot encoding, scaling, standardization, imputation
- Pearson correlation and feature selection
- `PMF`, Bayes, normal-distribution questions
- gradient-descent and derivative-shape questions
- model-order / overfitting comparisons
- confusion matrices, precision / recall / specificity / `F1`
- cross-validation counting and model-selection logic
- decision-tree impurity and regression-tree split metrics
- k-means clustering
- basic neural-network forward-pass / activation / parameter-count questions
- midterm-style open-ended scenario questions inside the calculator

## Most useful GUI pages

If you only use a few pages during the exam, start with:

1. `22. Exam-Style Solver`
2. `23. Metrics / Cross-Validation`
3. `24. K-Means Clustering`
4. `25. Decision Tree Metrics`
5. `26. Neural Network Forward`
6. `16. Tutorial Concepts / MCQ Guide`
7. `16b. Midterm Coverage / Open-Ended`
8. `29. Toolkit Self-Test`

## CLI examples

```bash
python -m ee2211_calc.cli param-count --features 5 --outputs 3
python -m ee2211_calc.cli system matrix.csv
python -m ee2211_calc.cli ridge-fit X.csv y.csv --lambda 0.1
python -m ee2211_calc.cli predict X.csv W.csv --y-true y.csv
```

## Validation

Current local validation includes:

- repo-derived regression checks from the midterm-style papers
- multiclass polynomial ridge checks
- decision-tree metrics from Tutorial 9
- k-means checks
- neural-network forward-pass checks
- probability and conditional-probability checks

## Notes

- No internet, API, or cloud dependency is required.
- The GUI is the main intended exam interface.
- The CLI exists as a smaller companion tool for scripted checks.
- The course-material PDFs remain in the repo so the toolkit can stay aligned with the actual examiner style.
