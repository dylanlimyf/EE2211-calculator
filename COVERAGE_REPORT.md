# EE2211 Final Calculator Coverage Map (Current Repo Snapshot)

## Audit status

- Repo scan completed.
- Current repository snapshot contains no lecture/tutorial/past-paper content besides an empty placeholder file.
- As of this audit, the provided Windows zip path is not available inside this container, so source materials could not be ingested yet.

## Files seen during audit

- `.gitkeep`

## Coverage vs exam scope (L1-L12)

Legend:
- ✅ implemented in calculator logic
- ⚠️ partially covered (generic support but not validated against course-specific examples yet)
- ❌ missing due to absent source materials / not yet implemented

### Core linear algebra and model setup

- ✅ bias column auto-insertion (`with_bias_column`)
- ✅ parameter counting (single/multi-output; with/without bias)
- ✅ under/even/over-determined system classification
- ✅ left/right/ordinary inverse existence checks (via rank + shape)
- ✅ linear and ridge regression solvers
- ✅ lambda effect conceptual helper

### Prediction and error analysis

- ✅ linear prediction
- ✅ residuals and MSE
- ✅ multiclass argmax prediction
- ✅ binary threshold prediction
- ⚠️ deterministic exam formatting helpers for all question variants (basic CLI added)

### Feature engineering and preprocessing

- ✅ 1D polynomial feature expansion
- ⚠️ categorical encoding beyond one-hot (label/ordinal variants not yet)
- ⚠️ feature selection aids (correlation helper only)

### Classification

- ✅ binary classification via linear scores + threshold
- ✅ multiclass one-hot workflow primitives
- ⚠️ polynomial classification workflows (building blocks exist, but no end-to-end guided flow yet)

### Probability/statistics

- ✅ Bayes posterior helper
- ✅ PMF validity check
- ✅ normal PDF/CDF helper
- ⚠️ richer probability question templates (conditional tables, independence diagnostics)

### Optimization / derivatives

- ✅ gradient descent update step
- ❌ derivative-shape symbolic reasoning helpers (not yet)

### Post-midterm likely topics (if in course materials)

- ❌ clustering workflows
- ❌ decision trees
- ❌ confusion-matrix-derived metrics beyond MSE
- ❌ topic-specific TF/MCQ/MRQ trap library extracted from actual past papers

## Validation status

- Added unit tests for implemented numerical core.
- Validated math-consistency properties (shape, inverse logic, lambda shrinkage, PMF constraints, multiclass argmax, etc.).
- Could not yet validate against lecture/tutorial/past-paper canonical answers because those resources are not present in this repo snapshot.

## Immediate next iteration plan (once resource zip is added)

1. Parse all lecture/tutorial/past-paper/trial files and produce topic-to-question-pattern matrix.
2. Add deterministic solver templates matching exact examiner phrasing patterns.
3. Add regression tests for every fixed-answer example encountered.
4. Expand post-midterm modules (clustering/tree/metrics) if confirmed in materials.
5. Add rapid exam-mode commands per question type (TF/MCQ/MRQ/FIB).
