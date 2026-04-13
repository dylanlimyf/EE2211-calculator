<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
# EE2211-calculator

=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
# EE2211 Calculator (Offline Exam Toolkit)

Offline-first Python toolkit for fast exam calculations (TF/MCQ/MRQ/FIB support primitives).

## Quick start

```bash
python -m pip install -e .
pytest
```

## CLI usage

```bash
python -m ee2211_calc.cli param-count --features 5 --outputs 3
python -m ee2211_calc.cli system matrix.csv
python -m ee2211_calc.cli ridge-fit X.csv y.csv --lambda 0.1
python -m ee2211_calc.cli predict X.csv W.csv --y-true y.csv
```

## Notes

- Fully offline (no internet/API dependency).
- Deterministic numerical outputs via NumPy linear algebra.
- Designed for rapid exam workflows; additional examiner-pattern templates will be added once lecture/past-paper resources are present in this repo.
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
