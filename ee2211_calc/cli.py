from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import (
    classify_from_linear_scores,
    classify_linear_system,
    mse,
    parameter_count,
    predict_linear,
    residuals,
    ridge_regression,
)


def _load_matrix(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return np.asarray(json.loads(p.read_text()), dtype=float)
    return np.loadtxt(p, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="EE2211 offline exam calculator toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("param-count")
    p_count.add_argument("--features", type=int, required=True)
    p_count.add_argument("--outputs", type=int, default=1)
    p_count.add_argument("--no-bias", action="store_true")

    p_sys = sub.add_parser("system")
    p_sys.add_argument("matrix", help="CSV or JSON matrix file")

    p_reg = sub.add_parser("ridge-fit")
    p_reg.add_argument("X")
    p_reg.add_argument("y")
    p_reg.add_argument("--lambda", type=float, default=0.0, dest="lam")
    p_reg.add_argument("--no-bias", action="store_true")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("X")
    p_pred.add_argument("W")
    p_pred.add_argument("--y-true")
    p_pred.add_argument("--classify", action="store_true")
    p_pred.add_argument("--no-bias", action="store_true")

    args = parser.parse_args()

    if args.cmd == "param-count":
        total = parameter_count(args.features, args.outputs, include_bias=not args.no_bias)
        print(f"Formula: (features + bias) * outputs")
        print(f"Total parameters = {total}")

    elif args.cmd == "system":
        A = _load_matrix(args.matrix)
        info = classify_linear_system(A)
        print(f"Shape: {info.rows} x {info.cols}")
        print(f"System type: {info.system_type}")
        print(f"Left inverse exists: {info.has_left_inverse}")
        print(f"Right inverse exists: {info.has_right_inverse}")
        print(f"Ordinary inverse exists: {info.has_inverse}")

    elif args.cmd == "ridge-fit":
        X = _load_matrix(args.X)
        y = _load_matrix(args.y)
        W = ridge_regression(X, y, lam=args.lam, include_bias=not args.no_bias)
        print("Formula: W = (Phi^T Phi + lambda I)^(-1) Phi^T y")
        print(f"W shape: {W.shape}")
        print("W:")
        print(np.round(W, 6))

    elif args.cmd == "predict":
        X = _load_matrix(args.X)
        W = _load_matrix(args.W)
        y_pred = predict_linear(X, W, include_bias=not args.no_bias)
        print(f"Prediction shape: {y_pred.shape}")
        if args.classify:
            labels = classify_from_linear_scores(X, W, include_bias=not args.no_bias)
            print("Predicted labels:")
            print(labels)
        else:
            print("Predictions:")
            print(np.round(y_pred, 6))

        if args.y_true:
            y_true = _load_matrix(args.y_true)
            r = residuals(y_true, y_pred)
            print(f"MSE: {mse(y_true, y_pred):.8f}")
            print("Residuals:")
            print(np.round(r, 6))


if __name__ == "__main__":
    main()
