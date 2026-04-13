from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SystemInfo:
    rows: int
    cols: int
    system_type: str
    has_left_inverse: bool
    has_right_inverse: bool
    has_inverse: bool


def with_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a bias column of ones."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def classify_linear_system(A: np.ndarray) -> SystemInfo:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)

    if m < n:
        system_type = "underdetermined"
    elif m == n:
        system_type = "even-determined"
    else:
        system_type = "overdetermined"

    has_left_inverse = bool(rank == n and m >= n)
    has_right_inverse = bool(rank == m and n >= m)
    has_inverse = bool(m == n and rank == n)

    return SystemInfo(m, n, system_type, has_left_inverse, has_right_inverse, has_inverse)


def parameter_count(n_features: int, n_outputs: int = 1, include_bias: bool = True) -> int:
    per_output = n_features + (1 if include_bias else 0)
    return per_output * n_outputs


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """1D polynomial expansion [x, x^2, ... x^degree]."""
    X = np.asarray(X, dtype=float).reshape(-1, 1)
    if degree < 1:
        raise ValueError("degree must be >= 1")
    cols = [X ** d for d in range(1, degree + 1)]
    return np.concatenate(cols, axis=1)


def ridge_regression(X: np.ndarray, y: np.ndarray, lam: float = 0.0, include_bias: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    Phi = with_bias_column(X) if include_bias else X
    n_params = Phi.shape[1]

    reg = lam * np.eye(n_params)
    if include_bias:
        reg[0, 0] = 0.0

    return np.linalg.pinv(Phi.T @ Phi + reg) @ Phi.T @ y


def predict_linear(X: np.ndarray, W: np.ndarray, include_bias: bool = True) -> np.ndarray:
    Phi = with_bias_column(X) if include_bias else np.asarray(X, dtype=float)
    return Phi @ np.asarray(W, dtype=float)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def argmax_labels(scores: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(scores, dtype=float), axis=1)


def one_hot(y: Sequence[int], n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    k = int(np.max(y) + 1) if n_classes is None else n_classes
    Y = np.zeros((len(y), k), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def classify_from_linear_scores(X: np.ndarray, W: np.ndarray, include_bias: bool = True) -> np.ndarray:
    scores = predict_linear(X, W, include_bias=include_bias)
    if scores.ndim == 1 or scores.shape[1] == 1:
        return (scores.reshape(-1) >= 0.5).astype(int)
    return argmax_labels(scores)


def bayes_posterior(p_b_given_a: float, p_a: float, p_b: float) -> float:
    if p_b <= 0:
        raise ValueError("P(B) must be positive")
    return (p_b_given_a * p_a) / p_b


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / sigma
    return (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * z * z)


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (x - mu) / (sigma * sqrt(2))
    return 0.5 * (1 + erf(z))


def pmf_valid(probabilities: Iterable[float], tol: float = 1e-9) -> bool:
    probs = np.asarray(list(probabilities), dtype=float)
    return bool(np.all(probs >= -tol) and abs(float(np.sum(probs)) - 1.0) <= tol)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have same number of elements")
    return float(np.corrcoef(x, y)[0, 1])


def gradient_descent_step(theta: np.ndarray, grad: np.ndarray, alpha: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    grad = np.asarray(grad, dtype=float)
    return theta - alpha * grad


def lambda_effect_summary(lam_small: float, lam_large: float) -> str:
    if lam_large <= lam_small:
        return "larger lambda not provided"
    return (
        "Higher lambda increases regularization: parameter magnitudes shrink, variance reduces, "
        "and training fit typically worsens while overfitting risk decreases."
    )
