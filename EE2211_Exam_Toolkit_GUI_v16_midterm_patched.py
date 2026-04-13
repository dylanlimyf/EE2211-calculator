import math
import statistics
from collections import Counter
from itertools import combinations_with_replacement, permutations
from typing import List, Tuple, Optional

import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    FigureCanvasTkAgg = None
    Figure = None
    MATPLOTLIB_AVAILABLE = False


# ------------------------------
# Parsing / formatting helpers
# ------------------------------

def clean_text(text: str) -> str:
    return text.strip().replace('\t', ' ')


def parse_numeric_matrix(text: str) -> np.ndarray:
    """
    Accepts rows separated by newlines or semicolons.
    Values can be separated by commas or spaces.
    Example:
        1, 2, 3
        4, 5, 6
    or
        1 2 3; 4 5 6
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Input is empty.")

    rows: List[List[float]] = []
    for raw_row in text.replace(';', '\n').splitlines():
        row = raw_row.strip()
        if not row:
            continue
        tokens = [tok for tok in row.replace(',', ' ').split() if tok]
        try:
            values = [float(tok) if tok.lower() not in {"nan", "na"} else float("nan") for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"Could not parse numeric row: {raw_row}") from exc
        rows.append(values)

    if not rows:
        raise ValueError("No valid rows found.")

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("All rows must have the same number of values.")

    return np.array(rows, dtype=float)


def parse_numeric_vector(text: str, column: bool = True) -> np.ndarray:
    arr = parse_numeric_matrix(text)
    if arr.ndim != 2:
        raise ValueError("Could not parse vector.")
    if arr.shape[0] == 1 and arr.shape[1] > 1:
        vec = arr.reshape(-1, 1) if column else arr.reshape(-1)
    elif arr.shape[1] == 1:
        vec = arr if column else arr.reshape(-1)
    else:
        # allow a single line like: 1, 2, 3
        if arr.shape[0] > 1 and arr.shape[1] > 1:
            raise ValueError("Expected a vector, but received a multi-column matrix.")
        vec = arr.reshape(-1, 1) if column else arr.reshape(-1)
    return vec


def parse_label_list(text: str) -> List[str]:
    """
    Safer label parser for classification tasks.

    Recommended input:
    - one label per line, OR
    - comma-separated labels

    Spaces inside a label are preserved, so labels like
    "Very Easy" or "Class A" stay intact.
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Label input is empty.")

    labels: List[str] = []
    for line in text.replace(';', '\n').splitlines():
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            parts = [part.strip() for part in line.split(',') if part.strip()]
            labels.extend(parts)
        else:
            labels.append(line)

    if not labels:
        raise ValueError("No labels found.")

    return labels


def format_array(arr: np.ndarray, decimals: int = 4) -> str:
    return np.array2string(np.asarray(arr), precision=decimals, suppress_small=False)


def safe_inv(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


def ensure_2d_float_array(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    return arr


def ensure_no_missing_or_infinite(arr: np.ndarray, name: str) -> None:
    arr = np.asarray(arr, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains missing values (NaN/NA). Clean or impute them first.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")


def ensure_no_all_nan_columns(X: np.ndarray, operation_name: str) -> None:
    X = ensure_2d_float_array(X, "X")
    bad_cols = np.where(np.all(np.isnan(X), axis=0))[0]
    if bad_cols.size:
        human_cols = ", ".join(str(int(idx) + 1) for idx in bad_cols)
        raise ValueError(
            f"{operation_name} cannot be performed because column(s) {human_cols} contain only missing values."
        )


def validate_probability(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1.")


def binary_sign(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0, 1.0, -1.0)


def mixed_label_sort_key(label: str):
    try:
        return (0, float(label), str(label))
    except (TypeError, ValueError):
        return (1, str(label))


def solve_least_squares(
    X: np.ndarray,
    Y: np.ndarray,
    ridge_lambda: float = 0.0,
) -> Tuple[np.ndarray, str]:
    X = ensure_2d_float_array(X, "X")
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    elif Y.ndim != 2:
        raise ValueError("Y must be a vector or a 2D matrix.")

    ensure_no_missing_or_infinite(X, "X")
    ensure_no_missing_or_infinite(Y, "Y")

    m, d = X.shape
    if Y.shape[0] != m:
        raise ValueError("X and Y row counts do not match.")
    if ridge_lambda < 0:
        raise ValueError("Ridge lambda must be non-negative.")

    if ridge_lambda > 0:
        A = X.T @ X + ridge_lambda * np.eye(d)
        B = X.T @ Y
        try:
            W = np.linalg.solve(A, B)
            return W, f"Ridge / regularized normal equation with lambda={ridge_lambda}"
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(A) @ B
            return W, (
                f"Ridge / regularized normal equation with lambda={ridge_lambda} "
                "(pseudoinverse fallback used because the system was ill-conditioned)"
            )

    if m == d:
        try:
            W = np.linalg.solve(X, Y)
            return W, "Even-determined exact solution: w = X^{-1}y"
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(X) @ Y
            return W, (
                "Square but singular system: used pseudoinverse to return the minimum-norm least-squares solution"
            )

    if m > d:
        A = X.T @ X
        B = X.T @ Y
        try:
            W = np.linalg.solve(A, B)
            return W, "Over-determined least-squares solution: w = (X^T X)^{-1}X^T y"
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(X) @ Y
            return W, (
                "Over-determined least-squares solution via pseudoinverse "
                "(X^T X was singular or ill-conditioned)"
            )

    A = X @ X.T
    try:
        alpha = np.linalg.solve(A, Y)
        W = X.T @ alpha
        return W, "Under-determined least-norm solution: w = X^T(XX^T)^{-1}y"
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(X) @ Y
        return W, (
            "Under-determined minimum-norm solution via pseudoinverse "
            "(XX^T was singular or ill-conditioned)"
        )


# ------------------------------
# Core math helpers
# ------------------------------

def one_hot_encode(labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Preserve the first-seen class order instead of sorting.

    This is safer for exam work because the one-hot columns stay aligned with
    the order the user entered, which matches the way many lecture examples
    present the labels.
    """
    classes = list(dict.fromkeys(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(labels), len(classes)), dtype=float)
    for i, label in enumerate(labels):
        Y[i, class_to_idx[label]] = 1.0
    return Y, classes


def linear_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = ensure_2d_float_array(X, "X")
    ensure_no_all_nan_columns(X, "Linear scaling")
    mins = np.nanmin(X, axis=0)
    maxs = np.nanmax(X, axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    return (X - mins) / denom, mins, maxs


def zscore_standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = ensure_2d_float_array(X, "X")
    ensure_no_all_nan_columns(X, "Z-score standardization")
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds, means, stds


def mean_impute(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_filled = ensure_2d_float_array(X, "X").copy().astype(float)
    ensure_no_all_nan_columns(X_filled, "Mean imputation")
    means = np.nanmean(X_filled, axis=0)
    inds = np.where(np.isnan(X_filled))
    X_filled[inds] = np.take(means, inds[1])
    return X_filled, means


def quartiles(values: List[float], method: str = "median_of_halves") -> Tuple[float, float, float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("At least one value is required.")

    if method == "median_of_halves":
        med = statistics.median(sorted_vals)
        if n % 2 == 0:
            lower = sorted_vals[: n // 2]
            upper = sorted_vals[n // 2 :]
        else:
            lower = sorted_vals[: n // 2]
            upper = sorted_vals[n // 2 + 1 :]
        q1 = statistics.median(lower) if lower else sorted_vals[0]
        q3 = statistics.median(upper) if upper else sorted_vals[-1]
        return q1, med, q3

    if method == "percentile_linear":
        arr = np.asarray(sorted_vals, dtype=float)
        q1, med, q3 = np.percentile(arr, [25, 50, 75], method="linear")
        return float(q1), float(med), float(q3)

    raise ValueError(f"Unknown quartile method: {method}")


def descriptive_stats(values: List[float]) -> dict:
    counts = Counter(values)
    max_freq = max(counts.values())
    modes = sorted([k for k, v in counts.items() if v == max_freq])

    q1_halves, med_halves, q3_halves = quartiles(values, method="median_of_halves")
    q1_pct, med_pct, q3_pct = quartiles(values, method="percentile_linear")

    arr = np.asarray(values, dtype=float)
    sample_std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")

    return {
        "count": len(values),
        "mean": float(np.mean(arr)),
        "median": med_halves,
        "mode": modes,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std_population": float(np.std(arr, ddof=0)),
        "std_sample": sample_std,
        "q1": q1_halves,
        "q3": q3_halves,
        "iqr": q3_halves - q1_halves,
        "quartiles_median_of_halves": (q1_halves, med_halves, q3_halves),
        "quartiles_percentile_linear": (q1_pct, med_pct, q3_pct),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    x = x.reshape(-1)
    y = y.reshape(-1)
    if len(x) < 2:
        raise ValueError("Need at least 2 data points.")
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = math.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        raise ValueError("Correlation is undefined when variance is zero.")
    return float(num / den)


def polynomial_feature_names(n_features: int, degree: int) -> List[str]:
    names = ["1"]
    for deg in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), deg):
            counts = Counter(comb)
            term = []
            for idx in sorted(counts):
                power = counts[idx]
                if power == 1:
                    term.append(f"x{idx+1}")
                else:
                    term.append(f"x{idx+1}^{power}")
            names.append("*".join(term))
    return names


def make_polynomial_features(X: np.ndarray, degree: int) -> Tuple[np.ndarray, List[str]]:
    if degree < 1:
        raise ValueError("Degree must be at least 1.")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    n_samples, n_features = X.shape

    cols = [np.ones(n_samples)]
    names = ["1"]
    for deg in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), deg):
            col = np.ones(n_samples)
            counts = Counter(comb)
            term_name_parts = []
            for idx in sorted(counts):
                power = counts[idx]
                col *= X[:, idx] ** power
                term_name_parts.append(f"x{idx+1}" if power == 1 else f"x{idx+1}^{power}")
            cols.append(col)
            names.append("*".join(term_name_parts))
    return np.column_stack(cols), names


def mse(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    return float(np.mean((Y_pred - Y_true) ** 2))


def solve_even_over_under(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 0.0) -> Tuple[str, np.ndarray]:
    w, description = solve_least_squares(X, y, ridge_lambda=ridge_lambda)
    return description, w


def add_bias_column(X: np.ndarray) -> np.ndarray:
    X = ensure_2d_float_array(X, "X")
    return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])


def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    if std <= 0:
        raise ValueError("Standard deviation must be > 0.")
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def count_full_polynomial_terms(n_features: int, degree: int) -> int:
    if n_features < 1:
        raise ValueError("Number of features must be at least 1.")
    if degree < 0:
        raise ValueError("Degree must be at least 0.")
    return sum(math.comb(n_features + k - 1, k) for k in range(degree + 1))


def determination_status(n_samples: int, n_parameters: int) -> str:
    if n_samples < n_parameters:
        return "Under-determined"
    if n_samples == n_parameters:
        return "Even-determined"
    return "Over-determined"


def polynomial_model_summary(X_raw: np.ndarray, degree: int, n_outputs: int = 1) -> dict:
    X_raw = ensure_2d_float_array(X_raw, "X_raw")
    n_samples, n_features = X_raw.shape
    params_per_output = count_full_polynomial_terms(n_features, degree)
    total_params = params_per_output * n_outputs
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "degree": int(degree),
        "params_per_output": int(params_per_output),
        "n_outputs": int(n_outputs),
        "total_params": int(total_params),
        "determination": determination_status(n_samples, params_per_output),
    }








def parse_float_list(text: str) -> List[float]:
    text = clean_text(text)
    if not text:
        return []
    tokens = [tok for tok in text.replace(';', ' ').replace(',', ' ').split() if tok]
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:
        raise ValueError("Could not parse one of the numeric values.") from exc


def stable_sigmoid(z: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(np.clip(-beta * z, -700.0, 700.0)))


def stable_exp_neg(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.exp(np.clip(-z, -700.0, 700.0))


def scalar_power_gradient_descent(power: int, x0: float, eta: float, num_steps: int) -> dict:
    if power < 1:
        raise ValueError("Power must be at least 1.")
    if num_steps < 0:
        raise ValueError("Number of steps must be non-negative.")
    if eta < 0:
        raise ValueError("Learning rate must be non-negative.")

    def grad(x: float) -> float:
        return power * (x ** (power - 1))

    def objective(x: float) -> float:
        return x ** power

    xs = [float(x0)]
    grads = [float(grad(x0))]
    costs = [float(objective(x0))]
    x = float(x0)
    for _ in range(num_steps):
        g = float(grad(x))
        x = float(x - eta * g)
        xs.append(x)
        grads.append(float(grad(x)))
        costs.append(float(objective(x)))
    return {
        "power": int(power),
        "x_values": np.asarray(xs, dtype=float),
        "grad_values": np.asarray(grads, dtype=float),
        "cost_values": np.asarray(costs, dtype=float),
    }


def gradient_formula_text(model_kind: str) -> str:
    model_kind = model_kind.strip().lower()
    if model_kind == "linear_quartic":
        return (
            "Linear model with quartic loss\n"
            "f(x_i, w) = x_i^T w\n"
            "C(w) = Σ_i (f(x_i, w) - y_i)^4\n"
            "∇_w C(w) = Σ_i 4(f(x_i, w) - y_i)^3 x_i\n"
            "        = Σ_i 4(x_i^T w - y_i)^3 x_i"
        )
    if model_kind == "sigmoid_quartic":
        return (
            "Sigmoid model with quartic loss\n"
            "f(x_i, w) = σ(x_i^T w),  where σ(a) = 1 / (1 + exp(-βa))\n"
            "C(w) = Σ_i (f(x_i, w) - y_i)^4\n"
            "σ'(a) = β σ(a)(1-σ(a))\n"
            "∇_w C(w) = Σ_i 4(f(x_i, w) - y_i)^3 β f(x_i, w)(1-f(x_i, w)) x_i"
        )
    if model_kind == "relu_squared_quartic":
        return (
            "Squared-ReLU model with quartic loss\n"
            "f(x_i, w) = max(0, x_i^T w)^2\n"
            "C(w) = Σ_i (f(x_i, w) - y_i)^4\n"
            "For a_i = x_i^T w, d/da max(0, a)^2 = 2 max(0, a) (use subgradient 0 at a=0)\n"
            "∇_w C(w) = Σ_i 8(f(x_i, w) - y_i)^3 max(0, x_i^T w) x_i"
        )
    if model_kind == "exponential_squared":
        return (
            "Exponential model with squared error\n"
            "f(x_i, w) = exp(-x_i^T w)\n"
            "C(w) = Σ_i (f(x_i, w) - y_i)^2\n"
            "∇_w C(w) = - Σ_i 2(f(x_i, w) - y_i) f(x_i, w) x_i"
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def evaluate_model_gradient(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    model_kind: str,
    beta: float = 1.0,
) -> dict:
    X = ensure_2d_float_array(X, "X")
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if y.shape[1] != 1:
        raise ValueError("This gradient evaluator expects a single target column y.")
    if w.shape[1] != 1:
        raise ValueError("This gradient evaluator expects a single weight vector w.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching row counts.")
    if X.shape[1] != w.shape[0]:
        raise ValueError("X columns must match the length of w.")
    ensure_no_missing_or_infinite(X, "X")
    ensure_no_missing_or_infinite(y, "y")
    ensure_no_missing_or_infinite(w, "w")

    z = X @ w
    kind = model_kind.strip().lower()
    if kind == "linear_quartic":
        f = z
        residual = f - y
        cost = float(np.sum(residual ** 4))
        coeff = 4.0 * (residual ** 3)
        grad = X.T @ coeff
    elif kind == "sigmoid_quartic":
        f = stable_sigmoid(z, beta=beta)
        residual = f - y
        cost = float(np.sum(residual ** 4))
        coeff = 4.0 * (residual ** 3) * beta * f * (1.0 - f)
        grad = X.T @ coeff
    elif kind == "relu_squared_quartic":
        relu = np.maximum(0.0, z)
        f = relu ** 2
        residual = f - y
        cost = float(np.sum(residual ** 4))
        coeff = 8.0 * (residual ** 3) * relu
        grad = X.T @ coeff
    elif kind == "exponential_squared":
        f = stable_exp_neg(z)
        residual = f - y
        cost = float(np.sum(residual ** 2))
        coeff = -2.0 * residual * f
        grad = X.T @ coeff
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    return {
        "z": z,
        "f": f,
        "residual": residual,
        "cost": cost,
        "gradient": grad,
    }


def fit_exponential_regression_gd(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    eta: float,
    num_steps: int,
    add_bias: bool = True,
    normalize_x: bool = True,
    normalize_y: bool = True,
    init_w: Optional[np.ndarray] = None,
    record_limit: int = 2500,
    early_stop_tol: Optional[float] = 1e-12,
    early_stop_patience: int = 2000,
) -> dict:
    x_raw = np.asarray(x_raw, dtype=float).reshape(-1)
    y_raw = np.asarray(y_raw, dtype=float).reshape(-1)
    if x_raw.size == 0 or y_raw.size == 0:
        raise ValueError("x and y must not be empty.")
    if x_raw.size != y_raw.size:
        raise ValueError("x and y must have the same number of samples.")
    if eta <= 0:
        raise ValueError("Learning rate eta must be > 0.")
    if num_steps < 0:
        raise ValueError("Number of iterations must be >= 0.")

    x_scale = float(np.max(np.abs(x_raw))) if normalize_x else 1.0
    y_scale = float(np.max(np.abs(y_raw))) if normalize_y else 1.0
    if x_scale == 0:
        x_scale = 1.0
    if y_scale == 0:
        y_scale = 1.0

    x_used = x_raw / x_scale if normalize_x else x_raw.copy()
    y_used = y_raw / y_scale if normalize_y else y_raw.copy()
    X = np.column_stack([np.ones_like(x_used), x_used]) if add_bias else x_used.reshape(-1, 1)

    if init_w is None:
        w = np.zeros(X.shape[1], dtype=float)
    else:
        w = np.asarray(init_w, dtype=float).reshape(-1)
        if w.size != X.shape[1]:
            raise ValueError(f"Initial w must have {X.shape[1]} entries.")

    sample_every = max(1, num_steps // max(1, record_limit)) if num_steps > 0 else 1
    iterations: List[int] = []
    costs: List[float] = []

    def compute_cost_and_grad(current_w: np.ndarray):
        z = X @ current_w
        f = stable_exp_neg(z)
        residual = f - y_used
        cost = float(np.sum(residual ** 2))
        grad = X.T @ (-2.0 * residual * f)
        return cost, grad, f

    cost, grad, f = compute_cost_and_grad(w)
    iterations.append(0)
    costs.append(cost)
    patience_counter = 0
    prev_cost = cost
    diverged = False
    stop_reason = "Reached the requested maximum number of iterations."
    last_step = 0

    for step in range(1, num_steps + 1):
        w = w - eta * grad
        if not np.isfinite(w).all():
            diverged = True
            stop_reason = f"Stopped at iteration {step} because the weights became non-finite."
            break

        cost, grad, f = compute_cost_and_grad(w)
        last_step = step
        if (step % sample_every == 0) or (step == num_steps):
            iterations.append(step)
            costs.append(cost)

        if not np.isfinite(cost) or not np.isfinite(grad).all():
            diverged = True
            stop_reason = f"Stopped at iteration {step} because the cost or gradient became non-finite."
            break

        if early_stop_tol is not None:
            if abs(prev_cost - cost) <= early_stop_tol * max(1.0, abs(prev_cost)):
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= early_stop_patience:
                stop_reason = (
                    f"Stopped early at iteration {step} because the cost change stayed below the tolerance "
                    f"for {early_stop_patience} consecutive iterations."
                )
                break
        prev_cost = cost
    else:
        last_step = num_steps

    if iterations[-1] != last_step:
        iterations.append(last_step)
        costs.append(float(cost))

    yhat_norm = f
    yhat_raw = yhat_norm * y_scale if normalize_y else yhat_norm.copy()
    residual_raw = yhat_raw - y_raw

    return {
        "x_raw": x_raw,
        "y_raw": y_raw,
        "x_used": x_used,
        "y_used": y_used,
        "X_used": X,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "w": w.reshape(-1, 1),
        "eta": float(eta),
        "num_steps_requested": int(num_steps),
        "num_steps_run": int(last_step),
        "iterations": np.asarray(iterations, dtype=int),
        "cost_history": np.asarray(costs, dtype=float),
        "yhat_norm": yhat_norm.reshape(-1, 1),
        "yhat_raw": yhat_raw.reshape(-1, 1),
        "residual_raw": residual_raw.reshape(-1, 1),
        "training_sse_raw": float(np.sum(residual_raw ** 2)),
        "training_mse_raw": float(np.mean(residual_raw ** 2)),
        "diverged": diverged,
        "stop_reason": stop_reason,
        "normalize_x": bool(normalize_x),
        "normalize_y": bool(normalize_y),
        "add_bias": bool(add_bias),
    }


def predict_exponential_regression(result: dict, x_query: np.ndarray) -> dict:
    x_query = np.asarray(x_query, dtype=float).reshape(-1)
    if x_query.size == 0:
        raise ValueError("x_query must not be empty.")
    x_scale = float(result["x_scale"])
    y_scale = float(result["y_scale"])
    normalize_x = bool(result["normalize_x"])
    normalize_y = bool(result["normalize_y"])
    add_bias = bool(result["add_bias"])
    w = np.asarray(result["w"], dtype=float).reshape(-1)

    x_used = x_query / x_scale if normalize_x else x_query.copy()
    Xq = np.column_stack([np.ones_like(x_used), x_used]) if add_bias else x_used.reshape(-1, 1)
    y_norm = stable_exp_neg(Xq @ w)
    y_raw = y_norm * y_scale if normalize_y else y_norm.copy()
    return {
        "x_query": x_query.reshape(-1, 1),
        "X_query_used": Xq,
        "yhat_norm": y_norm.reshape(-1, 1),
        "yhat_raw": y_raw.reshape(-1, 1),
    }

def describe_array(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return f"{name}: scalar value = {float(arr)}"

    lines = [
        f"{name} shape = {tuple(arr.shape)}",
        f"{name} total entries / values = {int(arr.size)}",
    ]
    if arr.ndim == 1:
        lines.append(f"{name} length = {arr.size}")
    elif arr.ndim == 2:
        lines.append(f"{name} rows = {arr.shape[0]}, cols = {arr.shape[1]}")
    return "\n".join(lines)


def describe_design_matrix(name: str, X: np.ndarray) -> str:
    X = np.asarray(X)
    if X.ndim != 2:
        return describe_array(name, X)
    m, d = X.shape
    lines = [
        describe_array(name, X),
        f"If {name} is the design matrix: samples / equations = {m}, coefficient columns / parameters per output = {d}",
    ]
    return "\n".join(lines)


def describe_target_matrix(name: str, Y: np.ndarray) -> str:
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        return describe_array(name, Y)
    m, c = Y.shape
    lines = [describe_array(name, Y)]
    if c == 1:
        lines.append(f"{name} contains {m} scalar target value(s).")
    else:
        lines.append(f"{name} contains {c} output / class column(s) across {m} sample(s).")
        lines.append(f"Total target values in {name} = {m * c}")
    return "\n".join(lines)


def describe_weight_matrix(name: str, W: np.ndarray) -> str:
    W = np.asarray(W)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if W.ndim != 2:
        return describe_array(name, W)
    d, c = W.shape
    lines = [describe_array(name, W)]
    if c == 1:
        lines.append(f"Unknown parameter count in {name} = {d}")
    else:
        lines.append(f"Parameters per output / class column in {name} = {d}")
        lines.append(f"Output / class columns in {name} = {c}")
        lines.append(f"Total unknown parameters in {name} = {d * c}")
    return "\n".join(lines)


def section_block(title: str, *sections: str) -> str:
    non_empty = [section for section in sections if section]
    if not non_empty:
        return title
    return "\n".join([title] + [""] + non_empty)

def parse_named_counts(text: str) -> List[Tuple[str, int]]:
    """
    Parse lines like:
        red, 5
        blue, 3
    or:
        red:5; blue:3

    Returns a list of (label, count) preserving order.
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Count input is empty.")

    pairs: List[Tuple[str, int]] = []
    for raw_line in text.replace(';', '\n').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ':' in line:
            label, value = line.split(':', 1)
        elif ',' in line:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 2:
                raise ValueError(f"Each count line must have exactly 2 fields: label and count. Bad line: {raw_line}")
            label, value = parts
        else:
            raise ValueError(f"Could not parse count line: {raw_line}")

        label = label.strip()
        if not label:
            raise ValueError("Category label cannot be blank.")
        try:
            numeric = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Could not parse count value in line: {raw_line}") from exc
        if numeric < 0 or abs(numeric - round(numeric)) > 1e-12:
            raise ValueError(f"Counts must be non-negative integers. Bad line: {raw_line}")
        pairs.append((label, int(round(numeric))))

    if not pairs:
        raise ValueError("No valid count lines found.")

    return pairs


def parse_group_success_records(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Parse rows like:
        Department A, Team A-1, 200, 150
    meaning:
        parent_group, subgroup, total_count, success_count
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Record input is empty.")

    records: List[Tuple[str, str, int, int]] = []
    for raw_line in text.replace(';', '\n').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 4:
            raise ValueError(
                "Each record must have exactly 4 comma-separated fields: "
                "Parent group, Subgroup, Total count, Success count. "
                f"Bad line: {raw_line}"
            )
        parent, subgroup, total_text, success_text = parts
        if not parent or not subgroup:
            raise ValueError(f"Parent group and subgroup labels cannot be blank. Bad line: {raw_line}")
        try:
            total = float(total_text)
            success = float(success_text)
        except ValueError as exc:
            raise ValueError(f"Could not parse total / success count in line: {raw_line}") from exc
        if total < 0 or success < 0:
            raise ValueError(f"Counts must be non-negative. Bad line: {raw_line}")
        if abs(total - round(total)) > 1e-12 or abs(success - round(success)) > 1e-12:
            raise ValueError(f"Counts must be integers. Bad line: {raw_line}")
        total = int(round(total))
        success = int(round(success))
        if success > total:
            raise ValueError(f"Success count cannot exceed total count. Bad line: {raw_line}")
        records.append((parent, subgroup, total, success))

    if not records:
        raise ValueError("No valid records found.")

    return records


def parse_probability_tokens(text: str) -> List[str]:
    tokens = [tok.strip() for tok in text.replace(';', ' ').replace(',', ' ').split() if tok.strip()]
    if not tokens:
        raise ValueError("Probability input is empty.")
    return tokens


def solve_pmf_unknowns(
    x_vals: List[float],
    prob_tokens: List[str],
    expected_value: Optional[float] = None,
) -> Tuple[List[float], List[int]]:
    """
    Supports 0, 1, or 2 unknown probabilities marked as k or ?.

    Equations used:
    1) sum p_i = 1
    2) sum x_i p_i = E[X], if expected_value is provided
    """
    if len(x_vals) != len(prob_tokens):
        raise ValueError("Number of x-values must match number of probability entries.")
    if len(set(x_vals)) != len(x_vals):
        raise ValueError("x values in a PMF must be unique.")

    unknown_idx: List[int] = []
    probs: List[Optional[float]] = []
    for i, tok in enumerate(prob_tokens):
        if tok.lower() in {"k", "?"}:
            unknown_idx.append(i)
            probs.append(None)
        else:
            value = float(tok)
            validate_probability(value, f"p(x) at index {i + 1}")
            probs.append(value)

    if len(unknown_idx) > 2:
        raise ValueError("At most two unknown probabilities are supported.")

    known_sum = sum(p for p in probs if p is not None)
    known_exp = sum(x_vals[i] * probs[i] for i in range(len(x_vals)) if probs[i] is not None)

    if len(unknown_idx) == 0:
        total = known_sum
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Probabilities sum to {total}, not 1.")
        solved = [float(p) for p in probs]  # type: ignore[arg-type]
        if expected_value is not None:
            mean = sum(x * p for x, p in zip(x_vals, solved))
            if abs(mean - expected_value) > 1e-8:
                raise ValueError(f"The PMF gives E[X] = {mean}, which does not match the provided expected value {expected_value}.")
        return solved, unknown_idx

    if len(unknown_idx) == 1:
        idx = unknown_idx[0]
        value = 1.0 - known_sum
        validate_probability(value, "Solved unknown probability")
        probs[idx] = value
        solved = [float(p) for p in probs]  # type: ignore[arg-type]
        if expected_value is not None:
            mean = sum(x * p for x, p in zip(x_vals, solved))
            if abs(mean - expected_value) > 1e-8:
                raise ValueError(
                    f"The sum rule gives the unknown as {value}, but this leads to E[X] = {mean}, "
                    f"which does not match the provided expected value {expected_value}."
                )
        return solved, unknown_idx

    if expected_value is None:
        raise ValueError("Two unknown probabilities require the expected value E[X] to also be provided.")

    i, j = unknown_idx
    A = np.array([
        [1.0, 1.0],
        [float(x_vals[i]), float(x_vals[j])],
    ], dtype=float)
    b = np.array([
        1.0 - known_sum,
        expected_value - known_exp,
    ], dtype=float)

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not uniquely solve the two unknown PMF probabilities.") from exc

    for value in sol:
        validate_probability(float(value), "Solved unknown probability")

    probs[i] = float(sol[0])
    probs[j] = float(sol[1])
    solved = [float(p) for p in probs]  # type: ignore[arg-type]
    total = sum(solved)
    mean = sum(x * p for x, p in zip(x_vals, solved))
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Solved probabilities sum to {total}, not 1.")
    if abs(mean - expected_value) > 1e-8:
        raise ValueError(f"Solved probabilities give E[X] = {mean}, not the required {expected_value}.")
    return solved, unknown_idx


def pmf_mean_variance(x_vals: List[float], probs: List[float]) -> Tuple[float, float]:
    mean = float(sum(x * p for x, p in zip(x_vals, probs)))
    var = float(sum(((x - mean) ** 2) * p for x, p in zip(x_vals, probs)))
    return mean, var


def exact_sequence_probability(counts: List[Tuple[str, int]], sequence: List[str], with_replacement: bool) -> Tuple[float, List[str]]:
    count_map = {label: count for label, count in counts}
    total_initial = sum(count_map.values())
    steps: List[str] = []
    prob = 1.0

    if total_initial <= 0:
        raise ValueError("Total population size must be positive.")

    if with_replacement:
        for draw_idx, label in enumerate(sequence, start=1):
            if label not in count_map:
                raise ValueError(f"Unknown category in draw sequence: {label}")
            draw_prob = count_map[label] / total_initial
            prob *= draw_prob
            steps.append(
                f"Draw {draw_idx}: P({label}) = {count_map[label]}/{total_initial} = {draw_prob:.6f}"
            )
        return prob, steps

    remaining = dict(count_map)
    for draw_idx, label in enumerate(sequence, start=1):
        if label not in remaining:
            raise ValueError(f"Unknown category in draw sequence: {label}")
        total_now = sum(remaining.values())
        available = remaining[label]
        if total_now <= 0 or available <= 0:
            steps.append(f"Draw {draw_idx}: category {label} unavailable, so probability becomes 0.")
            return 0.0, steps
        draw_prob = available / total_now
        prob *= draw_prob
        steps.append(
            f"Draw {draw_idx}: P({label}) = {available}/{total_now} = {draw_prob:.6f}"
        )
        remaining[label] -= 1
    return prob, steps


def unordered_sequence_probability(counts: List[Tuple[str, int]], sequence: List[str], with_replacement: bool) -> Tuple[float, List[Tuple[Tuple[str, ...], float]]]:
    unique_orders = sorted(set(permutations(sequence)))
    details: List[Tuple[Tuple[str, ...], float]] = []
    total = 0.0
    for order in unique_orders:
        p, _ = exact_sequence_probability(counts, list(order), with_replacement)
        details.append((order, p))
        total += p
    return total, details


def inverse_status_summary(X: np.ndarray) -> dict:
    X = ensure_2d_float_array(X, "X")
    ensure_no_missing_or_infinite(X, "X")
    m, d = X.shape
    r = int(np.linalg.matrix_rank(X))

    has_left = (m >= d and r == d)
    has_right = (m <= d and r == m)
    has_inverse = (m == d and r == m)

    summary = {
        "shape": (m, d),
        "rank": r,
        "has_left_inverse": has_left,
        "has_right_inverse": has_right,
        "has_inverse": has_inverse,
        "left_inverse_matrix": None,
        "right_inverse_matrix": None,
        "inverse_matrix": None,
    }

    if has_left:
        summary["left_inverse_matrix"] = np.linalg.inv(X.T @ X) @ X.T
    if has_right:
        summary["right_inverse_matrix"] = X.T @ np.linalg.inv(X @ X.T)
    if has_inverse:
        summary["inverse_matrix"] = np.linalg.inv(X)
    return summary


def derivative_shape_summary(
    output_kind: str,
    input_kind: str,
    output_dim: Optional[int] = None,
    input_dim: Optional[int] = None,
) -> Tuple[str, str]:
    output_kind = output_kind.strip().lower()
    input_kind = input_kind.strip().lower()

    if output_kind not in {"scalar", "vector"}:
        raise ValueError("Output kind must be 'scalar' or 'vector'.")
    if input_kind not in {"scalar", "vector"}:
        raise ValueError("Input kind must be 'scalar' or 'vector'.")

    if output_kind == "scalar" and input_kind == "scalar":
        return "scalar", "A scalar differentiated with respect to a scalar is a scalar."

    if output_kind == "scalar" and input_kind == "vector":
        if input_dim is None or input_dim < 1:
            raise ValueError("Input vector dimension must be at least 1.")
        return f"{input_dim} x 1 vector", (
            "A scalar with respect to a vector gives a gradient with one partial derivative per input variable."
        )

    if output_kind == "vector" and input_kind == "scalar":
        if output_dim is None or output_dim < 1:
            raise ValueError("Output vector dimension must be at least 1.")
        return f"{output_dim} x 1 vector", (
            "A vector with respect to a scalar gives a vector of ordinary derivatives."
        )

    if output_dim is None or output_dim < 1:
        raise ValueError("Output vector dimension must be at least 1.")
    if input_dim is None or input_dim < 1:
        raise ValueError("Input vector dimension must be at least 1.")
    return f"{output_dim} x {input_dim} matrix", (
        "A vector with respect to a vector gives the Jacobian matrix: one row per output component and one column per input variable."
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def binary_confusion_summary(y_true: List[str], y_pred: List[str], positive_label: str) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one prediction is required.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp == positive_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp != positive_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp == positive_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp != positive_label)

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    accuracy = safe_ratio(tp + tn, len(y_true))
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "positive_label": positive_label,
        "negative_count": tn + fp,
        "positive_count": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (
            float("nan")
            if math.isnan(recall) or math.isnan(specificity)
            else 0.5 * (recall + specificity)
        ),
    }


def multiclass_confusion_summary(
    y_true: List[str],
    y_pred: List[str],
    class_order: Optional[List[str]] = None,
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("At least one prediction is required.")

    if class_order is None:
        class_order = sorted(set(y_true) | set(y_pred), key=mixed_label_sort_key)
    if not class_order:
        raise ValueError("No classes found.")

    index = {label: idx for idx, label in enumerate(class_order)}
    matrix = np.zeros((len(class_order), len(class_order)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt not in index:
            raise ValueError(f"True label '{yt}' is not present in class_order.")
        if yp not in index:
            raise ValueError(f"Predicted label '{yp}' is not present in class_order.")
        matrix[index[yt], index[yp]] += 1

    per_class = []
    for idx, label in enumerate(class_order):
        tp = int(matrix[idx, idx])
        fn = int(np.sum(matrix[idx, :]) - tp)
        fp = int(np.sum(matrix[:, idx]) - tp)
        tn = int(np.sum(matrix) - tp - fn - fp)
        per_class.append({
            "label": label,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "precision": safe_ratio(tp, tp + fp),
            "recall": safe_ratio(tp, tp + fn),
        })

    return {
        "classes": class_order,
        "matrix": matrix,
        "accuracy": safe_ratio(np.trace(matrix), np.sum(matrix)),
        "per_class": per_class,
    }


def count_cross_validation_fits(n_candidates: int, n_folds: int) -> int:
    if n_candidates < 1 or n_folds < 2:
        raise ValueError("Use at least 1 candidate and at least 2 folds.")
    return n_candidates * n_folds


def choose_best_validation_candidate(
    rows: List[Tuple[float, float, float]],
    lower_is_better: bool = True,
    prefer_smaller_parameter: bool = True,
) -> dict:
    if not rows:
        raise ValueError("At least one candidate row is required.")

    def sort_key(item: Tuple[float, float, float]):
        param, train_metric, val_metric = item
        primary = val_metric if lower_is_better else -val_metric
        secondary = param if prefer_smaller_parameter else -param
        tertiary = train_metric if lower_is_better else -train_metric
        return (primary, secondary, tertiary)

    best = min(rows, key=sort_key)
    return {
        "best_parameter": best[0],
        "best_training_metric": best[1],
        "best_validation_metric": best[2],
        "all_rows": list(rows),
    }


def class_impurity_summary(class_counts: List[float]) -> dict:
    probs = np.asarray(class_counts, dtype=float).reshape(-1)
    if probs.size == 0:
        raise ValueError("At least one class count is required.")
    if np.any(probs < 0):
        raise ValueError("Class counts must be non-negative.")
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("The total class count must be positive.")

    p = probs / total
    nonzero = p[p > 0]
    gini = float(1.0 - np.sum(p ** 2))
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    misclassification = float(1.0 - np.max(p))
    return {
        "counts": probs,
        "total": total,
        "probabilities": p,
        "gini": gini,
        "entropy": entropy,
        "misclassification": misclassification,
    }


def weighted_child_impurity_summaries(child_counts: List[List[float]]) -> dict:
    if not child_counts:
        raise ValueError("At least one child node is required.")
    child_summaries = [class_impurity_summary(counts) for counts in child_counts]
    total = sum(summary["total"] for summary in child_summaries)
    if total <= 0:
        raise ValueError("The total number of samples across child nodes must be positive.")

    return {
        "children": child_summaries,
        "total": total,
        "weighted_gini": sum((summary["total"] / total) * summary["gini"] for summary in child_summaries),
        "weighted_entropy": sum((summary["total"] / total) * summary["entropy"] for summary in child_summaries),
        "weighted_misclassification": sum(
            (summary["total"] / total) * summary["misclassification"] for summary in child_summaries
        ),
    }


def regression_tree_split_summary(x_values: np.ndarray, y_values: np.ndarray, threshold: float) -> dict:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    ensure_no_missing_or_infinite(x, "x")
    ensure_no_missing_or_infinite(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")
    if x.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    left_mask = x <= threshold
    right_mask = x > threshold
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError("The threshold must split the data into non-empty left and right groups.")

    left_y = y[left_mask]
    right_y = y[right_mask]
    root_mean = float(np.mean(y))
    left_mean = float(np.mean(left_y))
    right_mean = float(np.mean(right_y))
    root_mse = float(np.mean((y - root_mean) ** 2))
    left_mse = float(np.mean((left_y - left_mean) ** 2))
    right_mse = float(np.mean((right_y - right_mean) ** 2))
    weighted_mse = float((left_y.size * left_mse + right_y.size * right_mse) / y.size)

    return {
        "threshold": float(threshold),
        "root_mean": root_mean,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "root_mse": root_mse,
        "left_mse": left_mse,
        "right_mse": right_mse,
        "weighted_mse": weighted_mse,
        "left_count": int(left_y.size),
        "right_count": int(right_y.size),
        "improvement": float(root_mse - weighted_mse),
    }


def kmeans_objective(X: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
    X = ensure_2d_float_array(X, "X")
    centroids = ensure_2d_float_array(centroids, "centroids")
    assignments = np.asarray(assignments, dtype=int).reshape(-1)
    if X.shape[0] != assignments.shape[0]:
        raise ValueError("Assignments must have one entry per sample.")
    sq_dist = np.sum((X - centroids[assignments]) ** 2, axis=1)
    return float(np.sum(sq_dist))


def run_kmeans(X: np.ndarray, initial_centroids: np.ndarray, max_iter: int = 100) -> dict:
    X = ensure_2d_float_array(X, "X")
    centroids = ensure_2d_float_array(initial_centroids, "Initial centroids")
    ensure_no_missing_or_infinite(X, "X")
    ensure_no_missing_or_infinite(centroids, "Initial centroids")
    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and the initial centroids must have the same number of feature columns.")
    if centroids.shape[0] < 1:
        raise ValueError("At least one centroid is required.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    history = []
    current = centroids.copy()
    for iteration in range(1, max_iter + 1):
        sq_dist = np.sum((X[:, None, :] - current[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(sq_dist, axis=1)
        updated = current.copy()
        empty_clusters = []
        for idx in range(current.shape[0]):
            members = X[assignments == idx]
            if members.size == 0:
                empty_clusters.append(idx)
            else:
                updated[idx] = np.mean(members, axis=0)

        history.append({
            "iteration": iteration,
            "centroids_before": current.copy(),
            "assignments": assignments.copy(),
            "centroids_after": updated.copy(),
            "objective": kmeans_objective(X, assignments, updated),
            "empty_clusters": empty_clusters,
        })

        if np.allclose(updated, current):
            return {
                "converged": True,
                "iterations_run": iteration,
                "final_centroids": updated,
                "final_assignments": assignments,
                "history": history,
            }
        current = updated

    final_sq_dist = np.sum((X[:, None, :] - current[None, :, :]) ** 2, axis=2)
    final_assignments = np.argmin(final_sq_dist, axis=1)
    return {
        "converged": False,
        "iterations_run": max_iter,
        "final_centroids": current,
        "final_assignments": final_assignments,
        "history": history,
    }


def best_clustering_accuracy(true_labels: List[str], cluster_assignments: np.ndarray) -> dict:
    if len(true_labels) != len(cluster_assignments):
        raise ValueError("The number of labels must match the number of assignments.")
    if not true_labels:
        raise ValueError("At least one labelled sample is required.")

    assignments = [int(value) for value in np.asarray(cluster_assignments, dtype=int).reshape(-1)]
    clusters = sorted(set(assignments))
    labels = sorted(set(true_labels), key=mixed_label_sort_key)
    if len(clusters) != len(labels):
        raise ValueError("Permutation-based clustering accuracy requires the same number of clusters and labels.")

    best_accuracy = -1.0
    best_mapping = None
    for perm in permutations(labels):
        mapping = {cluster: perm[idx] for idx, cluster in enumerate(clusters)}
        preds = [mapping[cluster] for cluster in assignments]
        accuracy = sum(int(pred == truth) for pred, truth in zip(preds, true_labels)) / len(true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = mapping

    return {
        "accuracy": best_accuracy,
        "mapping": best_mapping,
        "predicted_labels": [best_mapping[cluster] for cluster in assignments],
    }


def apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    activation = activation.strip().lower()
    arr = np.asarray(values, dtype=float)
    if activation == "linear":
        return arr
    if activation == "relu":
        return np.maximum(0.0, arr)
    if activation == "sigmoid":
        return stable_sigmoid(arr)
    raise ValueError("Activation must be one of: linear, relu, sigmoid.")


def forward_neural_network(
    X: np.ndarray,
    weight_matrices: List[np.ndarray],
    hidden_activation: str = "relu",
    output_activation: str = "linear",
    add_bias: bool = True,
) -> dict:
    current = ensure_2d_float_array(X, "X")
    ensure_no_missing_or_infinite(current, "X")
    if not weight_matrices:
        raise ValueError("At least one weight matrix is required.")

    layers = []
    total_parameters = 0
    for idx, W in enumerate(weight_matrices, start=1):
        W = ensure_2d_float_array(W, f"W{idx}")
        ensure_no_missing_or_infinite(W, f"W{idx}")
        layer_input = add_bias_column(current) if add_bias else current
        expected_rows = layer_input.shape[1]
        if W.shape[0] != expected_rows:
            raise ValueError(
                f"W{idx} has shape {W.shape}, but layer {idx} expects {expected_rows} rows "
                f"({'with' if add_bias else 'without'} bias)."
            )
        Z = layer_input @ W
        activation = output_activation if idx == len(weight_matrices) else hidden_activation
        A = apply_activation(Z, activation)
        total_parameters += int(W.size)
        layers.append({
            "layer_index": idx,
            "input_with_bias": layer_input,
            "weights": W,
            "pre_activation": Z,
            "activation_name": activation,
            "output": A,
            "parameter_count": int(W.size),
        })
        current = A

    return {
        "input": X,
        "layers": layers,
        "final_output": current,
        "total_parameters": total_parameters,
        "add_bias": add_bias,
    }


TUTORIAL_CONCEPTS = {
    "ML vs AI": (
        "Artificial Intelligence (AI) is the broader goal of making machines act intelligently.\n"
        "Machine Learning (ML) is a subset of AI where the system improves from data or experience instead of being fully hand-programmed."
    ),
    "Supervised vs Unsupervised": (
        "Supervised learning uses labelled targets y and learns a mapping X -> y.\n"
        "Unsupervised learning has no target labels and looks for structure such as clusters, latent factors, or lower-dimensional representations."
    ),
    "Classification vs Regression": (
        "Classification predicts categories or class labels, such as spam / not spam.\n"
        "Regression predicts numerical values, such as temperature, stock volume, or number of books sold."
    ),
    "Task / Performance / Experience": (
        "Task T = what the system must do.\n"
        "Performance P = how success is measured.\n"
        "Experience E = the data or feedback used for learning.\n\n"
        "Example: weather prediction ->\n"
        "T: predict tomorrow's weather,\n"
        "P: prediction accuracy / MSE,\n"
        "E: historical weather data."
    ),
    "Inductive vs Deductive": (
        "Inductive reasoning goes from specific observations to a more general rule.\n"
        "Deductive reasoning starts from a general rule and applies it to a specific case."
    ),
    "NOIR data types": (
        "Nominal: categories with no order.\n"
        "Ordinal: categories with order but unequal gaps.\n"
        "Interval: ordered, equal gaps, but no true zero.\n"
        "Ratio: interval data with a meaningful zero."
    ),
    "Boxplot blanks": (
        "A boxplot uses the five-number summary: minimum, Q1, median, Q3, maximum.\n"
        "The proportion between Q1 and Q3 is 50% of the data."
    ),
    "Valid probability assignment": (
        "For a valid PMF / event probability assignment, every probability must satisfy 0 <= p <= 1,\n"
        "and the probabilities over all mutually exclusive outcomes must sum to 1."
    ),
    "Correlation vs causation": (
        "Correlation means two variables move together.\n"
        "Causation means one variable directly influences the other.\n"
        "Correlation alone does not prove causation; a hidden variable may explain both."
    ),
    "Affine vs linear": (
        "A linear map passes through the origin and has no constant offset.\n"
        "An affine function is linear plus a bias / offset term, for example f(x) = w^T x + b.\n"
        "In ML, many 'linear regression' models are actually affine because they include a bias."
    ),
    "Derivative with respect to a d-vector": (
        "For a scalar function f(x) where x is d x 1, the derivative / gradient with respect to x is a d x 1 vector."
    ),
    "Polynomial model size": (
        "A full polynomial model of degree p in d variables has\n"
        "sum_{k=0}^{p} C(d+k-1, k) = C(d+p, p) parameters when the constant term is included."
    ),
    "Ridge regression effect": (
        "Ridge regression adds lambda * ||w||^2 to penalize large weights.\n"
        "As lambda increases, the weight norm usually shrinks, which can reduce overfitting and improve test performance."
    ),
}


TUTORIAL_CONCEPTS.update({
    "Convexity vs unique solution": (
        "A convex objective guarantees that every local minimum is global, but it does not automatically guarantee uniqueness. "
        "Uniqueness needs strict convexity, which in least squares is tied to the rank / positive-definiteness of X^T X."
    ),
    "Data wrangling vs validation": (
        "Data wrangling means preparing or transforming data into a usable form. "
        "Data validation means checking whether the data is correct, consistent, complete, or within expected rules. "
        "Wrangling is not itself a typical step of validation."
    ),
    "Interpolation vs extrapolation": (
        "Interpolation predicts inside the range of the observed training inputs. "
        "Extrapolation predicts outside the observed training range and is usually riskier."
    ),
    "Discrete vs continuous": (
        "Discrete variables take countable distinct values, such as number of books sold. "
        "Continuous variables can vary smoothly over an interval, such as temperature or speed."
    ),
    "Simpson's paradox": (
        "Simpson's paradox happens when a trend seen in separate groups reverses after the groups are combined. "
        "It is a warning that aggregated data can hide a confounding variable."
    ),
    "Clustering vs reinforcement learning": (
        "Clustering is unsupervised learning that groups similar data points without labels. "
        "Reinforcement learning learns actions through rewards and penalties over time."
    ),
    "Decomposable outcomes / events": (
        "In a PMF or event space, mutually exclusive elementary outcomes can be combined into larger events. "
        "Probabilities of disjoint outcomes add."
    ),
})



TUTORIAL_CONCEPTS.update({
    "Underfitting vs overfitting": (
        "Underfitting means the model is too simple: training error is high and test error is also high.\n"
        "Overfitting means the model is too complex: training error is low, but test error is high because the model also fits noise."
    ),
    "Bias-variance trade-off": (
        "High bias usually comes from overly simple models and leads to underfitting.\n"
        "High variance usually comes from overly complex models and leads to overfitting.\n"
        "The goal is to pick a model complexity that balances bias and variance for low test / generalization error."
    ),
    "Feature selection on training set only": (
        "Feature selection should be done using the training set only.\n"
        "Using the test set during feature selection leaks information from the test data into the model and gives misleading performance estimates."
    ),
    "Regularization motivation": (
        "Regularization adds a penalty such as λ||w||^2 to discourage overly large parameters.\n"
        "It helps stabilize ill-posed problems, reduce overfitting, and improve generalization."
    ),
    "MSE vs SSE": (
        "Minimizing MSE and minimizing SSE give the same optimal weights because MSE = (1/m) * SSE.\n"
        "The factor 1/m only rescales the objective and does not change the minimizer."
    ),
    "Loss vs cost vs objective": (
        "Loss function: error for one training example.\n"
        "Cost function: total or average loss over the dataset.\n"
        "Objective function: what we optimize; it may be the cost alone or cost plus regularization."
    ),
    "Gradient descent update rule": (
        "Gradient descent updates parameters by w_(k+1) = w_k - η∇C(w_k).\n"
        "The gradient points in the direction of steepest increase, so we subtract it to move downhill."
    ),
    "Learning rate too big vs too small": (
        "If η is too big, gradient descent can overshoot the minimum, oscillate, diverge, or converge slowly.\n"
        "If η is too small, convergence is stable but very slow."
    ),
    "Why sigmoid instead of sign": (
        "The sign function gives only hard labels and is not differentiable at 0, so it is inconvenient for gradient-based training.\n"
        "The sigmoid is differentiable and outputs values between 0 and 1, which can be interpreted as probabilities."
    ),
    "Different model functions": (
        "Sign: hard binary label from a score.\n"
        "Sigmoid: smooth probability-like output in (0,1).\n"
        "ReLU: max(0, a), useful as a nonlinear activation.\n"
        "Exponential: exp(-a), used in Tutorial 8 for gradient-descent-based regression."
    ),
})


TUTORIAL_CONCEPTS.update({
    "Linear model in transformed feature space": (
        "A linear model is linear in its parameters, not necessarily a straight line in the original input space.\n"
        "After a feature mapping such as polynomial expansion, the model can still be linear in the new features while becoming curved in the original x-space."
    ),
    "Right inverse vs left inverse": (
        "For X with shape m x d:\n"
        "- Left inverse exists when X has full column rank, so rank(X)=d and typically m >= d.\n"
        "- Right inverse exists when X has full row rank, so rank(X)=m and typically m <= d.\n"
        "A tall 3 x 2 full-column-rank matrix can have a left inverse, but not a right inverse."
    ),
    "Regularizing the bias term": (
        "Some questions explicitly say all parameters, including the bias / constant term, are regularized.\n"
        "That means the ridge penalty uses the full identity matrix, including the row / coefficient for the constant 1 feature."
    ),
    "Parameters across all classes": (
        "If there are C output columns / classes and p parameters per output, then the total number of learned parameters is pC.\n"
        "Example: with 3 features and a bias term, a linear one-hot classifier has 4 parameters per class. For 3 classes, that is 12 total parameters."
    ),
    "Argmax after one-hot regression": (
        "When one-hot targets are fit by linear or polynomial regression, prediction is made by computing the score vector and choosing the class with the largest score.\n"
        "So the predicted class is argmax_j f_j(x), not the class whose score is closest to 0 or 1 individually."
    ),
    "Ridge in under-determined systems": (
        "An under-determined unregularized polynomial model may have infinitely many solutions.\n"
        "Adding ridge regularization with λ > 0 typically makes the system well-posed and selects a unique penalized solution."
    ),
})


TUTORIAL_CONCEPTS.update({
    "K-means clustering algorithm": (
        "K-means is an unsupervised algorithm that partitions N data points into K clusters.\n\n"
        "Algorithm:\n"
        "1. Initialize K centroids (e.g., pick first K data points, or randomly).\n"
        "2. Assignment step: assign each point to its nearest centroid (Euclidean distance).\n"
        "3. Update step: recompute each centroid as the mean of all points assigned to it.\n"
        "4. Repeat steps 2-3 until centroids stop moving (convergence) or max iterations reached.\n\n"
        "Objective (WCSS - Within-Cluster Sum of Squares):\n"
        "  min Σ_k Σ_{x in C_k} ||x - μ_k||²\n"
        "where μ_k is the centroid of cluster k.\n\n"
        "K-means minimizes WCSS at each step. It is not guaranteed to find the global minimum."
    ),
    "Elbow method for choosing K": (
        "Plot WCSS (within-cluster sum of squares) vs K.\n"
        "As K increases, WCSS decreases. The 'elbow' point where improvement slows down is a good choice for K.\n"
        "There is no unique elbow in all cases, so this is a heuristic guideline."
    ),
    "K-means vs supervised learning": (
        "K-means is unsupervised: no labels are needed.\n"
        "It discovers structure in the data by grouping similar points.\n"
        "Supervised learning (regression, classification) uses known labels to train a model."
    ),
    "Decision tree splitting criterion (MSE)": (
        "For regression trees, the best split minimizes the total MSE:\n"
        "  Total MSE = (1/N) * [Σ_{i in left} (y_i - ȳ_left)² + Σ_{i in right} (y_i - ȳ_right)²]\n"
        "where ȳ_left and ȳ_right are the means of the left and right partitions.\n\n"
        "Procedure:\n"
        "1. Sort data by the feature being split on.\n"
        "2. Try every possible split threshold.\n"
        "3. Pick the split that gives the smallest total MSE.\n"
        "4. Prediction for each leaf = mean of y values in that leaf."
    ),
    "Decision tree depth and overfitting": (
        "A shallow tree (depth 1-2) underfits: it cannot capture complex patterns.\n"
        "A deep tree can overfit: it memorizes training data including noise.\n"
        "Hyperparameters like max_depth and min_samples_split control regularization."
    ),
    "Confusion matrix": (
        "A confusion matrix summarizes classification results for C classes.\n"
        "It is a C x C matrix where entry (i, j) = number of true-class-i samples predicted as class j.\n\n"
        "From it we can compute:\n"
        "  Accuracy = (sum of diagonal) / (total samples)\n"
        "  Error rate = 1 - Accuracy\n"
        "  Precision for class c = TP_c / (TP_c + FP_c)  (column sum)\n"
        "  Recall for class c    = TP_c / (TP_c + FN_c)  (row sum)"
    ),
    "Accuracy vs error rate": (
        "Accuracy = (correct predictions) / (total predictions)\n"
        "Error rate = 1 - Accuracy = (wrong predictions) / (total predictions)\n\n"
        "These are standard evaluation metrics for classification."
    ),
    "Regression metrics: MSE, RMSE, MAE, R2": (
        "MSE (Mean Squared Error) = (1/N) Σ (ŷ_i - y_i)²  — penalizes large errors more.\n"
        "RMSE = sqrt(MSE)  — same units as y, easier to interpret.\n"
        "MAE (Mean Absolute Error) = (1/N) Σ |ŷ_i - y_i|  — more robust to outliers.\n"
        "R² (Coefficient of Determination) = 1 - SS_res / SS_tot\n"
        "  where SS_res = Σ(ŷ_i - y_i)², SS_tot = Σ(y_i - ȳ)².\n"
        "R² = 1 means perfect fit; R² = 0 means the model does no better than predicting the mean; R² < 0 is possible for very bad models."
    ),
    "Validation set vs test set": (
        "Training set: used to fit model weights.\n"
        "Validation set: used to tune hyperparameters (e.g., polynomial order, ridge lambda, K in K-means).\n"
        "Test set: held-out final evaluation of generalization ability — must NOT be used to tune.\n\n"
        "Using the test set during tuning leads to optimistically biased estimates of generalization performance."
    ),
    "K-fold cross-validation": (
        "Split training data into K equal folds.\n"
        "For each fold i: train on the other K-1 folds, validate on fold i.\n"
        "Average validation performance across K folds.\n\n"
        "This gives a more reliable estimate of generalization than a single validation split, especially when data is limited."
    ),
    "Feature encoding: nominal vs ordinal warning": (
        "Nominal variables have no natural ordering (e.g., color: red, blue, green).\n"
        "Encoding nominal categories as integers (0, 1, 2, ...) introduces a false ordering.\n"
        "This misleads models that compute differences or distances, such as linear regression or k-means.\n\n"
        "Correct approach for nominal variables: use one-hot encoding.\n"
        "Ordinal variables (e.g., low < medium < high) CAN be safely encoded as integers."
    ),
    "Neural network (brief overview)": (
        "A neural network stacks multiple layers of linear transformations followed by nonlinear activations.\n"
        "Common activations: ReLU (max(0, a)), sigmoid (1/(1+e^{-a})), tanh.\n"
        "Training uses backpropagation (chain rule) and gradient descent.\n"
        "Deep networks can approximate complex functions but need regularization to avoid overfitting."
    ),
    "Random forest brief overview": (
        "A random forest trains many decision trees on bootstrap samples of the data.\n"
        "At each split, a random subset of features is considered.\n"
        "Final prediction = majority vote (classification) or average (regression) across all trees.\n"
        "This reduces overfitting compared to a single deep tree."
    ),
})


MIDTERM_STYLE_GUIDE = {
    "Polynomial parameter count + system type": (
        "Midterm-style answer pattern:\n"
        "1. For a full polynomial model of degree p in d variables, parameters including bias = C(d+p, p).\n"
        "2. Compare sample count N against that parameter count.\n"
        "3. If N < parameters -> under-determined; N = parameters -> even-determined; N > parameters -> over-determined.\n\n"
        "Trap: the examiner often asks both the parameter count and the system type in the same question."
    ),
    "Linear multiclass dimensions with bias": (
        "Midterm-style answer pattern:\n"
        "1. Start with d raw input features.\n"
        "2. Add bias / offset -> parameters per class = d + 1.\n"
        "3. With C classes, W has shape (d+1) x C.\n"
        "4. Total parameters across all classes = (d+1)C.\n\n"
        "Trap: options often swap the matrix shape into C x (d+1), which is wrong for the convention XW ≈ Y."
    ),
    "Bias / offset term role": (
        "Midterm-style answer pattern:\n"
        "- The bias lets the line / hyperplane shift away from the origin.\n"
        "- Removing the bias forces the model to pass through the origin.\n"
        "- Including a bias is equivalent to adding a constant feature of 1 to every sample.\n\n"
        "Trap: the bias increases flexibility, but it does NOT by itself make the model nonlinear."
    ),
    "Ridge lambda increases": (
        "Midterm-style answer pattern:\n"
        "- Larger lambda means stronger L2 penalty on the weights.\n"
        "- As lambda increases, the weight norm usually shrinks.\n"
        "- This can reduce overfitting, though too much regularization can underfit.\n\n"
        "Trap: the examiner often pairs 'more penalty' with 'weights decrease / shrink'."
    ),
    "Loss vs cost vs objective": (
        "Midterm-style answer pattern:\n"
        "- Loss = error on one sample.\n"
        "- Cost = total or average loss across the dataset.\n"
        "- Objective = what is optimized, possibly cost plus regularization.\n"
        "- MSE and SSE have the same minimizer because MSE = SSE / N.\n\n"
        "Trap: a statement calling dataset-wide SSE the 'loss' is usually imprecise or false in exam wording."
    ),
    "Derivative shape rules": (
        "Midterm-style answer pattern:\n"
        "- Scalar wrt scalar -> scalar.\n"
        "- Scalar wrt d x 1 vector -> d x 1 vector.\n"
        "- b x 1 vector wrt scalar -> b x 1 vector.\n"
        "- b x 1 vector wrt d x 1 vector -> b x d Jacobian matrix.\n\n"
        "Trap: scalar wrt vector is NOT a scalar, and vector wrt vector is NOT generally another vector."
    ),
    "Left inverse vs right inverse": (
        "Midterm-style answer pattern:\n"
        "- For X of shape m x d:\n"
        "  Left inverse requires full column rank: rank(X)=d and typically m >= d.\n"
        "  Right inverse requires full row rank: rank(X)=m and typically m <= d.\n"
        "  Ordinary inverse requires square full-rank: m=d=rank(X).\n\n"
        "Trap: a tall full-column-rank matrix can have a left inverse but cannot have a right inverse."
    ),
    "Linear model is not always a straight line": (
        "Midterm-style answer pattern:\n"
        "- A linear model is linear in its parameters.\n"
        "- After feature transformation (for example polynomial features), it can become curved in the original input space.\n\n"
        "Trap: 'linear model' does NOT always mean a straight line in the raw x-space."
    ),
    "Clustering questions": (
        "Midterm-style answer pattern:\n"
        "- Clustering is unsupervised learning.\n"
        "- No target labels are required during clustering itself.\n"
        "- The goal is to discover structure / hidden groups from feature similarity.\n\n"
        "Trap: statements that clustering needs labels or is supervised are false."
    ),
    "Nominal categories + one-hot": (
        "Midterm-style answer pattern:\n"
        "- A feature like City or Major is categorical / nominal.\n"
        "- One-hot encoding is appropriate.\n"
        "- Integer coding such as Singapore=1, Tokyo=2, Seoul=3 introduces a fake order.\n\n"
        "Trap: z-score normalization is for numeric features, not for directly converting nominal categories."
    ),
    "Under-determined high-dimensional data": (
        "Midterm-style answer pattern:\n"
        "- If the number of learnable parameters exceeds the number of samples, the system is under-determined.\n"
        "- This often happens with many features but few observations.\n"
        "- Ridge is often introduced to stabilize or select a unique penalized solution.\n\n"
        "Trap: the examiner often wraps this in a real-world story such as gene-expression or medical features."
    ),
    "Interpolation vs extrapolation": (
        "Midterm-style answer pattern:\n"
        "- Interpolation = prediction within the observed training range.\n"
        "- Extrapolation = prediction outside that range.\n"
        "- Extrapolation is usually less reliable even if the fitted model is linear.\n\n"
        "Trap: 'the model is linear so x=1000 prediction should be accurate' is false."
    ),
    "Bayes from counts / defect rates": (
        "Midterm-style answer pattern:\n"
        "- Convert counts to priors and conditionals first.\n"
        "- Then use Bayes: P(B|A) = P(A|B)P(B) / P(A).\n"
        "- In machine / defect questions, first compute the overall defect probability if needed.\n\n"
        "Trap: do not confuse P(A|B) with P(B|A)."
    ),
    "Repeated-outcome probability": (
        "Midterm-style answer pattern:\n"
        "- For two fair die rolls, there are 36 ordered outcomes.\n"
        "- Equal-number outcomes are (1,1) to (6,6), so probability = 6/36 = 1/6.\n"
        "- For without-replacement card questions, multiply stage-by-stage conditional probabilities.\n\n"
        "Trap: ordered-pair sample spaces are common in the midterm wording."
    ),
    "One-hot regression prediction by argmax": (
        "Midterm-style answer pattern:\n"
        "- Fit scores for every class.\n"
        "- Prediction = class with largest score = argmax_j f_j(x).\n"
        "- If required, convert that winning class into a one-hot vector afterward.\n\n"
        "Trap: do not choose the class whose score is individually closest to 1."
    ),
    "Regularize all parameters including bias": (
        "Midterm-style answer pattern:\n"
        "- If the question explicitly says all parameters, including the bias term, are regularized,\n"
        "  then the ridge penalty applies to the full parameter matrix, not only the non-bias rows.\n\n"
        "Trap: some textbooks exclude the bias from regularization, but the midterm can state the opposite explicitly."
    ),
}


SCENARIO_LEARNING_GUIDE = {
    "Labeled category prediction": (
        "Learning paradigm: Supervised learning\n"
        "Task family: Classification\n"
        "Reason: historical inputs are paired with known class labels, and the goal is to predict one of those categories for new samples."
    ),
    "Labeled numeric prediction": (
        "Learning paradigm: Supervised learning\n"
        "Task family: Regression\n"
        "Reason: historical inputs are paired with known numeric targets, and the goal is to predict a number."
    ),
    "Unlabeled grouping / clusters": (
        "Learning paradigm: Unsupervised learning\n"
        "Task family: Clustering\n"
        "Reason: there are no target labels, and the goal is to discover groups or hidden structure."
    ),
    "Unlabeled unusual-pattern detection": (
        "Learning paradigm: Unsupervised learning\n"
        "Task family: Anomaly detection\n"
        "Reason: unusual patterns are identified without labeled target outcomes."
    ),
    "Reward-driven sequential decisions": (
        "Learning paradigm: Reinforcement learning\n"
        "Task family: Sequential decision-making\n"
        "Reason: the agent learns actions from rewards / penalties over time."
    ),
}


VARIABLE_ENCODING_GUIDE = {
    "City / Major / hand-gesture class": (
        "Data type: Nominal\n"
        "Recommended encoding: One-hot encoding\n"
        "Trap: integer codes such as 1,2,3 introduce a fake ordering."
    ),
    "Poor / Fair / Good / Excellent": (
        "Data type: Ordinal\n"
        "Recommended encoding: ordered integer coding is acceptable when the ordering matters\n"
        "Trap: ordinal data has order, but the gaps are not guaranteed to be equal."
    ),
    "Temperature in Celsius": (
        "Data type: Interval\n"
        "Reason: equal gaps exist, but zero is not an absolute absence."
    ),
    "Speed / height / count with true zero": (
        "Data type: Ratio\n"
        "Reason: equal gaps exist and zero is meaningful."
    ),
}


PREPROCESSING_SCENARIO_GUIDE = {
    "Expert removes poor-quality samples": (
        "Recommended answer: Data cleaning\n"
        "Reason: low-quality / noisy samples are being removed before model training.\n"
        "Trap: this is not feature extraction."
    ),
    "Nominal categories need numeric representation": (
        "Recommended answer: One-hot encoding\n"
        "Reason: nominal categories have no natural order, so one binary column per category is appropriate."
    ),
    "Continuous numeric feature should be mapped to 0..1": (
        "Recommended answer: Linear scaling / min-max scaling\n"
        "Reason: the goal is to place numeric values onto a comparable bounded range."
    ),
    "Numeric features have very different scales around different means": (
        "Recommended answer: Z-score standardization\n"
        "Reason: subtract the mean and divide by standard deviation to make features comparable."
    ),
    "Missing numeric entries are present": (
        "Recommended answer: Mean imputation (or careful cleaning depending on data volume)\n"
        "Reason: arbitrary replacement can distort the analysis."
    ),
}


MIDTERM_PAPER_COVERAGE = {
    "Sem2 Midterm Q15 student data story": (
        "Covered answers:\n"
        "1. Discover groups without labels -> Unsupervised learning\n"
        "2. Predict pass/fail from labeled history -> Supervised learning\n"
        "3. Major -> Nominal data\n"
        "4. Engineering=1, Business=2, Arts=3 is problematic because it introduces an artificial ordering\n\n"
        "Best calculator support:\n"
        "- 16. Tutorial Concepts / MCQ Guide\n"
        "- 12. NOIR Helper\n"
        "- 13. Data Prep"
    ),
    "Sem2 Midterm Q16 factory defect Bayes": (
        "Covered answers:\n"
        "- P(defective)\n"
        "- P(Machine B | defective)\n"
        "- which machine is more likely given non-defective\n\n"
        "Best calculator support:\n"
        "- 15. Bayes / PMF / Normal\n"
        "- 15c. Count / Conditional Probability"
    ),
    "Sem2 Midterm Q17 regression + ridge FITB": (
        "Covered answers:\n"
        "- bias term with bias column of ones\n"
        "- training MSE for linear regression\n"
        "- training MSE for full 2nd-order polynomial ridge\n"
        "- effect of increasing lambda on training MSE\n\n"
        "Best calculator support:\n"
        "- 22. Exam-Style Solver\n"
        "- 10. Ridge (Reg / Binary / Multi)"
    ),
    "Sem2 Midterm Q18 multiclass polynomial ridge FITB": (
        "Covered answers:\n"
        "- total parameters for linear multiclass with bias\n"
        "- total parameters for full 2nd-order polynomial multiclass\n"
        "- class-specific bias coefficient\n"
        "- predicted label by argmax for Xnew\n\n"
        "Best calculator support:\n"
        "- 22. Exam-Style Solver"
    ),
    "PYP Midterm Q22 gesture classification story": (
        "Covered answers:\n"
        "- expert review = data cleaning, not feature extraction\n"
        "- gesture prediction = classification, not regression or clustering\n"
        "- gesture class labels are nominal, not ordinal\n"
        "- valid one-hot encodings need distinct binary indicator patterns with one active class per label\n\n"
        "Best calculator support:\n"
        "- 16b. Midterm Paper Coverage / Open-Ended\n"
        "- 12. NOIR Helper\n"
        "- 13. Data Prep"
    ),
    "PYP Midterm Q23 AutoDrive open-ended": (
        "Covered answers:\n"
        "- labeled object recognition -> supervised learning\n"
        "- anomaly detection without labels -> unsupervised learning\n"
        "- Poor/Fair/Good/Excellent -> ordinal data\n"
        "- scale speed to 0..1 -> linear scaling\n\n"
        "Best calculator support:\n"
        "- 16b. Midterm Paper Coverage / Open-Ended\n"
        "- 12. NOIR Helper\n"
        "- 13. Data Prep"
    ),
    "PYP Midterm Q24 department / team promotion": (
        "Covered answers:\n"
        "- overall promotion probability\n"
        "- conditional probability given not promoted\n"
        "- highest team promotion rate\n"
        "- highest department promotion rate\n\n"
        "Best calculator support:\n"
        "- 15c. Count / Conditional Probability"
    ),
}


# ------------------------------
# Additional math helpers (Phase 2 additions)
# ------------------------------

def kmeans_cluster(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    init_centroids: Optional[np.ndarray] = None,
    tol: float = 1e-9,
) -> dict:
    """
    Run k-means clustering on data matrix X.

    Parameters
    ----------
    X            : (N, d) data matrix
    k            : number of clusters
    max_iter     : maximum number of update iterations
    init_centroids: (k, d) initial centroids; if None the first k rows of X are used
    tol          : convergence tolerance (centroid shift)

    Returns
    -------
    dict with keys:
        labels       : (N,) integer cluster assignments (0-indexed)
        centroids    : (k, d) final centroids
        history      : list of dicts, one per iteration, each containing
                         'iter', 'labels', 'centroids', 'wcss'
        wcss         : final within-cluster sum of squares
    """
    X = ensure_2d_float_array(X, "X")
    ensure_no_missing_or_infinite(X, "X")
    N, d = X.shape
    if k < 1:
        raise ValueError("Number of clusters k must be at least 1.")
    if k > N:
        raise ValueError("k cannot exceed the number of data points.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    if init_centroids is None:
        centroids = X[:k].copy()
    else:
        centroids = ensure_2d_float_array(np.asarray(init_centroids, dtype=float), "init_centroids")
        if centroids.shape != (k, d):
            raise ValueError(f"init_centroids must have shape ({k}, {d}), got {centroids.shape}.")

    history = []
    labels = np.zeros(N, dtype=int)

    for it in range(max_iter):
        # Assignment step: assign each point to nearest centroid
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)  # (N, k)
        new_labels = np.argmin(dists, axis=1)

        # Compute WCSS
        wcss = float(sum(
            np.sum(np.linalg.norm(X[new_labels == j] - centroids[j], axis=1) ** 2)
            for j in range(k)
            if np.any(new_labels == j)
        ))

        history.append({
            "iter": it,
            "labels": new_labels.copy(),
            "centroids": centroids.copy(),
            "wcss": wcss,
        })

        # Update step: recompute centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centroids[j] = np.mean(X[mask], axis=0)
            else:
                new_centroids[j] = centroids[j]  # keep old centroid if cluster is empty

        # Convergence check
        shift = float(np.linalg.norm(new_centroids - centroids))
        labels = new_labels
        centroids = new_centroids
        if shift < tol:
            break

    return {
        "labels": labels,
        "centroids": centroids,
        "history": history,
        "wcss": history[-1]["wcss"] if history else float("nan"),
        "n_iter": len(history),
    }


def decision_tree_mse_splits(y: np.ndarray) -> dict:
    """
    Find the best binary split for a 1-D target vector y (assumed pre-sorted)
    using total squared error as the criterion, matching the Lecture 9 demo.

    For each candidate split index i (splitting at i+1 vs the rest):
        left  = y[:i+1]  (1 to N-1 elements)
        right = y[i+1:]

    Returns a dict with:
        mse_vec   : (N-1,) array of total MSE for each split point
        best_index: 0-indexed split position with minimum total MSE
        left_mean : mean of left partition at best split
        right_mean: mean of right partition at best split
        best_mse  : MSE at best split
        left_data : left partition values
        right_data: right partition values
        all_splits: list of dicts with details for every split
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    if n < 2:
        raise ValueError("Need at least 2 values to consider a split.")

    mse_vec = np.zeros(n - 1)
    all_splits = []
    for i in range(n - 1):
        left = y[: i + 1]
        right = y[i + 1 :]
        mean_left = float(np.mean(left))
        mean_right = float(np.mean(right))
        sq_err_left = float(np.sum((left - mean_left) ** 2))
        sq_err_right = float(np.sum((right - mean_right) ** 2))
        total_mse = (sq_err_left + sq_err_right) / n
        mse_vec[i] = total_mse
        all_splits.append({
            "split_after_index": i,
            "left_count": len(left),
            "right_count": len(right),
            "left_mean": mean_left,
            "right_mean": mean_right,
            "sq_err_left": sq_err_left,
            "sq_err_right": sq_err_right,
            "total_mse": total_mse,
        })

    best_idx = int(np.argmin(mse_vec))
    return {
        "mse_vec": mse_vec,
        "best_index": best_idx,
        "left_mean": all_splits[best_idx]["left_mean"],
        "right_mean": all_splits[best_idx]["right_mean"],
        "best_mse": float(mse_vec[best_idx]),
        "left_data": y[: best_idx + 1],
        "right_data": y[best_idx + 1 :],
        "all_splits": all_splits,
    }


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> dict:
    """
    Compute classification accuracy, error rate, and a confusion matrix.

    Returns dict with:
        classes      : sorted list of unique class labels
        accuracy     : fraction of correct predictions
        error_rate   : 1 - accuracy
        n_correct    : number of correct predictions
        n_total      : total number of predictions
        confusion     : 2-D dict confusion[true][pred] = count
        confusion_mat: numpy array (n_classes x n_classes), rows = true, cols = pred
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Need at least one prediction.")

    classes = sorted(set(y_true) | set(y_pred), key=mixed_label_sort_key)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    confusion: dict = {c: {c2: 0 for c2 in classes} for c in classes}
    n_correct = 0
    for t, p in zip(y_true, y_pred):
        if t not in class_to_idx:
            raise ValueError(f"True label '{t}' is not in the discovered class set.")
        if p not in class_to_idx:
            raise ValueError(f"Predicted label '{p}' is not in the discovered class set.")
        confusion[t][p] += 1
        if t == p:
            n_correct += 1

    confusion_mat = np.zeros((n_classes, n_classes), dtype=int)
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            confusion_mat[i, j] = confusion[c_true][c_pred]

    n_total = len(y_true)
    accuracy = n_correct / n_total
    return {
        "classes": classes,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "confusion": confusion,
        "confusion_mat": confusion_mat,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute standard regression evaluation metrics.

    Returns dict with:
        mse   : mean squared error
        rmse  : root mean squared error
        mae   : mean absolute error
        r2    : coefficient of determination R²
        ss_res: residual sum of squares
        ss_tot: total sum of squares
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    residuals = y_pred - y_true
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    mse_val = float(np.mean(residuals ** 2))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(residuals)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "mse": mse_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "r2": r2,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


# ------------------------------
# GUI widgets/helpers
# ------------------------------

def set_text(widget: tk.Text, text: str) -> None:
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)


def append_result(widget: tk.Text, text: str) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)
    widget.configure(state="disabled")


def labeled_scrolled_text(parent, label: str, height: int = 6, width: int = 40):
    frame = ttk.Frame(parent)
    ttk.Label(frame, text=label).pack(anchor="w")
    txt = ScrolledText(frame, height=height, width=width, wrap="word")
    txt.pack(fill="both", expand=True)
    return frame, txt


# ------------------------------
# Main application
# ------------------------------

class EE2211ToolkitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EE2211 Exam Toolkit GUI")
        self.geometry("1450x900")
        self.minsize(1200, 760)

        self.pages = {}
        self.page_titles = []
        self.filtered_titles = []
        self._build_header()
        self._build_navigation_layout()
        self._build_pages()
        self._populate_nav_list()
        self._select_first_page()

    def _build_header(self):
        header = ttk.Frame(self, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header,
            text="EE2211 Exam Toolkit GUI",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Input format: rows separated by new lines (or ';'), values separated by spaces or commas.\n"
                "Use NaN or NA for missing values. For labels, enter one label per line or separate them by commas. Spaces inside a label are preserved."
            ),
        ).pack(anchor="w", pady=(4, 0))

    def _build_navigation_layout(self):
        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        sidebar = ttk.LabelFrame(body, text="Tool Selection", padding=10)
        sidebar.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(sidebar, text="Search tool:").pack(anchor="w")
        self.nav_search_var = tk.StringVar()
        search_entry = ttk.Entry(sidebar, textvariable=self.nav_search_var, width=28)
        search_entry.pack(fill="x", pady=(4, 8))
        search_entry.bind("<KeyRelease>", self._filter_nav_list)

        ttk.Label(
            sidebar,
            text="Use the left menu to switch between scripts.",
            wraplength=200,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        list_frame = ttk.Frame(sidebar)
        list_frame.pack(fill="both", expand=True)

        self.nav_listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            height=25,
            font=("Segoe UI", 10),
            activestyle="none",
        )
        self.nav_listbox.pack(side="left", fill="both", expand=True)
        self.nav_listbox.bind("<<ListboxSelect>>", self._on_nav_select)

        nav_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.nav_listbox.yview)
        nav_scroll.pack(side="right", fill="y")
        self.nav_listbox.configure(yscrollcommand=nav_scroll.set)

        quick_btns = ttk.Frame(sidebar)
        quick_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(quick_btns, text="Previous", command=self._show_previous_page).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(quick_btns, text="Next", command=self._show_next_page).pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.content_area = ttk.Frame(body)
        self.content_area.pack(side="right", fill="both", expand=True)

    def _build_pages(self):
        self._tab_matrix_inverse()
        self._tab_transpose_rank()
        self._tab_products()
        self._tab_linear_systems()
        self._tab_linreg_one_output()
        self._tab_linreg_multi_output()
        self._tab_binary_classification()
        self._tab_multiclass_classification()
        self._tab_polynomial()
        self._tab_ridge()
        self._tab_stats()
        self._tab_noir()
        self._tab_data_prep()
        self._tab_correlation()
        self._tab_bayes_pmf()
        self._tab_inverse_type_checker()
        self._tab_derivative_shape_checker()
        self._tab_count_probability()
        self._tab_sequential_probability()
        self._tab_tutorial_concepts()
        self._tab_midterm_open_ended()
        self._tab_feature_selection()
        self._tab_model_order_compare()
        self._tab_gradient_descent_scalar()
        self._tab_exponential_gd()
        self._tab_gradient_formula_builder()
        self._tab_exam_style_solver()
        self._tab_metrics_cv()
        self._tab_kmeans()
        self._tab_decision_tree_mse()
        self._tab_neural_network()
        self._tab_classification_metrics()
        self._tab_regression_metrics()
        self._tab_self_test()

    def _create_page(self, title: str):
        page = ttk.Frame(self.content_area)
        self.pages[title] = page
        self.page_titles.append(title)
        return page

    def _populate_nav_list(self, titles=None):
        titles = self.page_titles if titles is None else titles
        self.filtered_titles = list(titles)
        self.nav_listbox.delete(0, tk.END)
        for title in self.filtered_titles:
            self.nav_listbox.insert(tk.END, title)

    def _select_first_page(self):
        if self.filtered_titles:
            self.nav_listbox.selection_clear(0, tk.END)
            self.nav_listbox.selection_set(0)
            self.nav_listbox.activate(0)
            self._show_page(self.filtered_titles[0])

    def _filter_nav_list(self, event=None):
        query = self.nav_search_var.get().strip().lower()
        if not query:
            filtered = self.page_titles
        else:
            filtered = [title for title in self.page_titles if query in title.lower()]
        self._populate_nav_list(filtered)
        if self.filtered_titles:
            self.nav_listbox.selection_set(0)
            self.nav_listbox.activate(0)
            self._show_page(self.filtered_titles[0])

    def _on_nav_select(self, event=None):
        selection = self.nav_listbox.curselection()
        if not selection:
            return
        title = self.filtered_titles[selection[0]]
        self._show_page(title)

    def _show_page(self, title: str):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[title].pack(fill="both", expand=True)
        self.title(f"EE2211 Exam Toolkit GUI — {title}")

    def _show_previous_page(self):
        if not self.filtered_titles:
            return
        selection = self.nav_listbox.curselection()
        idx = selection[0] if selection else 0
        idx = max(0, idx - 1)
        self.nav_listbox.selection_clear(0, tk.END)
        self.nav_listbox.selection_set(idx)
        self.nav_listbox.activate(idx)
        self._show_page(self.filtered_titles[idx])

    def _show_next_page(self):
        if not self.filtered_titles:
            return
        selection = self.nav_listbox.curselection()
        idx = selection[0] if selection else 0
        idx = min(len(self.filtered_titles) - 1, idx + 1)
        self.nav_listbox.selection_clear(0, tk.END)
        self.nav_listbox.selection_set(idx)
        self.nav_listbox.activate(idx)
        self._show_page(self.filtered_titles[idx])

    def _make_two_col_layout(self, parent):
        left = ttk.Frame(parent, padding=10)
        right = ttk.Frame(parent, padding=10)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="both", expand=True)
        return left, right

    def _new_result_box(self, parent):
        frame, txt = labeled_scrolled_text(parent, "Output", height=24, width=60)
        frame.pack(fill="both", expand=True)
        txt.configure(state="disabled")
        append_result(txt, "Results will appear here after you click the Execute / Calculate button.")
        return txt

    def _new_plot_box(self, parent, title: str = "Graph Preview"):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill="both", expand=True, pady=(10, 0))
        self._set_plot_message(
            frame,
            "A graph preview will appear here after calculation when the fitted regression can be drawn in 2D."
        )
        return frame

    def _set_plot_message(self, plot_frame, message: str):
        for child in plot_frame.winfo_children():
            child.destroy()
        ttk.Label(
            plot_frame,
            text=message,
            justify="left",
            wraplength=520,
        ).pack(anchor="w", fill="both", expand=True)

    def _render_matplotlib_figure(self, plot_frame, figure):
        for child in plot_frame.winfo_children():
            child.destroy()
        if not MATPLOTLIB_AVAILABLE:
            self._set_plot_message(plot_frame, "Matplotlib is not available, so the graph preview cannot be shown.")
            return
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plot_frame._figure_canvas = canvas

    def _build_linear_regression_figure(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        w: np.ndarray,
        Xnew: Optional[np.ndarray] = None,
    ):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix for graph preview.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have matching row counts for graph preview.")

        if X.shape[1] == 1:
            x_train = X[:, 0]
            slope = float(w[0])
            equation = f"ŷ = {slope:.4f}x"

            def predictor(xs):
                return slope * xs

        elif X.shape[1] == 2 and np.allclose(X[:, 0], 1.0):
            x_train = X[:, 1]
            intercept = float(w[0])
            slope = float(w[1])
            equation = f"ŷ = {intercept:.4f} {'+' if slope >= 0 else '-'} {abs(slope):.4f}x"

            def predictor(xs):
                return intercept + slope * xs

        else:
            raise ValueError(
                "Graph preview is only available when the effective regression uses one feature: "
                "either X has one column, or X has two columns with the first column all 1s for the bias term."
            )

        x_all = np.asarray(x_train, dtype=float)
        xnew_points = None
        ynew_points = None
        if Xnew is not None:
            Xnew = np.asarray(Xnew, dtype=float)
            if Xnew.ndim != 2 or Xnew.shape[1] != X.shape[1]:
                raise ValueError("Xnew must match X in column count for graph preview.")
            xnew_points = Xnew[:, 0] if X.shape[1] == 1 else Xnew[:, 1]
            ynew_points = (Xnew @ w.reshape(-1, 1)).reshape(-1)
            if xnew_points.size:
                x_all = np.concatenate([x_all, xnew_points])

        xmin = float(np.min(x_all))
        xmax = float(np.max(x_all))
        if math.isclose(xmin, xmax):
            pad = 1.0 if math.isclose(xmin, 0.0) else max(1.0, abs(xmin) * 0.2)
        else:
            pad = 0.1 * (xmax - xmin)
        xs = np.linspace(xmin - pad, xmax + pad, 200)
        ys = predictor(xs)

        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib is not available, so graph preview cannot be shown.")

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x_train, Y, label="Training data")
        ax.plot(xs, ys, label="Fitted line")
        if xnew_points is not None and ynew_points is not None and len(xnew_points) > 0:
            ax.scatter(xnew_points, ynew_points, marker="x", s=80, label="Xnew prediction")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Linear regression fit: {equation}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig, equation

    def _format_polynomial_equation(self, w: np.ndarray) -> str:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.size == 0:
            return "ŷ = 0"

        pieces = [f"ŷ = {w[0]:.4f}"]
        for power, coeff in enumerate(w[1:], start=1):
            sign = '+' if coeff >= 0 else '-'
            if power == 1:
                pieces.append(f" {sign} {abs(coeff):.4f}x")
            else:
                pieces.append(f" {sign} {abs(coeff):.4f}x^{power}")
        return ''.join(pieces)

    def _build_polynomial_regression_figure(
        self,
        X_raw: np.ndarray,
        Y: np.ndarray,
        degree: int,
        w: np.ndarray,
        Xnew: Optional[np.ndarray] = None,
        title_prefix: str = "Polynomial regression",
        curve_label: str = "Fitted curve",
    ):
        X_raw = np.asarray(X_raw, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)

        if X_raw.ndim != 2:
            raise ValueError("Raw X must be a 2D matrix for graph preview.")
        if X_raw.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have matching row counts for graph preview.")
        if X_raw.shape[1] != 1:
            raise ValueError(
                "Graph preview is only available for 1D polynomial / ridge regression, where raw X has exactly one feature column."
            )

        x_train = X_raw[:, 0]
        x_all = np.asarray(x_train, dtype=float)
        xnew_points = None
        ynew_points = None

        if Xnew is not None:
            Xnew = np.asarray(Xnew, dtype=float)
            if Xnew.ndim != 2 or Xnew.shape[1] != 1:
                raise ValueError("Xnew must be a one-column raw feature matrix for graph preview.")
            xnew_points = Xnew[:, 0]
            Pnew, _ = make_polynomial_features(Xnew, degree)
            ynew_points = (Pnew @ w.reshape(-1, 1)).reshape(-1)
            if xnew_points.size:
                x_all = np.concatenate([x_all, xnew_points])

        xmin = float(np.min(x_all))
        xmax = float(np.max(x_all))
        if math.isclose(xmin, xmax):
            pad = 1.0 if math.isclose(xmin, 0.0) else max(1.0, abs(xmin) * 0.2)
        else:
            pad = 0.1 * (xmax - xmin)

        xs = np.linspace(xmin - pad, xmax + pad, 400)
        Pgrid, _ = make_polynomial_features(xs.reshape(-1, 1), degree)
        ys = (Pgrid @ w.reshape(-1, 1)).reshape(-1)
        equation = self._format_polynomial_equation(w)

        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib is not available, so graph preview cannot be shown.")

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x_train, Y, label="Training data")
        ax.plot(xs, ys, label=curve_label)
        if xnew_points is not None and ynew_points is not None and len(xnew_points) > 0:
            ax.scatter(xnew_points, ynew_points, marker="x", s=80, label="Xnew prediction")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title_prefix}: {equation}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig, equation



    def _build_scalar_gd_figure(self, x_values: np.ndarray, cost_values: np.ndarray, power: int):
        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib is not available, so graph preview cannot be shown.")

        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        cost_values = np.asarray(cost_values, dtype=float).reshape(-1)
        xmin = float(np.min(x_values))
        xmax = float(np.max(x_values))
        if math.isclose(xmin, xmax):
            pad = 1.0 if math.isclose(xmin, 0.0) else max(1.0, abs(xmin) * 0.2)
        else:
            pad = 0.2 * (xmax - xmin)
        xs = np.linspace(xmin - pad, xmax + pad, 400)
        ys = xs ** power

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(xs, ys, label=f"g(x) = x^{power}")
        ax.scatter(x_values, cost_values, label="GD iterates")
        ax.set_xlabel("x")
        ax.set_ylabel("g(x)")
        ax.set_title("Scalar gradient descent path")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def _build_exponential_cost_figure(self, histories: List[Tuple[str, np.ndarray, np.ndarray]]):
        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib is not available, so graph preview cannot be shown.")
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        for label, iters, costs in histories:
            ax.plot(np.asarray(iters, dtype=float), np.asarray(costs, dtype=float), label=label)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost (SSE)")
        ax.set_title("Exponential GD cost history")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def _build_exponential_fit_figure(self, x_train: np.ndarray, y_train: np.ndarray, x_plot: np.ndarray, y_plot: np.ndarray):
        if not MATPLOTLIB_AVAILABLE:
            raise ValueError("Matplotlib is not available, so graph preview cannot be shown.")
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(np.asarray(x_train, dtype=float).reshape(-1), np.asarray(y_train, dtype=float).reshape(-1), label="Training data")
        ax.plot(np.asarray(x_plot, dtype=float).reshape(-1), np.asarray(y_plot, dtype=float).reshape(-1), label="Exponential fit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Exponential regression fit")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def _add_equation_guide(self, parent, equation: str, expected_form: str, note: str = ""):
        box = ttk.LabelFrame(parent, text="Equation / Expected Input Form", padding=10)
        box.pack(fill="x", pady=(0, 10))
        ttk.Label(
            box,
            text=f"Intended equation / operation:\n{equation}",
            justify="left",
            wraplength=520,
        ).pack(anchor="w")
        ttk.Label(
            box,
            text=f"Enter into this calculator as:\n{expected_form}",
            justify="left",
            wraplength=520,
        ).pack(anchor="w", pady=(8, 0))
        if note.strip():
            ttk.Label(
                box,
                text=f"Note:\n{note}",
                justify="left",
                wraplength=520,
            ).pack(anchor="w", pady=(8, 0))


    def _tab_matrix_inverse(self):
        tab = self._create_page("1. Inverse / det / adj")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "For a square matrix A, compute det(A), adj(A), and A^{-1}.",
            "Enter the matrix directly as A. This calculator expects the matrix itself, not a transformed system.",
            "A^{-1} exists only when A is square and det(A) != 0."
        )

        frame, self.inv_A = labeled_scrolled_text(left, "Matrix A")
        frame.pack(fill="both", expand=True)
        set_text(self.inv_A, "1, 2\n3, 4")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)

        def run():
            try:
                A = parse_numeric_matrix(self.inv_A.get("1.0", tk.END))
                ensure_no_missing_or_infinite(A, "A")
                result = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_array("A", A),
                    ),
                    "",
                    f"A =\n{format_array(A)}\n",
                ]
                if A.shape[0] != A.shape[1]:
                    result.append("Matrix is not square, so determinant / inverse / adjoint do not exist.")
                else:
                    det_A = np.linalg.det(A)
                    result.append(f"det(A) = {det_A}\n")
                    if abs(det_A) < 1e-12:
                        result.append("A is singular, so A^{-1} does not exist.")
                    else:
                        A_inv = np.linalg.inv(A)
                        adj_A = det_A * A_inv
                        recon = (1 / det_A) * adj_A
                        result.append(section_block(
                            "Inverse-object parameter summary",
                            describe_array("adj(A)", adj_A),
                            describe_array("A^{-1}", A_inv),
                        ))
                        result.append(f"adj(A) =\n{format_array(adj_A)}\n")
                        result.append(f"A^{-1} =\n{format_array(A_inv)}\n")
                        result.append(f"Reconstructed inverse from adj(A)/det(A) =\n{format_array(recon)}")
                append_result(out, "\n".join(result))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(btns, text="Execute / Calculate", command=run).pack(side="left")
        out = self._new_result_box(right)

    def _tab_transpose_rank(self):
        tab = self._create_page("2. Transpose & Rank")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Given a matrix X, compute X^T and rank(X).",
            "Enter the matrix directly as X.",
            "Use this when the question asks about linear independence, invertibility clues, or the number of pivots."
        )

        frame, self.rank_X = labeled_scrolled_text(left, "Matrix X")
        frame.pack(fill="both", expand=True)
        set_text(self.rank_X, "1, 4, 3\n0, 4, 2\n1, 8, 5")

        def run():
            try:
                X = parse_numeric_matrix(self.rank_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")
                r = np.linalg.matrix_rank(X)
                res = "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                        describe_array("X^T", X.T),
                    ),
                    f"X =\n{format_array(X)}\n\nrank(X) = {r}\n\nX^T =\n{format_array(X.T)}",
                ])
                append_result(out, res)
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_products(self):
        tab = self._create_page("3. Matrix / Vector Product")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute a matrix or vector product such as AB, Xw, or w^T X.",
            "Enter the left object in A and the right object in B exactly in the order you want multiplied: result = A @ B.",
            "The inner dimensions must match. For example, to compute Xw, enter X on the left and w on the right."
        )

        frame_a, self.prod_A = labeled_scrolled_text(left, "Left matrix / vector A")
        frame_a.pack(fill="both", expand=True)
        set_text(self.prod_A, "1, 4\n0, 4\n3, -2")

        frame_b, self.prod_B = labeled_scrolled_text(left, "Right matrix / vector B")
        frame_b.pack(fill="both", expand=True)
        set_text(self.prod_B, "1, 4, 3\n0, 4, 2")

        def run():
            try:
                A = parse_numeric_matrix(self.prod_A.get("1.0", tk.END))
                B = parse_numeric_matrix(self.prod_B.get("1.0", tk.END))
                ensure_no_missing_or_infinite(A, "A")
                ensure_no_missing_or_infinite(B, "B")
                C = A @ B
                res = "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_array("A", A),
                        describe_array("B", B),
                        describe_array("A @ B", C),
                    ),
                    f"A shape = {A.shape}\nB shape = {B.shape}\n\nA @ B =\n{format_array(C)}",
                ])
                append_result(out, res)
            except Exception as exc:
                append_result(out, f"Error: {exc}\n\nCheck that the inner dimensions match.")

        ttk.Button(left, text="Execute / Calculate Product", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_linear_systems(self):
        tab = self._create_page("4. Even / Over / Under")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Solve a linear system / least-squares system written in the toolkit as Xw = y.",
            "Put the coefficient matrix into X and the right-hand side into y. The calculator expects the unknown to be the column vector w in Xw = y.",
            "If your question is written in another form, rewrite it first. Example: w^T X = y^T must be transposed to X^T w = y before entering, so place X^T in the X box and y in the y box."
        )

        frame_x, self.sys_X = labeled_scrolled_text(left, "Design matrix X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.sys_X, "1, 1\n1, -1\n1, 0")

        frame_y, self.sys_y = labeled_scrolled_text(left, "Target y")
        frame_y.pack(fill="both", expand=True)
        set_text(self.sys_y, "1\n0\n2")

        options = ttk.Frame(left)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Ridge lambda (0 for no regularization):").pack(side="left")
        self.sys_lambda = tk.StringVar(value="0")
        ttk.Entry(options, textvariable=self.sys_lambda, width=12).pack(side="left", padx=6)

        def run():
            try:
                X = parse_numeric_matrix(self.sys_X.get("1.0", tk.END))
                y = parse_numeric_matrix(self.sys_y.get("1.0", tk.END))
                lam = float(self.sys_lambda.get().strip())
                desc, w = solve_even_over_under(X, y, lam)
                rank_X = np.linalg.matrix_rank(X)
                aug = np.hstack([X, y])
                rank_aug = np.linalg.matrix_rank(aug)
                m, d = X.shape

                if m < d:
                    system_type = "Under-determined"
                elif m == d:
                    system_type = "Even-determined"
                else:
                    system_type = "Over-determined"

                consistent = (rank_X == rank_aug)
                if consistent:
                    if rank_X == d:
                        exact_status = "Exact solution exists and is unique."
                    else:
                        exact_status = "Exact solution exists, but it is not unique (infinitely many exact solutions)."
                else:
                    exact_status = "No exact solution exists because the system is inconsistent (rank(X) != rank([X|y]))."

                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                        describe_target_matrix("y", y),
                        describe_weight_matrix("w", w),
                    ),
                    "",
                    f"X shape = {X.shape}, y shape = {y.shape}",
                    f"m = {m}, d = {d}",
                    f"System type = {system_type}",
                    f"rank(X) = {rank_X}",
                    f"rank([X|y]) = {rank_aug}",
                    f"Consistency = {'consistent' if consistent else 'inconsistent'}",
                    f"Exact-solution check = {exact_status}",
                    "",
                ]

                if lam > 0:
                    res.append(
                        "Note: lambda > 0 means the displayed w is a ridge-regularized solution. "
                        "The exact-solution check above still refers to the original unregularized system Xw = y."
                    )
                    res.append("")

                res += [
                    desc,
                    f"\nw =\n{format_array(w)}",
                ]

                try:
                    y_hat = X @ w
                    residual = y_hat - y
                    residual_norm = float(np.linalg.norm(residual))
                    res.append(f"\nXw =\n{format_array(y_hat)}")
                    res.append(f"\nResidual Xw - y =\n{format_array(residual)}")
                    res.append(f"\nResidual norm ||Xw - y||_2 = {residual_norm}")

                    if lam == 0:
                        if consistent:
                            res.append("\nInterpretation: this w satisfies the system exactly up to numerical precision.")
                        else:
                            res.append(
                                "\nInterpretation: the displayed w is only a least-squares approximation. "
                                "There is no exact solution to Xw = y."
                            )
                except Exception:
                    pass

                append_result(out, "\n".join(res))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate Solution", command=run).pack(anchor="w")
        out = self._new_result_box(right)

    def _tab_linreg_one_output(self):
        tab = self._create_page("5. Linear Reg (1 output)")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Solve linear regression with one output: Xw ≈ y, usually using w = (X^T X)^{-1} X^T y when appropriate.",
            "Enter the design matrix X and the target vector Y. You can either include the bias column yourself, or tick the checkbox to auto-add a leading column of 1s to X and Xnew.",
            "When auto-bias is enabled, enter raw features only. When it is disabled, X and Xnew must already include the bias / offset column if your model needs one. A graph preview is shown automatically when the effective regression is 1D: either one raw feature through the origin, or one raw feature with a bias column."
        )

        frame_x, self.lr1_X = labeled_scrolled_text(left, "X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.lr1_X, "-9\n-7\n-5\n1\n5\n9")

        frame_y, self.lr1_Y = labeled_scrolled_text(left, "Y")
        frame_y.pack(fill="both", expand=True)
        set_text(self.lr1_Y, "-6\n-6\n-4\n-1\n1\n4")

        self.lr1_use_bias = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Automatically add bias / offset column of 1s to X and Xnew",
            variable=self.lr1_use_bias,
        ).pack(anchor="w", pady=(2, 6))

        ttk.Label(
            left,
            text=(
                "Checked: enter raw features only, and the toolkit prepends a column of 1s.\n"
                "Unchecked: enter the full design matrix exactly as you want it used.\n"
                "The graph preview is available only when the effective fitted model has one x-feature."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        frame_xnew, self.lr1_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.lr1_Xnew, "-1")

        output_holder = ttk.Frame(right)
        output_holder.pack(fill="both", expand=True)
        out = self._new_result_box(output_holder)
        plot_box = self._new_plot_box(right, "Linear Regression Graph Preview")

        def run():
            try:
                X_input = parse_numeric_matrix(self.lr1_X.get("1.0", tk.END))
                Y = parse_numeric_matrix(self.lr1_Y.get("1.0", tk.END))
                use_bias = self.lr1_use_bias.get()
                X = add_bias_column(X_input) if use_bias else X_input
                w, method = solve_least_squares(X, Y)
                Yhat = X @ w
                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X_input),
                        describe_design_matrix("Effective X", X),
                        describe_target_matrix("Y", Y),
                        describe_weight_matrix("w", w),
                        describe_target_matrix("Training prediction Xw", Yhat),
                    )
                ]
                if use_bias:
                    res.append("Auto-bias enabled: prepended a leading column of 1s to X and Xnew.")
                    res.append(f"\nRaw input X =\n{format_array(X_input)}")
                    res.append(f"\nEffective design matrix used =\n{format_array(X)}")
                res += [
                    method,
                    f"\nw =\n{format_array(w)}",
                    f"\nTraining prediction Xw =\n{format_array(Yhat)}",
                    f"\nMSE = {mse(Y, Yhat)}",
                ]

                Xnew = None
                xnew_text = clean_text(self.lr1_Xnew.get("1.0", tk.END))
                if xnew_text:
                    Xnew_input = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew_input, "Xnew")
                    Xnew = add_bias_column(Xnew_input) if use_bias else Xnew_input
                    if Xnew.shape[1] != X.shape[1]:
                        raise ValueError(f"Xnew must have {X.shape[1]} columns to match X.")
                    Ynew = Xnew @ w
                    res.append(section_block(
                        "Xnew parameter summary",
                        describe_design_matrix("Raw Xnew", Xnew_input),
                        describe_design_matrix("Effective Xnew", Xnew),
                        describe_target_matrix("Prediction for Xnew", Ynew),
                    ))
                    if use_bias:
                        res.append(f"\nRaw Xnew =\n{format_array(Xnew_input)}")
                        res.append(f"\nEffective Xnew used =\n{format_array(Xnew)}")
                    res.append(f"\nPrediction for Xnew =\n{format_array(Ynew)}")

                try:
                    fig, equation = self._build_linear_regression_figure(X, Y, w, Xnew)
                    self._render_matplotlib_figure(plot_box, fig)
                    res.append(f"\nGraph preview equation = {equation}")
                    if Xnew is not None:
                        res.append("The graph marks Xnew predictions with x-shaped markers.")
                except Exception as plot_exc:
                    self._set_plot_message(plot_box, str(plot_exc))
                    res.append(f"\nGraph preview note: {plot_exc}")

                append_result(out, "\n".join(res))
            except Exception as exc:
                self._set_plot_message(plot_box, "Graph unavailable because the calculation failed.")
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)

    def _tab_linreg_multi_output(self):

        tab = self._create_page("6. Linear Reg (multi-output)")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Solve multi-output linear regression: XW ≈ Y.",
            "Enter the design matrix X and the multi-column target matrix Y. You can either include the bias column yourself, or tick the checkbox to auto-add a leading column of 1s to X and Xnew.",
            "Each column of Y is one output, and each column of W contains the weights for that output. When auto-bias is enabled, enter raw features only."
        )

        frame_x, self.lrm_X = labeled_scrolled_text(left, "X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.lrm_X, "1, 1\n-1, 1\n1, 3\n1, 0")

        frame_y, self.lrm_Y = labeled_scrolled_text(left, "Y (multiple columns allowed)")
        frame_y.pack(fill="both", expand=True)
        set_text(self.lrm_Y, "1, 0\n0, 1\n2, -1\n-1, 3")

        self.lrm_use_bias = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Automatically add bias / offset column of 1s to X and Xnew",
            variable=self.lrm_use_bias,
        ).pack(anchor="w", pady=(2, 6))

        ttk.Label(
            left,
            text=(
                "Checked: enter raw features only, and the toolkit prepends a column of 1s.\n"
                "Unchecked: enter the full design matrix exactly as you want it used."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        frame_xnew, self.lrm_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.lrm_Xnew, "6, 8\n0, -1")

        def run():
            try:
                X_input = parse_numeric_matrix(self.lrm_X.get("1.0", tk.END))
                Y = parse_numeric_matrix(self.lrm_Y.get("1.0", tk.END))
                use_bias = self.lrm_use_bias.get()
                X = add_bias_column(X_input) if use_bias else X_input
                W, method = solve_least_squares(X, Y)
                Yhat = X @ W
                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X_input),
                        describe_design_matrix("Effective X", X),
                        describe_target_matrix("Y", Y),
                        describe_weight_matrix("W", W),
                        describe_target_matrix("Training prediction XW", Yhat),
                    )
                ]
                if use_bias:
                    res.append("Auto-bias enabled: prepended a leading column of 1s to X and Xnew.")
                    res.append(f"\nRaw input X =\n{format_array(X_input)}")
                    res.append(f"\nEffective design matrix used =\n{format_array(X)}")
                res += [
                    method,
                    f"\nW =\n{format_array(W)}",
                    f"\nTraining prediction XW =\n{format_array(Yhat)}",
                    f"\nOverall MSE = {mse(Y, Yhat)}",
                ]
                xnew_text = clean_text(self.lrm_Xnew.get("1.0", tk.END))
                if xnew_text:
                    Xnew_input = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew_input, "Xnew")
                    Xnew = add_bias_column(Xnew_input) if use_bias else Xnew_input
                    if Xnew.shape[1] != X.shape[1]:
                        raise ValueError(f"Xnew must have {X.shape[1]} columns to match X.")
                    Ynew = Xnew @ W
                    if use_bias:
                        res.append(f"\nRaw Xnew =\n{format_array(Xnew_input)}")
                        res.append(f"\nEffective Xnew used =\n{format_array(Xnew)}")
                    res.append(f"\nPrediction for Xnew =\n{format_array(Ynew)}")
                append_result(out, "\n".join(res))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)


    def _tab_binary_classification(self):
        tab = self._create_page("7. Binary Classification")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Train with Xw ≈ y using labels in {-1, +1}, then classify with sign(Xnew w).",
            "Enter raw features and tick auto-bias to prepend a column of 1s automatically, or enter the full design matrix yourself if auto-bias is off.",
            "This tab uses sign(raw score). A raw score of 0 is mapped to +1. Xnew is optional: leave it blank when the question only asks for the fitted classifier."
        )
    
        frame_x, self.bin_X = labeled_scrolled_text(left, "X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.bin_X, "-9\n-7\n-5\n1\n5\n9")
    
        frame_y, self.bin_y = labeled_scrolled_text(left, "y in {-1, +1}")
        frame_y.pack(fill="both", expand=True)
        set_text(self.bin_y, "-1\n-1\n-1\n1\n1\n1")
    
        self.bin_use_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            left,
            text="Automatically add bias / offset column of 1s to X and Xnew",
            variable=self.bin_use_bias,
        ).pack(anchor="w", pady=(2, 6))
    
        frame_xnew, self.bin_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.bin_Xnew, "-2\n2")
    
        def run():
            try:
                X_input = parse_numeric_matrix(self.bin_X.get("1.0", tk.END))
                y = parse_numeric_matrix(self.bin_y.get("1.0", tk.END))
                unique_y = set(np.asarray(y, dtype=float).reshape(-1).tolist())
                if not unique_y.issubset({-1.0, 1.0}):
                    raise ValueError("Binary classification expects y values only in {-1, +1}.")
    
                use_bias = self.bin_use_bias.get()
                X = add_bias_column(X_input) if use_bias else X_input
                w, method = solve_least_squares(X, y)
                train_raw = X @ w
                train_class = binary_sign(train_raw)
    
                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X_input),
                        describe_design_matrix("Effective X", X),
                        describe_target_matrix("y", y),
                        describe_weight_matrix("w", w),
                        describe_target_matrix("Training raw scores Xw", train_raw),
                    )
                ]
                if use_bias:
                    res.append("Auto-bias enabled: prepended a leading column of 1s to X and Xnew.")
                    res.append(f"\nRaw input X =\n{format_array(X_input)}")
                    res.append(f"\nEffective design matrix used =\n{format_array(X)}")
                res += [
                    method,
                    f"\nEstimated w =\n{format_array(w)}",
                    f"\nTraining raw scores Xw =\n{format_array(train_raw)}",
                    f"\nTraining predicted class sign(raw) =\n{format_array(train_class)}",
                ]
    
                xnew_text = clean_text(self.bin_Xnew.get("1.0", tk.END))
                if xnew_text:
                    Xnew_input = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew_input, "Xnew")
                    Xnew = add_bias_column(Xnew_input) if use_bias else Xnew_input
                    if Xnew.shape[1] != X.shape[1]:
                        raise ValueError(f"Xnew must have {X.shape[1]} columns to match X.")
                    y_raw = Xnew @ w
                    y_class = binary_sign(y_raw)
                    res.append(section_block(
                        "Xnew parameter summary",
                        describe_design_matrix("Raw Xnew", Xnew_input),
                        describe_design_matrix("Effective Xnew", Xnew),
                        describe_target_matrix("Raw prediction", y_raw),
                    ))
                    if use_bias:
                        res.append(f"\nRaw Xnew =\n{format_array(Xnew_input)}")
                        res.append(f"\nEffective Xnew used =\n{format_array(Xnew)}")
                    res += [
                        f"\nRaw prediction =\n{format_array(y_raw)}",
                        f"\nPredicted class sign(raw) with ties mapped to +1 =\n{format_array(y_class)}",
                    ]
                else:
                    res.append("\nNo Xnew provided, so only the fitted classifier and training fit are shown.")
    
                append_result(out, "\n".join(res))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)


    def _tab_multiclass_classification(self):
        tab = self._create_page("8. Multi-class Classification")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Train with XW ≈ Y_onehot, then classify with argmax(Xnew W).",
            "Enter raw features and tick auto-bias to prepend a column of 1s automatically, or enter the full design matrix yourself if auto-bias is off.",
            "The toolkit converts labels to one-hot form using the first-seen label order, then predicts the class with the largest score. Xnew is optional."
        )
    
        frame_x, self.mc_X = labeled_scrolled_text(left, "X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.mc_X, "1, 1\n-1, 1\n1, 3\n1, 0")
    
        frame_y, self.mc_y = labeled_scrolled_text(left, "Class labels (one per row or comma-separated)")
        frame_y.pack(fill="both", expand=True)
        set_text(self.mc_y, "1\n2\n1\n3")
    
        self.mc_use_bias = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="Automatically add bias / offset column of 1s to X and Xnew",
            variable=self.mc_use_bias,
        ).pack(anchor="w", pady=(2, 6))
    
        frame_xnew, self.mc_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.mc_Xnew, "6, 8\n0, -1")
    
        def run():
            try:
                X_input = parse_numeric_matrix(self.mc_X.get("1.0", tk.END))
                labels = parse_label_list(self.mc_y.get("1.0", tk.END))
                if len(labels) != X_input.shape[0]:
                    raise ValueError("Number of labels must match number of samples.")
    
                use_bias = self.mc_use_bias.get()
                X = add_bias_column(X_input) if use_bias else X_input
    
                Y, classes = one_hot_encode(labels)
                W, method = solve_least_squares(X, Y)
                train_scores = X @ W
                train_idx = np.argmax(train_scores, axis=1)
                train_labels = [classes[i] for i in train_idx]
    
                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X_input),
                        describe_design_matrix("Effective X", X),
                        describe_target_matrix("One-hot Y", Y),
                        describe_weight_matrix("W", W),
                        describe_target_matrix("Training raw scores X @ W", train_scores),
                    )
                ]
                if use_bias:
                    res.append("Auto-bias enabled: prepended a leading column of 1s to X and Xnew.")
                    res.append(f"\nRaw input X =\n{format_array(X_input)}")
                    res.append(f"\nEffective design matrix used =\n{format_array(X)}")
                res += [
                    f"Classes = {classes}",
                    f"\nOne-hot encoded Y =\n{format_array(Y)}",
                    f"\n{method}",
                    f"\nEstimated W =\n{format_array(W)}",
                    f"\nTraining raw scores X @ W =\n{format_array(train_scores)}",
                    f"\nTraining predicted labels = {train_labels}",
                ]
    
                xnew_text = clean_text(self.mc_Xnew.get("1.0", tk.END))
                if xnew_text:
                    Xnew_input = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew_input, "Xnew")
                    Xnew = add_bias_column(Xnew_input) if use_bias else Xnew_input
                    if Xnew.shape[1] != X.shape[1]:
                        raise ValueError(f"Xnew must have {X.shape[1]} columns to match X.")
                    Yraw = Xnew @ W
                    pred_idx = np.argmax(Yraw, axis=1)
                    pred_labels = [classes[i] for i in pred_idx]
                    pred_onehot = np.zeros_like(Yraw)
                    pred_onehot[np.arange(len(pred_idx)), pred_idx] = 1
                    res.append(section_block(
                        "Xnew parameter summary",
                        describe_design_matrix("Raw Xnew", Xnew_input),
                        describe_design_matrix("Effective Xnew", Xnew),
                        describe_target_matrix("Raw scores Xnew @ W", Yraw),
                        describe_target_matrix("Predicted one-hot", pred_onehot),
                    ))
                    if use_bias:
                        res.append(f"\nRaw Xnew =\n{format_array(Xnew_input)}")
                        res.append(f"\nEffective Xnew used =\n{format_array(Xnew)}")
                    res += [
                        f"\nRaw scores Xnew @ W =\n{format_array(Yraw)}",
                        f"\nPredicted labels = {pred_labels}",
                        f"\nPredicted one-hot =\n{format_array(pred_onehot)}",
                    ]
                else:
                    res.append("\nNo Xnew provided, so only the fitted multiclass classifier and training fit are shown.")
    
                append_result(out, "\n".join(res))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_polynomial(self):
        tab = self._create_page("9. Polynomial Regression / Classif")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "First build polynomial features P = Φ(X), then solve either Pw ≈ y, sign(Pnew w), or argmax(Pnew W) depending on the selected mode.",
            "Enter the original raw features X without a bias column. This calculator automatically expands them into polynomial features including the constant term.",
            "Use regression for continuous targets, binary for labels in {-1, +1}, and multiclass for class labels. Xnew is optional: leave it blank when the question only asks for the fitted model / weights."
        )

        frame_x, self.poly_X = labeled_scrolled_text(left, "Original X (NO bias column here)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.poly_X, "0, 0\n1, 1\n1, 0\n0, 1")

        frame_y, self.poly_y = labeled_scrolled_text(left, "y (continuous, binary, or class labels)")
        frame_y.pack(fill="both", expand=True)
        set_text(self.poly_y, "-1\n-1\n1\n1")

        frame_xnew, self.poly_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.poly_Xnew, "0.1, 0.1\n0.9, 0.9\n0.1, 0.9\n0.9, 0.1")

        options = ttk.Frame(left)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Degree:").pack(side="left")
        self.poly_degree = tk.StringVar(value="2")
        ttk.Entry(options, textvariable=self.poly_degree, width=8).pack(side="left", padx=(4, 12))

        ttk.Label(options, text="Mode:").pack(side="left")
        self.poly_mode = tk.StringVar(value="binary")
        ttk.Combobox(options, textvariable=self.poly_mode, values=["regression", "binary", "multiclass"], width=14, state="readonly").pack(side="left")

        output_holder = ttk.Frame(right)
        output_holder.pack(fill="both", expand=True)
        out = self._new_result_box(output_holder)
        plot_box = self._new_plot_box(right, "Polynomial Regression Graph Preview")

        def run():
            try:
                X = parse_numeric_matrix(self.poly_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")
                xnew_text = clean_text(self.poly_Xnew.get("1.0", tk.END))
                Xnew = None
                if xnew_text:
                    Xnew = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew, "Xnew")
                degree = int(self.poly_degree.get().strip())
                mode = self.poly_mode.get().strip()
                P, names = make_polynomial_features(X, degree)
                Pnew = None if Xnew is None else make_polynomial_features(Xnew, degree)[0]
                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X),
                        describe_design_matrix("Polynomial feature matrix P", P),
                    ),
                    f"Polynomial feature names = {names}",
                    f"\nP =\n{format_array(P)}",
                ]

                if mode == "regression":
                    y = parse_numeric_matrix(self.poly_y.get("1.0", tk.END))
                    w, method = solve_least_squares(P, y)
                    yhat = P @ w
                    res += [
                        section_block(
                            "Regression parameter summary",
                            describe_target_matrix("y", y),
                            describe_weight_matrix("w", w),
                            describe_target_matrix("Training prediction P @ w", yhat),
                        ),
                        f"\n{method}",
                        f"\nw =\n{format_array(w)}",
                        f"\nTraining prediction P @ w =\n{format_array(yhat)}",
                        f"\nTraining MSE = {mse(y, yhat)}",
                    ]
                    if Pnew is not None:
                        ynew = Pnew @ w
                        res.append(section_block(
                            "Xnew parameter summary",
                            describe_design_matrix("Raw Xnew", Xnew),
                            describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                            describe_target_matrix("Prediction Pnew @ w", ynew),
                        ))
                        res.append(f"\nPrediction Pnew @ w =\n{format_array(ynew)}")
                    else:
                        res.append("\nNo Xnew provided, so only the fitted polynomial model / training fit is shown.")
                    try:
                        fig, equation = self._build_polynomial_regression_figure(
                            X, y, degree, w, Xnew, title_prefix="Polynomial regression", curve_label="Fitted polynomial"
                        )
                        self._render_matplotlib_figure(plot_box, fig)
                        res.append(f"\nGraph preview equation = {equation}")
                        if Xnew is not None and Xnew.size:
                            res.append("The graph marks Xnew predictions with x-shaped markers.")
                    except Exception as plot_exc:
                        self._set_plot_message(plot_box, str(plot_exc))
                        res.append(f"\nGraph preview note: {plot_exc}")
                elif mode == "binary":
                    y = parse_numeric_matrix(self.poly_y.get("1.0", tk.END))
                    unique_y = set(np.asarray(y, dtype=float).reshape(-1).tolist())
                    if not unique_y.issubset({-1.0, 1.0}):
                        raise ValueError("Binary mode expects y values only in {-1, +1}.")
                    w, method = solve_least_squares(P, y)
                    res += [
                        section_block(
                            "Binary polynomial parameter summary",
                            describe_target_matrix("y", y),
                            describe_weight_matrix("w", w),
                        ),
                        f"\n{method}",
                        f"\nw =\n{format_array(w)}",
                    ]
                    if Pnew is not None:
                        ynew = Pnew @ w
                        yclass = binary_sign(ynew)
                        res += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                                describe_target_matrix("Raw prediction", ynew),
                            ),
                            f"\nRaw prediction =\n{format_array(ynew)}",
                            f"\nPredicted class sign(raw) with ties mapped to +1 =\n{format_array(yclass)}",
                        ]
                    else:
                        res.append("\nNo Xnew provided, so only the fitted classifier weights are shown.")
                    self._set_plot_message(
                        plot_box,
                        "Graph preview is currently provided for polynomial regression only. For binary classification, inspect the scores / labels instead."
                    )
                else:
                    labels = parse_label_list(self.poly_y.get("1.0", tk.END))
                    if len(labels) != X.shape[0]:
                        raise ValueError("Number of labels must match number of samples.")
                    Y, classes = one_hot_encode(labels)
                    W, method = solve_least_squares(P, Y)
                    res += [
                        section_block(
                            "Multiclass polynomial parameter summary",
                            describe_target_matrix("One-hot Y", Y),
                            describe_weight_matrix("W", W),
                        ),
                        f"\nClasses = {classes}",
                        f"\n{method}",
                        f"\nW =\n{format_array(W)}",
                    ]
                    if Pnew is not None:
                        Yraw = Pnew @ W
                        idx = np.argmax(Yraw, axis=1)
                        pred_labels = [classes[i] for i in idx]
                        res += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                                describe_target_matrix("Raw scores", Yraw),
                            ),
                            f"\nRaw scores =\n{format_array(Yraw)}",
                            f"\nPredicted labels = {pred_labels}",
                        ]
                    else:
                        res.append("\nNo Xnew provided, so only the fitted multiclass weights are shown.")
                    self._set_plot_message(
                        plot_box,
                        "Graph preview is currently provided for polynomial regression only. For multiclass classification, inspect the class scores / predicted labels instead."
                    )

                append_result(out, "\n".join(res))
            except Exception as exc:
                self._set_plot_message(plot_box, "Graph unavailable because the calculation failed.")
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w", pady=6)



    def _tab_ridge(self):
        tab = self._create_page("10. Ridge (Reg / Binary / Multi)")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Build polynomial features P = Φ(X), then solve ridge regression with W = (P^T P + λI)^(-1) P^T Y. This tab also checks the dual form W = P^T (P P^T + λI)^(-1) Y.",
            "Enter the original raw features X without a bias column. This calculator automatically expands them into polynomial features including the constant term. Choose regression, binary, or multiclass mode.",
            "Use λ > 0. For multiclass, enter class labels and the toolkit uses one-hot targets. All parameters, including the constant / bias term, are regularized in this tab."
        )

        frame_x, self.ridge_X = labeled_scrolled_text(left, "X (original features, NO bias column)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.ridge_X, "0, 0\n1, 1\n1, 0\n0, 1")

        frame_y, self.ridge_y = labeled_scrolled_text(left, "y / class labels")
        frame_y.pack(fill="both", expand=True)
        set_text(self.ridge_y, "-1\n-1\n1\n1")

        frame_xnew, self.ridge_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.ridge_Xnew, "0.1, 0.1\n0.9, 0.9\n0.1, 0.9\n0.9, 0.1")

        options = ttk.Frame(left)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Degree:").pack(side="left")
        self.ridge_degree = tk.StringVar(value="2")
        ttk.Entry(options, textvariable=self.ridge_degree, width=8).pack(side="left", padx=(4, 12))

        ttk.Label(options, text="Lambda:").pack(side="left")
        self.ridge_lambda = tk.StringVar(value="0.0001")
        ttk.Entry(options, textvariable=self.ridge_lambda, width=12).pack(side="left", padx=4)

        ttk.Label(options, text="Mode:").pack(side="left", padx=(12, 0))
        self.ridge_mode = tk.StringVar(value="regression")
        ttk.Combobox(
            options,
            textvariable=self.ridge_mode,
            values=["regression", "binary", "multiclass"],
            width=14,
            state="readonly",
        ).pack(side="left", padx=4)

        ttk.Label(
            left,
            text=(
                "Regression: y must be numeric.\n"
                "Binary: y must contain only -1 and +1.\n"
                "Multiclass: enter class labels one per row or comma-separated.\n"
                "This tab reports parameters per output/class, total parameters across all outputs/classes, "
                "and the bias / constant-term coefficients explicitly."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        output_holder = ttk.Frame(right)
        output_holder.pack(fill="both", expand=True)
        out = self._new_result_box(output_holder)
        plot_box = self._new_plot_box(right, "Ridge Graph Preview")

        def run():
            try:
                X = parse_numeric_matrix(self.ridge_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")
                xnew_text = clean_text(self.ridge_Xnew.get("1.0", tk.END))
                Xnew = None
                if xnew_text:
                    Xnew = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew, "Xnew")

                degree = int(self.ridge_degree.get().strip())
                lam = float(self.ridge_lambda.get().strip())
                mode = self.ridge_mode.get().strip()
                if lam <= 0:
                    raise ValueError("Lambda must be > 0 for ridge regression.")

                P, names = make_polynomial_features(X, degree)
                Pnew = None if Xnew is None else make_polynomial_features(Xnew, degree)[0]

                classes = None
                if mode == "regression":
                    Y = parse_numeric_matrix(self.ridge_y.get("1.0", tk.END))
                    ensure_no_missing_or_infinite(Y, "y")
                elif mode == "binary":
                    Y = parse_numeric_matrix(self.ridge_y.get("1.0", tk.END))
                    ensure_no_missing_or_infinite(Y, "y")
                    unique_y = set(np.asarray(Y, dtype=float).reshape(-1).tolist())
                    if not unique_y.issubset({-1.0, 1.0}):
                        raise ValueError("Binary mode expects y values only in {-1, +1}.")
                else:
                    labels = parse_label_list(self.ridge_y.get("1.0", tk.END))
                    if len(labels) != X.shape[0]:
                        raise ValueError("Number of labels must match number of samples.")
                    Y, classes = one_hot_encode(labels)

                I_primal = np.eye(P.shape[1])
                I_dual = np.eye(P.shape[0])
                A_primal = P.T @ P + lam * I_primal
                B_primal = P.T @ Y
                A_dual = P @ P.T + lam * I_dual

                try:
                    W_primal = np.linalg.solve(A_primal, B_primal)
                except np.linalg.LinAlgError:
                    W_primal = np.linalg.pinv(A_primal) @ B_primal

                try:
                    W_dual = P.T @ np.linalg.solve(A_dual, Y)
                except np.linalg.LinAlgError:
                    W_dual = P.T @ (np.linalg.pinv(A_dual) @ Y)

                train_primal = P @ W_primal
                summary = polynomial_model_summary(X, degree, n_outputs=Y.shape[1] if Y.ndim == 2 else 1)

                res = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Raw X", X),
                        describe_design_matrix("Polynomial feature matrix P", P),
                        describe_target_matrix("Target matrix Y", Y),
                        describe_weight_matrix("W_primal", W_primal),
                        describe_weight_matrix("W_dual", W_dual),
                        describe_target_matrix("Training score / prediction P @ W_primal", train_primal),
                    ),
                    f"Polynomial feature names = {names}",
                    f"\nMode = {mode}",
                    f"lambda = {lam}",
                    f"Samples N = {summary['n_samples']}",
                    f"Raw feature count d = {summary['n_features']}",
                    f"Polynomial degree p = {summary['degree']}",
                    f"Parameters per output / class = {summary['params_per_output']}",
                    f"Number of outputs / classes = {summary['n_outputs']}",
                    f"Total learned parameters across all outputs / classes = {summary['total_params']}",
                    f"System type before ridge regularization = {summary['determination']}",
                    "\nAll parameters, including the bias / constant-term coefficient row, are regularized in this tab.",
                    f"\nBias / constant-term coefficients from primal solution =\n{format_array(W_primal[0:1, :])}",
                    f"\nPrimal solution W = (P^T P + lambda I)^(-1) P^T Y\n{format_array(W_primal)}",
                    f"\nDual solution W = P^T (P P^T + lambda I)^(-1) Y\n{format_array(W_dual)}",
                    f"\nMax |primal - dual| = {np.max(np.abs(W_primal - W_dual))}",
                ]

                if mode == "regression":
                    res += [
                        f"\nTraining prediction P @ W_primal =\n{format_array(train_primal)}",
                        f"\nTraining MSE = {mse(Y, train_primal)}",
                    ]
                    if Pnew is not None:
                        Y_primal = Pnew @ W_primal
                        Y_dual = Pnew @ W_dual
                        res += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                                describe_target_matrix("Primal prediction Pnew @ W_primal", Y_primal),
                                describe_target_matrix("Dual prediction Pnew @ W_dual", Y_dual),
                            ),
                            f"\nPrimal prediction Pnew @ W_primal =\n{format_array(Y_primal)}",
                            f"\nDual prediction Pnew @ W_dual =\n{format_array(Y_dual)}",
                        ]
                    else:
                        res.append("\nNo Xnew provided, so only the fitted ridge model / training fit is shown.")

                    try:
                        fig, equation = self._build_polynomial_regression_figure(
                            X, Y, degree, W_primal, Xnew, title_prefix=f"Ridge regression (λ={lam})", curve_label="Fitted ridge curve"
                        )
                        self._render_matplotlib_figure(plot_box, fig)
                        res.append(f"\nGraph preview equation = {equation}")
                        if Xnew is not None and Xnew.size:
                            res.append("The graph marks Xnew predictions with x-shaped markers.")
                        res.append("The graph uses the primal ridge solution; the dual curve should overlap up to numerical precision.")
                    except Exception as plot_exc:
                        self._set_plot_message(plot_box, str(plot_exc))
                        res.append(f"\nGraph preview note: {plot_exc}")

                elif mode == "binary":
                    train_class = binary_sign(train_primal)
                    res += [
                        f"\nTraining raw scores P @ W_primal =\n{format_array(train_primal)}",
                        f"\nTraining predicted class sign(raw) with ties mapped to +1 =\n{format_array(train_class)}",
                        f"\nTraining MSE on raw scores = {mse(Y, train_primal)}",
                    ]
                    if Pnew is not None:
                        Y_primal = Pnew @ W_primal
                        Y_dual = Pnew @ W_dual
                        Y_class = binary_sign(Y_primal)
                        res += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                                describe_target_matrix("Raw prediction Pnew @ W_primal", Y_primal),
                            ),
                            f"\nPrimal raw prediction Pnew @ W_primal =\n{format_array(Y_primal)}",
                            f"\nDual raw prediction Pnew @ W_dual =\n{format_array(Y_dual)}",
                            f"\nPredicted class sign(raw) with ties mapped to +1 =\n{format_array(Y_class)}",
                        ]
                    else:
                        res.append("\nNo Xnew provided, so only the fitted ridge classifier / training fit is shown.")
                    self._set_plot_message(
                        plot_box,
                        "Graph preview is currently shown for numeric ridge regression only. For binary ridge classification, inspect the raw scores and the sign-based predicted labels."
                    )

                else:
                    train_idx = np.argmax(train_primal, axis=1)
                    train_labels = [classes[i] for i in train_idx]
                    train_onehot = np.zeros_like(train_primal)
                    train_onehot[np.arange(len(train_idx)), train_idx] = 1.0
                    res += [
                        f"\nClasses = {classes}",
                        f"\nOne-hot target Y =\n{format_array(Y)}",
                        f"\nTraining raw scores P @ W_primal =\n{format_array(train_primal)}",
                        f"\nTraining predicted labels via argmax = {train_labels}",
                        f"\nTraining predicted one-hot via argmax =\n{format_array(train_onehot)}",
                        f"\nTraining MSE on one-hot targets = {mse(Y, train_primal)}",
                    ]
                    if Pnew is not None:
                        Y_primal = Pnew @ W_primal
                        Y_dual = Pnew @ W_dual
                        idx = np.argmax(Y_primal, axis=1)
                        pred_labels = [classes[i] for i in idx]
                        pred_onehot = np.zeros_like(Y_primal)
                        pred_onehot[np.arange(len(idx)), idx] = 1.0
                        res += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Polynomial feature matrix Pnew", Pnew),
                                describe_target_matrix("Raw class scores Pnew @ W_primal", Y_primal),
                                describe_target_matrix("Predicted one-hot via argmax", pred_onehot),
                            ),
                            f"\nPrimal raw class scores Pnew @ W_primal =\n{format_array(Y_primal)}",
                            f"\nDual raw class scores Pnew @ W_dual =\n{format_array(Y_dual)}",
                            f"\nPredicted labels via argmax = {pred_labels}",
                            f"\nPredicted one-hot via argmax =\n{format_array(pred_onehot)}",
                        ]
                    else:
                        res.append("\nNo Xnew provided, so only the fitted ridge multiclass model / training fit is shown.")
                    self._set_plot_message(
                        plot_box,
                        "Graph preview is currently shown for numeric ridge regression only. For multiclass ridge classification, inspect the class scores, bias row, total parameter count, and argmax-based predicted labels."
                    )

                append_result(out, "\n".join(res))
            except Exception as exc:
                self._set_plot_message(plot_box, "Graph unavailable because the calculation failed.")
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate Ridge", command=run).pack(anchor="w", pady=6)

    def _tab_stats(self):
        tab = self._create_page("11. Stats & Quartiles")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute descriptive statistics such as mean, median, mode, quartiles, IQR, and standard deviation.",
            "Enter the raw list of values directly.",
            "This tab reports both population and sample standard deviation, and it shows two quartile conventions so you can match your tutorial or lecturer."
        )

        frame, self.stats_vals = labeled_scrolled_text(left, "Values")
        frame.pack(fill="both", expand=True)
        set_text(self.stats_vals, "1, 3, 4, 6, 6, 7, 8")

        ttk.Label(
            left,
            text=(
                "This tab reports both population and sample standard deviation, and shows two quartile conventions:\n"
                "1) median of halves, and 2) percentile-linear."
            ),
            justify="left",
        ).pack(anchor="w", pady=(4, 6))

        def run():
            try:
                vals = parse_numeric_matrix(self.stats_vals.get("1.0", tk.END)).reshape(-1)
                ensure_no_missing_or_infinite(vals, "Values")
                stats = descriptive_stats(vals.tolist())

                q1_halves, med_halves, q3_halves = stats["quartiles_median_of_halves"]
                q1_pct, med_pct, q3_pct = stats["quartiles_percentile_linear"]

                res = "\n".join([
                    "Parameter / Dimension Summary",
                    "",
                    f"Number of scalar values entered = {len(vals)}",
                    "Each value contributes 1 scalar parameter / observation slot.",
                    "",
                    f"count = {stats['count']}",
                    f"mean = {stats['mean']}",
                    f"median = {stats['median']}",
                    f"mode = {stats['mode']}",
                    f"min = {stats['min']}",
                    f"max = {stats['max']}",
                    "",
                    f"std (population, ddof=0) = {stats['std_population']}",
                    f"std (sample, ddof=1) = {stats['std_sample']}",
                    "",
                    f"Q1 / median / Q3 using median-of-halves = ({q1_halves}, {med_halves}, {q3_halves})",
                    f"IQR using median-of-halves = {stats['iqr']}",
                    f"Q1 / median / Q3 using percentile-linear = ({q1_pct}, {med_pct}, {q3_pct})",
                ])
                append_result(out, res)
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate Stats", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_noir(self):
        tab = self._create_page("12. NOIR Helper")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Classify a variable into the NOIR scale: Nominal, Ordinal, Interval, or Ratio.",
            "Answer the three prompts about ordering, equal intervals, and true zero.",
            "This is a logic helper rather than a numerical solver."
        )

        container = ttk.Frame(left)
        container.pack(fill="x", pady=10)

        self.noir_ordered = tk.StringVar(value="No")
        self.noir_equal = tk.StringVar(value="No")
        self.noir_zero = tk.StringVar(value="No")

        def add_combo(label, var, row):
            ttk.Label(container, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Combobox(container, textvariable=var, values=["Yes", "No"], width=10, state="readonly").grid(row=row, column=1, sticky="w", padx=8)

        add_combo("Ordered?", self.noir_ordered, 0)
        add_combo("Equal intervals?", self.noir_equal, 1)
        add_combo("True zero means none?", self.noir_zero, 2)

        ttk.Label(left, text="Logic: Not ordered → Nominal; ordered but not equal intervals → Ordinal; equal intervals without true zero → Interval; equal intervals with true zero → Ratio.").pack(anchor="w", pady=(8, 0))

        def run():
            ordered = self.noir_ordered.get()
            equal = self.noir_equal.get()
            zero = self.noir_zero.get()
            if ordered == "No":
                ans = "Nominal"
            elif equal == "No":
                ans = "Ordinal"
            elif zero == "No":
                ans = "Interval"
            else:
                ans = "Ratio"
            res = (
                f"Ordered? {ordered}\n"
                f"Equal intervals? {equal}\n"
                f"True zero means none? {zero}\n\n"
                f"NOIR type = {ans}"
            )
            append_result(out, res)

        ttk.Button(left, text="Execute / Calculate NOIR", command=run).pack(anchor="w", pady=10)
        out = self._new_result_box(right)

    def _tab_data_prep(self):
        tab = self._create_page("13. Data Prep")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Apply common preprocessing steps: one-hot encoding, linear scaling, z-score standardization, and mean imputation.",
            "Enter the raw numeric matrix X for numeric preprocessing. Enter raw categorical labels for one-hot encoding.",
            "Use NaN or NA for missing values."
        )

        frame_x, self.prep_X = labeled_scrolled_text(left, "Numeric matrix X (use NaN/NA for missing values)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.prep_X, "1, 10, NaN\n2, 20, 30\n3, 30, 60")

        frame_lbl, self.prep_labels = labeled_scrolled_text(left, "Categorical labels for one-hot encoding (one per row or comma-separated)")
        frame_lbl.pack(fill="both", expand=True)
        set_text(self.prep_labels, "red\nyellow\ngreen\nred")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=6)

        def one_hot_run():
            try:
                labels = parse_label_list(self.prep_labels.get("1.0", tk.END))
                Y, classes = one_hot_encode(labels)
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Number of raw labels = {len(labels)}",
                        f"Number of discovered classes = {len(classes)}",
                        describe_target_matrix("One-hot matrix", Y),
                    ),
                    f"Classes = {classes}\n\nOne-hot =\n{format_array(Y)}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        def scale_run():
            try:
                X = parse_numeric_matrix(self.prep_X.get("1.0", tk.END))
                Xs, mins, maxs = linear_scale(X)
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Input X", X),
                        describe_array("mins", mins),
                        describe_array("maxs", maxs),
                        describe_design_matrix("Scaled X", Xs),
                    ),
                    f"mins = {format_array(mins)}\nmaxs = {format_array(maxs)}\n\nScaled X =\n{format_array(Xs)}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        def zscore_run():
            try:
                X = parse_numeric_matrix(self.prep_X.get("1.0", tk.END))
                Xz, means, stds = zscore_standardize(X)
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Input X", X),
                        describe_array("means", means),
                        describe_array("stds", stds),
                        describe_design_matrix("Z-score X", Xz),
                    ),
                    f"means = {format_array(means)}\nstds = {format_array(stds)}\n\nZ-score X =\n{format_array(Xz)}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        def impute_run():
            try:
                X = parse_numeric_matrix(self.prep_X.get("1.0", tk.END))
                Xi, means = mean_impute(X)
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Input X", X),
                        describe_array("Column means", means),
                        describe_design_matrix("Imputed X", Xi),
                    ),
                    f"Column means used for imputation = {format_array(means)}\n\nImputed X =\n{format_array(Xi)}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(btns, text="Execute One-hot Encode", command=one_hot_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Execute Linear Scale [0,1]", command=scale_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Execute Z-score", command=zscore_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Execute Mean Impute", command=impute_run).pack(side="left", padx=4)
        out = self._new_result_box(right)

    def _tab_correlation(self):
        tab = self._create_page("14. Pearson Correlation")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute the Pearson correlation coefficient r between two variables x and y.",
            "Enter matching x and y values with the same number of samples.",
            "This measures linear association, not causation."
        )

        frame_x, self.corr_x = labeled_scrolled_text(left, "x values")
        frame_x.pack(fill="both", expand=True)
        set_text(self.corr_x, "1, 2, 3, 4, 5")

        frame_y, self.corr_y = labeled_scrolled_text(left, "y values")
        frame_y.pack(fill="both", expand=True)
        set_text(self.corr_y, "2, 4, 6, 8, 10")

        def run():
            try:
                x = parse_numeric_matrix(self.corr_x.get("1.0", tk.END)).reshape(-1)
                y = parse_numeric_matrix(self.corr_y.get("1.0", tk.END)).reshape(-1)
                ensure_no_missing_or_infinite(x, "x")
                ensure_no_missing_or_infinite(y, "y")
                r = pearson_r(x, y)
                strength = (
                    "strong" if abs(r) > 0.9 else
                    "medium" if abs(r) > 0.7 else
                    "weak" if abs(r) > 0.5 else
                    "none / doubtful"
                )
                direction = "positive" if r > 0 else "negative" if r < 0 else "no"
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_array("x", x),
                        describe_array("y", y),
                        "Each pair (x_i, y_i) contributes one observation pair to the correlation calculation.",
                    ),
                    f"Pearson r = {r}\n\nDirection = {direction}\nStrength = {strength}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate r", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)



    def _tab_bayes_pmf(self):
        tab = self._create_page("15. Bayes / PMF / Normal")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Bayes: P(A|B) = P(B|A)P(A)/P(B).  Normal: compute P(a <= X <= b), P(X <= b), or P(X >= a).  PMF: solve unknown probability entries using the sum rule and, if needed, E[X].",
            "For Bayes, enter P(A), P(B|A), and P(B). For Normal, enter μ, σ, and one or two bounds. For PMF, enter x-values and their probabilities, with up to two unknowns marked as k or ?.",
            "PMF x-values must be unique. If two probabilities are unknown, you must also provide E[X]. Duplicate event values are removed before summing."
        )
    
        bayes_box = ttk.LabelFrame(left, text="Bayes' Rule", padding=10)
        bayes_box.pack(fill="x", pady=(0, 10))
        self.bayes_pa = tk.StringVar(value="0.4")
        self.bayes_pbgivena = tk.StringVar(value="0.75")
        self.bayes_pb = tk.StringVar(value="0.45")
        for r, (label, var) in enumerate([
            ("P(A)", self.bayes_pa),
            ("P(B|A)", self.bayes_pbgivena),
            ("P(B)", self.bayes_pb),
        ]):
            ttk.Label(bayes_box, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(bayes_box, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=6)
    
        normal_box = ttk.LabelFrame(left, text="Normal Probability", padding=10)
        normal_box.pack(fill="x", pady=(0, 10))
        self.norm_mean = tk.StringVar(value="30")
        self.norm_std = tk.StringVar(value="1.8")
        self.norm_lower = tk.StringVar(value="28")
        self.norm_upper = tk.StringVar(value="33")
        for r, (label, var) in enumerate([
            ("Mean μ", self.norm_mean),
            ("Std σ", self.norm_std),
            ("Lower bound a (blank allowed)", self.norm_lower),
            ("Upper bound b (blank allowed)", self.norm_upper),
        ]):
            ttk.Label(normal_box, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(normal_box, textvariable=var, width=16).grid(row=r, column=1, sticky="w", padx=6)
    
        pmf_box = ttk.LabelFrame(left, text="PMF / unknown k / expectation", padding=10)
        pmf_box.pack(fill="both", expand=True)
        ttk.Label(pmf_box, text="x values").pack(anchor="w")
        self.pmf_x = ScrolledText(pmf_box, height=4, width=40)
        self.pmf_x.pack(fill="x")
        set_text(self.pmf_x, "1, 2, 3, 4, 5")
    
        ttk.Label(pmf_box, text="p(x) values (use k or ? for up to two unknowns)").pack(anchor="w", pady=(8, 0))
        self.pmf_p = ScrolledText(pmf_box, height=4, width=40)
        self.pmf_p.pack(fill="x")
        set_text(self.pmf_p, "0.1, ?, 0.2, 0.4, ?")
    
        exp_row = ttk.Frame(pmf_box)
        exp_row.pack(fill="x", pady=(8, 0))
        ttk.Label(exp_row, text="Expected value E[X] (optional, required if 2 unknowns):").pack(side="left")
        self.pmf_expected = tk.StringVar(value="3.5")
        ttk.Entry(exp_row, textvariable=self.pmf_expected, width=12).pack(side="left", padx=6)
    
        ttk.Label(pmf_box, text="Event x-values to sum, e.g. odd values: 1,3,5").pack(anchor="w", pady=(8, 0))
        self.pmf_event = ttk.Entry(pmf_box)
        self.pmf_event.pack(fill="x")
        self.pmf_event.insert(0, "1, 3, 5")
    
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
    
        def bayes_run():
            try:
                pa = float(self.bayes_pa.get())
                pbgivena = float(self.bayes_pbgivena.get())
                pb = float(self.bayes_pb.get())
                validate_probability(pa, "P(A)")
                validate_probability(pbgivena, "P(B|A)")
                validate_probability(pb, "P(B)")
                if pb == 0:
                    raise ValueError("P(B) cannot be zero.")
                posterior = pbgivena * pa / pb
                if posterior > 1 + 1e-12:
                    raise ValueError("These probabilities are inconsistent because the computed posterior exceeds 1.")
                posterior = min(max(posterior, 0.0), 1.0)
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        "Scalar probability inputs used = 3: P(A), P(B|A), P(B)",
                        "Unknown scalar posterior solved = 1: P(A|B)",
                    ),
                    f"P(A|B) = P(B|A) P(A) / P(B) = {posterior}",
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        def normal_run():
            try:
                mean = float(self.norm_mean.get().strip())
                std = float(self.norm_std.get().strip())
                lower_text = self.norm_lower.get().strip()
                upper_text = self.norm_upper.get().strip()
    
                if not lower_text and not upper_text:
                    raise ValueError("Enter at least one bound.")
    
                if lower_text and upper_text:
                    a = float(lower_text)
                    b = float(upper_text)
                    if a > b:
                        raise ValueError("Lower bound must be <= upper bound.")
                    prob = normal_cdf(b, mean, std) - normal_cdf(a, mean, std)
                    lines = [
                        f"X ~ N(mean={mean}, std={std})",
                        f"P({a} <= X <= {b}) = {prob}",
                    ]
                elif upper_text:
                    b = float(upper_text)
                    prob = normal_cdf(b, mean, std)
                    lines = [
                        f"X ~ N(mean={mean}, std={std})",
                        f"P(X <= {b}) = {prob}",
                    ]
                else:
                    a = float(lower_text)
                    prob = 1.0 - normal_cdf(a, mean, std)
                    lines = [
                        f"X ~ N(mean={mean}, std={std})",
                        f"P(X >= {a}) = {prob}",
                    ]
    
                append_result(out, "\n\n".join([
                    section_block(
                        "Parameter / Dimension Summary",
                        "Normal model scalar inputs used = mean, std, and one or two bounds",
                        f"Number of bound parameters entered = {1 if (lower_text and not upper_text) or (upper_text and not lower_text) else 2}",
                    ),
                    "\n".join(lines),
                ]))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        def pmf_run():
            try:
                x_vals = parse_numeric_matrix(self.pmf_x.get("1.0", tk.END)).reshape(-1)
                ensure_no_missing_or_infinite(x_vals, "x values")
                x_list = x_vals.astype(float).tolist()
    
                prob_tokens = parse_probability_tokens(self.pmf_p.get("1.0", tk.END))
                exp_text = self.pmf_expected.get().strip()
                expected_value = None if exp_text == "" else float(exp_text)
    
                solved_probs, unknown_idx = solve_pmf_unknowns(x_list, prob_tokens, expected_value)
                pmf = {x: float(p) for x, p in zip(x_list, solved_probs)}
                mean, var = pmf_mean_variance(x_list, solved_probs)
    
                event_tokens = [tok for tok in self.pmf_event.get().replace(',', ' ').split() if tok.strip()]
                event_vals = [float(tok) for tok in event_tokens]
                event_vals_unique = list(dict.fromkeys(event_vals))
                event_prob = sum(pmf.get(v, 0.0) for v in event_vals_unique)
    
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Number of x-values = {len(x_list)}",
                        f"Number of PMF probability slots = {len(solved_probs)}",
                        f"Unknown PMF parameters solved = {len(unknown_idx)}",
                    ),
                    f"PMF = {pmf}"
                ]
                if unknown_idx:
                    lines.append("Solved unknown probability indices (1-based): " + ", ".join(str(i + 1) for i in unknown_idx))
                    for idx in unknown_idx:
                        lines.append(f"p(x={x_list[idx]}) = {solved_probs[idx]}")
                lines.append(f"E[X] = {mean}")
                lines.append(f"Var(X) = {var}")
                if len(event_vals_unique) != len(event_vals):
                    lines.append(
                        f"Duplicate event values were removed before summing: original {event_vals}, unique {event_vals_unique}"
                    )
                if event_vals_unique:
                    lines.append(f"P(event on x in {event_vals_unique}) = {event_prob}")
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(btns, text="Execute Bayes", command=bayes_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Execute Normal Probability", command=normal_run).pack(side="left", padx=4)
        ttk.Button(btns, text="Execute PMF / Expectation", command=pmf_run).pack(side="left", padx=4)
        out = self._new_result_box(right)
    

    def _tab_inverse_type_checker(self):
        tab = self._create_page("15a. Inverse Type Checker")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Determine whether a matrix has an ordinary inverse, a left inverse, a right inverse, or neither.",
            "Enter the matrix X directly.",
            "Use this for left-inverse / right-inverse MCQs and for deciding whether (X^T X)^(-1)X^T or X^T(XX^T)^(-1) applies."
        )
    
        frame_x, self.invcheck_X = labeled_scrolled_text(left, "Matrix X")
        frame_x.pack(fill="both", expand=True)
        set_text(self.invcheck_X, "2, 0, 0\n0, -1, 1")
    
        def run():
            try:
                X = parse_numeric_matrix(self.invcheck_X.get("1.0", tk.END))
                summary = inverse_status_summary(X)
                r = summary["rank"]
    
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                    ),
                    "",
                    f"X shape = {summary['shape']}",
                    f"rank(X) = {r}",
                    "",
                    f"Ordinary inverse exists? {'Yes' if summary['has_inverse'] else 'No'}",
                    f"Left inverse exists? {'Yes' if summary['has_left_inverse'] else 'No'}",
                    f"Right inverse exists? {'Yes' if summary['has_right_inverse'] else 'No'}",
                    "",
                ]
    
                if summary["has_inverse"]:
                    lines.append("Interpretation: X is square and full rank, so it has an ordinary inverse. It also has both a left and right inverse.")
                    lines.append(f"X^(-1) =\n{format_array(summary['inverse_matrix'])}")
                else:
                    if summary["has_left_inverse"]:
                        lines.append("Interpretation: X is tall-or-square with full column rank, so it has a left inverse.")
                        lines.append("Formula: X_left^+ = (X^T X)^(-1) X^T")
                        lines.append(f"Computed left inverse =\n{format_array(summary['left_inverse_matrix'])}")
                    if summary["has_right_inverse"]:
                        lines.append("Interpretation: X is wide-or-square with full row rank, so it has a right inverse.")
                        lines.append("Formula: X_right^+ = X^T (X X^T)^(-1)")
                        lines.append(f"Computed right inverse =\n{format_array(summary['right_inverse_matrix'])}")
                    if not summary["has_left_inverse"] and not summary["has_right_inverse"]:
                        lines.append("Interpretation: X has neither a left inverse nor a right inverse.")
    
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Check Inverses", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_derivative_shape_checker(self):
        tab = self._create_page("15b. Derivative Shape Checker")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Determine the shape of the first derivative for scalar/vector output with respect to scalar/vector input.",
            "Choose whether the function output is scalar or vector, and whether the variable is scalar or vector. If vector, also enter its dimension.",
            "For vector wrt vector, the result is the Jacobian with shape (output dimension) x (input dimension)."
        )
    
        box = ttk.LabelFrame(left, text="Derivative shape inputs", padding=10)
        box.pack(fill="x", pady=(0, 10))
    
        self.deriv_output_kind = tk.StringVar(value="scalar")
        self.deriv_input_kind = tk.StringVar(value="vector")
        self.deriv_output_dim = tk.StringVar(value="2")
        self.deriv_input_dim = tk.StringVar(value="3")
    
        ttk.Label(box, text="Function output type").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Combobox(box, textvariable=self.deriv_output_kind, values=["scalar", "vector"], width=12, state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(box, text="Output dimension b (only if vector)").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.deriv_output_dim, width=10).grid(row=1, column=1, sticky="w", padx=6)
    
        ttk.Label(box, text="Variable type").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(box, textvariable=self.deriv_input_kind, values=["scalar", "vector"], width=12, state="readonly").grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(box, text="Input dimension d (only if vector)").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.deriv_input_dim, width=10).grid(row=3, column=1, sticky="w", padx=6)
    
        def run():
            try:
                output_kind = self.deriv_output_kind.get()
                input_kind = self.deriv_input_kind.get()
                output_dim = None if output_kind == "scalar" else int(self.deriv_output_dim.get().strip())
                input_dim = None if input_kind == "scalar" else int(self.deriv_input_dim.get().strip())
                shape, explanation = derivative_shape_summary(output_kind, input_kind, output_dim, input_dim)
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Output dimension parameter b = {output_dim if output_dim is not None else 1}",
                        f"Input dimension parameter d = {input_dim if input_dim is not None else 1}",
                    ),
                    f"Output type = {output_kind}",
                    f"Input type = {input_kind}",
                    f"Derivative shape = {shape}",
                    "",
                    explanation,
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Check Shape", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_count_probability(self):
        tab = self._create_page("15c. Count / Conditional Probability")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute probabilities and rates from grouped success / failure counts.",
            "Enter records as: Parent group, Subgroup, Total count, Success count. Failure is computed as Total - Success.",
            "This is designed for department/team promotion-style questions. It reports overall probabilities, group success rates, subgroup success rates, and helpful conditional probabilities."
        )
    
        frame_rec, self.countprob_records = labeled_scrolled_text(left, "Records: Parent group, Subgroup, Total count, Success count")
        frame_rec.pack(fill="both", expand=True)
        set_text(
            self.countprob_records,
            "Department A, Team A-1, 200, 150\n"
            "Department A, Team A-2, 100, 50\n"
            "Department B, Team B, 200, 80\n"
            "Department C, Team C-1, 150, 90\n"
            "Department C, Team C-2, 150, 60"
        )
    
        query_box = ttk.LabelFrame(left, text="Optional query values", padding=10)
        query_box.pack(fill="x", pady=(6, 10))
        ttk.Label(query_box, text="Parent-group query value").grid(row=0, column=0, sticky="w", pady=3)
        self.countprob_parent_query = tk.StringVar(value="Department A")
        ttk.Entry(query_box, textvariable=self.countprob_parent_query, width=20).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(query_box, text="Subgroup query value").grid(row=1, column=0, sticky="w", pady=3)
        self.countprob_sub_query = tk.StringVar(value="Team B")
        ttk.Entry(query_box, textvariable=self.countprob_sub_query, width=20).grid(row=1, column=1, sticky="w", padx=6)
    
        def run():
            try:
                records = parse_group_success_records(self.countprob_records.get("1.0", tk.END))
    
                overall_total = sum(total for _, _, total, _ in records)
                overall_success = sum(success for _, _, _, success in records)
                overall_failure = overall_total - overall_success
                if overall_total <= 0:
                    raise ValueError("Overall total count must be positive.")
    
                by_parent = {}
                by_sub = {}
                by_pair = {}
                for parent, subgroup, total, success in records:
                    failure = total - success
                    by_parent.setdefault(parent, {"total": 0, "success": 0, "failure": 0})
                    by_sub.setdefault(subgroup, {"total": 0, "success": 0, "failure": 0})
                    by_parent[parent]["total"] += total
                    by_parent[parent]["success"] += success
                    by_parent[parent]["failure"] += failure
                    by_sub[subgroup]["total"] += total
                    by_sub[subgroup]["success"] += success
                    by_sub[subgroup]["failure"] += failure
                    by_pair[(parent, subgroup)] = {"total": total, "success": success, "failure": failure}
    
                parent_rates = {
                    parent: vals["success"] / vals["total"] if vals["total"] else float("nan")
                    for parent, vals in by_parent.items()
                }
                sub_rates = {
                    sub: vals["success"] / vals["total"] if vals["total"] else float("nan")
                    for sub, vals in by_sub.items()
                }
    
                best_parent = max(parent_rates.items(), key=lambda kv: kv[1])
                best_sub = max(sub_rates.items(), key=lambda kv: kv[1])
    
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Number of records entered = {len(records)}",
                        f"Number of parent groups = {len(by_parent)}",
                        f"Number of subgroups = {len(by_sub)}",
                        f"Overall population count = {overall_total}",
                    ),
                    f"Overall total = {overall_total}",
                    f"Overall success = {overall_success}",
                    f"Overall failure = {overall_failure}",
                    f"P(success) = {overall_success / overall_total:.6f}",
                    f"P(failure) = {overall_failure / overall_total:.6f}",
                    "",
                    "Parent-group summary:",
                ]
                for parent in sorted(by_parent):
                    vals = by_parent[parent]
                    rate = vals["success"] / vals["total"] if vals["total"] else float("nan")
                    lines.append(
                        f"{parent}: total={vals['total']}, success={vals['success']}, failure={vals['failure']}, success rate={rate:.6f}"
                    )
    
                lines.append("")
                lines.append("Subgroup summary:")
                for subgroup in sorted(by_sub):
                    vals = by_sub[subgroup]
                    rate = vals["success"] / vals["total"] if vals["total"] else float("nan")
                    lines.append(
                        f"{subgroup}: total={vals['total']}, success={vals['success']}, failure={vals['failure']}, success rate={rate:.6f}"
                    )
    
                lines.append("")
                lines.append("Pair summary:")
                for parent, subgroup in sorted(by_pair):
                    vals = by_pair[(parent, subgroup)]
                    rate = vals["success"] / vals["total"] if vals["total"] else float("nan")
                    lines.append(
                        f"{parent} / {subgroup}: total={vals['total']}, success={vals['success']}, failure={vals['failure']}, success rate={rate:.6f}"
                    )
    
                lines.append("")
                lines.append(f"Highest parent-group success rate = {best_parent[0]} ({best_parent[1]:.6f})")
                lines.append(f"Highest subgroup success rate = {best_sub[0]} ({best_sub[1]:.6f})")
    
                parent_query = self.countprob_parent_query.get().strip()
                if parent_query:
                    if parent_query in by_parent:
                        vals = by_parent[parent_query]
                        lines.append("")
                        lines.append(f"Query for parent group '{parent_query}':")
                        lines.append(f"P({parent_query}) = {vals['total'] / overall_total:.6f}")
                        if overall_success > 0:
                            lines.append(f"P({parent_query} | success) = {vals['success'] / overall_success:.6f}")
                        if overall_failure > 0:
                            lines.append(f"P({parent_query} | failure) = {vals['failure'] / overall_failure:.6f}")
                            lines.append(f"P(parent != {parent_query} | failure) = {(overall_failure - vals['failure']) / overall_failure:.6f}")
                        lines.append(f"P(success | {parent_query}) = {vals['success'] / vals['total']:.6f}")
                        lines.append(f"P(failure | {parent_query}) = {vals['failure'] / vals['total']:.6f}")
                    else:
                        lines.append("")
                        lines.append(f"Parent-group query '{parent_query}' was not found.")
    
                sub_query = self.countprob_sub_query.get().strip()
                if sub_query:
                    if sub_query in by_sub:
                        vals = by_sub[sub_query]
                        lines.append("")
                        lines.append(f"Query for subgroup '{sub_query}':")
                        lines.append(f"P({sub_query}) = {vals['total'] / overall_total:.6f}")
                        if overall_success > 0:
                            lines.append(f"P({sub_query} | success) = {vals['success'] / overall_success:.6f}")
                        if overall_failure > 0:
                            lines.append(f"P({sub_query} | failure) = {vals['failure'] / overall_failure:.6f}")
                            lines.append(f"P(subgroup != {sub_query} | failure) = {(overall_failure - vals['failure']) / overall_failure:.6f}")
                        lines.append(f"P(success | {sub_query}) = {vals['success'] / vals['total']:.6f}")
                        lines.append(f"P(failure | {sub_query}) = {vals['failure'] / vals['total']:.6f}")
                    else:
                        lines.append("")
                        lines.append(f"Subgroup query '{sub_query}' was not found.")
    
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Compute Probabilities", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_sequential_probability(self):
        tab = self._create_page("15d. Sequential Probability")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute the probability of a draw sequence from category counts, with or without replacement.",
            "Enter category counts as lines like 'Queen, 4' or 'Not Queen, 48'. Enter the desired draw sequence as labels separated by commas or new lines.",
            "If 'order does not matter' is ticked, the toolkit sums over all distinct permutations of the requested multiset of labels."
        )
    
        frame_counts, self.seq_counts = labeled_scrolled_text(left, "Category counts")
        frame_counts.pack(fill="both", expand=True)
        set_text(self.seq_counts, "Queen, 4\nNot Queen, 48")
    
        frame_seq, self.seq_targets = labeled_scrolled_text(left, "Desired draw sequence")
        frame_seq.pack(fill="both", expand=True)
        set_text(self.seq_targets, "Queen, Queen")
    
        self.seq_with_replacement = tk.BooleanVar(value=False)
        self.seq_ignore_order = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="With replacement", variable=self.seq_with_replacement).pack(anchor="w", pady=(2, 2))
        ttk.Checkbutton(left, text="Order does NOT matter (sum over distinct permutations)", variable=self.seq_ignore_order).pack(anchor="w", pady=(0, 6))
    
        def run():
            try:
                counts = parse_named_counts(self.seq_counts.get("1.0", tk.END))
                seq_tokens = [tok.strip() for tok in self.seq_targets.get("1.0", tk.END).replace(';', '\n').replace(',', '\n').splitlines() if tok.strip()]
                if not seq_tokens:
                    raise ValueError("Draw sequence is empty.")
                with_replacement = self.seq_with_replacement.get()
                ignore_order = self.seq_ignore_order.get()
    
                if ignore_order:
                    total_prob, details = unordered_sequence_probability(counts, seq_tokens, with_replacement)
                    lines = [
                        section_block(
                            "Parameter / Dimension Summary",
                            f"Number of categories entered = {len(counts)}",
                            f"Total population size = {sum(count for _, count in counts)}",
                            f"Requested draw length = {len(seq_tokens)}",
                            f"Distinct permutations summed = {len(details)}",
                        ),
                        f"Counts = {counts}",
                        f"Requested multiset of draws = {seq_tokens}",
                        f"With replacement = {with_replacement}",
                        f"Order matters = False",
                        "",
                        f"Total probability = {total_prob:.10f}",
                        "",
                        "Distinct order contributions:",
                    ]
                    for order, prob in details:
                        lines.append(f"{list(order)}: {prob:.10f}")
                else:
                    total_prob, steps = exact_sequence_probability(counts, seq_tokens, with_replacement)
                    lines = [
                        section_block(
                            "Parameter / Dimension Summary",
                            f"Number of categories entered = {len(counts)}",
                            f"Total population size = {sum(count for _, count in counts)}",
                            f"Requested draw length = {len(seq_tokens)}",
                        ),
                        f"Counts = {counts}",
                        f"Requested sequence = {seq_tokens}",
                        f"With replacement = {with_replacement}",
                        f"Order matters = True",
                        "",
                        "Step-by-step:",
                    ]
                    lines.extend(steps)
                    lines.append("")
                    lines.append(f"Total probability = {total_prob:.10f}")
    
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")
    
        ttk.Button(left, text="Execute / Compute Sequence Probability", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_tutorial_concepts(self):
        tab = self._create_page("16. Tutorial Concepts / MCQ Guide")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "This page is a concept helper for tutorial-style and midterm-style theory / MCQ / MRQ questions, plus polynomial parameter counting.",
            "Choose either a concept topic or a midterm-style question archetype for a compact answer pattern, or enter d, p, and N to check polynomial size and system type.",
            "Use this tab when the question is conceptual rather than purely computational."
        )

        ttk.Label(
            left,
            text=(
                "Quick-reference explanations for common tutorial concept questions and actual midterm question styles, "
                "plus a polynomial parameter counter for model-size and under/even/over-determined checks."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        topic_box = ttk.LabelFrame(left, text="Concept quick answers", padding=10)
        topic_box.pack(fill="x", pady=(0, 10))
        ttk.Label(topic_box, text="Topic").grid(row=0, column=0, sticky="w")
        self.concept_topic = tk.StringVar(value=list(TUTORIAL_CONCEPTS.keys())[0])
        ttk.Combobox(
            topic_box,
            textvariable=self.concept_topic,
            values=list(TUTORIAL_CONCEPTS.keys()),
            width=38,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        midterm_box = ttk.LabelFrame(left, text="Midterm-style quick answers", padding=10)
        midterm_box.pack(fill="x", pady=(0, 10))
        ttk.Label(midterm_box, text="Question style").grid(row=0, column=0, sticky="w")
        self.midterm_topic = tk.StringVar(value=list(MIDTERM_STYLE_GUIDE.keys())[0])
        ttk.Combobox(
            midterm_box,
            textvariable=self.midterm_topic,
            values=list(MIDTERM_STYLE_GUIDE.keys()),
            width=38,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        param_box = ttk.LabelFrame(left, text="Polynomial parameter / determination checker", padding=10)
        param_box.pack(fill="x", pady=(0, 10))
        self.param_d = tk.StringVar(value="2")
        self.param_degree = tk.StringVar(value="3")
        self.param_samples = tk.StringVar(value="3")

        for r, (label, var) in enumerate([
            ("Number of input features d", self.param_d),
            ("Polynomial degree p", self.param_degree),
            ("Number of samples N", self.param_samples),
        ]):
            ttk.Label(param_box, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(param_box, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=6)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)

        def show_concept():
            topic = self.concept_topic.get()
            answer = TUTORIAL_CONCEPTS.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        def show_midterm_pattern():
            topic = self.midterm_topic.get()
            answer = MIDTERM_STYLE_GUIDE.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        def check_params():
            try:
                d = int(self.param_d.get().strip())
                degree = int(self.param_degree.get().strip())
                n_samples = int(self.param_samples.get().strip())
                if n_samples < 1:
                    raise ValueError("Number of samples must be at least 1.")

                n_terms = count_full_polynomial_terms(d, degree)
                if n_samples < n_terms:
                    system_type = "under-determined"
                elif n_samples == n_terms:
                    system_type = "even-determined"
                else:
                    system_type = "over-determined"

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Input feature count d = {d}",
                        f"Polynomial degree p = {degree}",
                        f"Learnable polynomial parameters including bias = {n_terms}",
                        f"Sample count N = {n_samples}",
                    ),
                    f"Full polynomial parameters including bias = C(d+p, p) = C({d}+{degree}, {degree}) = {n_terms}",
                    f"With N = {n_samples} samples, the learning system is {system_type}.",
                    "",
                    "Notes:",
                    "- Unique exact solutions without regularization still depend on rank.",
                    "- Ridge regression with lambda > 0 makes the primal system invertible in the usual full-column setting.",
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(btns, text="Show Concept Answer", command=show_concept).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Midterm Pattern", command=show_midterm_pattern).pack(side="left", padx=4)
        ttk.Button(btns, text="Check Polynomial Size / System Type", command=check_params).pack(side="left", padx=4)
        out = self._new_result_box(right)

    def _tab_midterm_open_ended(self):
        tab = self._create_page("16b. Midterm Coverage / Open-Ended")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "This page covers the open-ended and structured story-question styles that appeared in the repo midterms.",
            "Choose a midterm paper scenario, or use the smaller helpers for learning paradigm, variable type, and preprocessing recommendation.",
            "Use this when the question is phrased as a real-world story instead of a direct formula prompt."
        )

        ttk.Label(
            left,
            text=(
                "This is a direct exam-reference page for the midterm papers in the repo. "
                "It maps actual story-question styles to compact answers and points you to the exact calculator tab for the numerical part."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        paper_box = ttk.LabelFrame(left, text="Midterm paper scenario coverage", padding=10)
        paper_box.pack(fill="x", pady=(0, 10))
        ttk.Label(paper_box, text="Scenario").grid(row=0, column=0, sticky="w")
        self.midterm_case = tk.StringVar(value=list(MIDTERM_PAPER_COVERAGE.keys())[0])
        ttk.Combobox(
            paper_box,
            textvariable=self.midterm_case,
            values=list(MIDTERM_PAPER_COVERAGE.keys()),
            width=42,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        learning_box = ttk.LabelFrame(left, text="Learning paradigm helper", padding=10)
        learning_box.pack(fill="x", pady=(0, 10))
        ttk.Label(learning_box, text="Scenario type").grid(row=0, column=0, sticky="w")
        self.learning_case = tk.StringVar(value=list(SCENARIO_LEARNING_GUIDE.keys())[0])
        ttk.Combobox(
            learning_box,
            textvariable=self.learning_case,
            values=list(SCENARIO_LEARNING_GUIDE.keys()),
            width=42,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        variable_box = ttk.LabelFrame(left, text="Variable type / encoding helper", padding=10)
        variable_box.pack(fill="x", pady=(0, 10))
        ttk.Label(variable_box, text="Variable pattern").grid(row=0, column=0, sticky="w")
        self.variable_case = tk.StringVar(value=list(VARIABLE_ENCODING_GUIDE.keys())[0])
        ttk.Combobox(
            variable_box,
            textvariable=self.variable_case,
            values=list(VARIABLE_ENCODING_GUIDE.keys()),
            width=42,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        prep_box = ttk.LabelFrame(left, text="Preprocessing helper", padding=10)
        prep_box.pack(fill="x", pady=(0, 10))
        ttk.Label(prep_box, text="Situation").grid(row=0, column=0, sticky="w")
        self.prep_case = tk.StringVar(value=list(PREPROCESSING_SCENARIO_GUIDE.keys())[0])
        ttk.Combobox(
            prep_box,
            textvariable=self.prep_case,
            values=list(PREPROCESSING_SCENARIO_GUIDE.keys()),
            width=42,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(
            left,
            text=(
                "High-yield open-ended coverage now includes:\n"
                "- supervised / unsupervised / reinforcement identification\n"
                "- classification vs regression vs clustering vs anomaly detection\n"
                "- nominal / ordinal / interval / ratio recognition\n"
                "- one-hot vs scaling vs standardization vs cleaning\n"
                "- direct mappings for the story questions from the repo midterms"
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        out = self._new_result_box(right)

        def show_paper_case():
            topic = self.midterm_case.get()
            answer = MIDTERM_PAPER_COVERAGE.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        def show_learning_case():
            topic = self.learning_case.get()
            answer = SCENARIO_LEARNING_GUIDE.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        def show_variable_case():
            topic = self.variable_case.get()
            answer = VARIABLE_ENCODING_GUIDE.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        def show_prep_case():
            topic = self.prep_case.get()
            answer = PREPROCESSING_SCENARIO_GUIDE.get(topic, "No explanation available.")
            append_result(out, f"{topic}\n\n{answer}")

        ttk.Button(btns, text="Show Midterm Scenario", command=show_paper_case).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Learning Type", command=show_learning_case).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Variable Type", command=show_variable_case).pack(side="left", padx=4)
        ttk.Button(btns, text="Show Preprocessing", command=show_prep_case).pack(side="left", padx=4)

    def _tab_feature_selection(self):
        tab = self._create_page("17. Pearson Feature Selection")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Rank features by the magnitude of their Pearson correlation with the target: |r(x_j, y)|.",
            "Enter X with samples as rows and features as columns, and enter y as the target vector.",
            "The largest |r| values are ranked as the strongest linear feature candidates."
        )

        frame_x, self.fs_X = labeled_scrolled_text(left, "Feature matrix X (rows = samples, cols = features)")
        frame_x.pack(fill="both", expand=True)
        set_text(
            self.fs_X,
            "0.3510, 1.1796, -0.9852\n"
            "2.1812, 2.1068, 1.3766\n"
            "0.2415, 1.7753, -1.3244\n"
            "-0.1096, 1.2747, -0.6316\n"
            "0.1544, 2.0851, -0.8320"
        )

        frame_y, self.fs_y = labeled_scrolled_text(left, "Target y")
        frame_y.pack(fill="both", expand=True)
        set_text(self.fs_y, "0.2758\n1.4392\n-0.4611\n0.6154\n1.0006")

        options = ttk.Frame(left)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Select top-k features by |r|:").pack(side="left")
        self.fs_topk = tk.StringVar(value="2")
        ttk.Entry(options, textvariable=self.fs_topk, width=8).pack(side="left", padx=6)
        ttk.Label(options, text="Threshold on |r| (optional):").pack(side="left", padx=(12, 0))
        self.fs_thresh = tk.StringVar(value="")
        ttk.Entry(options, textvariable=self.fs_thresh, width=8).pack(side="left", padx=6)

        def run():
            try:
                X = parse_numeric_matrix(self.fs_X.get("1.0", tk.END))
                y = parse_numeric_matrix(self.fs_y.get("1.0", tk.END)).reshape(-1)
                ensure_no_missing_or_infinite(X, "X")
                ensure_no_missing_or_infinite(y, "y")

                if X.shape[0] != y.shape[0]:
                    raise ValueError("X and y must have the same number of samples.")
                topk = int(self.fs_topk.get().strip())
                if topk < 1:
                    raise ValueError("top-k must be at least 1.")
                topk = min(topk, X.shape[1])

                results = []
                for j in range(X.shape[1]):
                    r = pearson_r(X[:, j], y)
                    results.append((j, r, abs(r)))

                results.sort(key=lambda item: item[2], reverse=True)
                selected = results[:topk]
                thresh_text = clean_text(self.fs_thresh.get())
                thresh_selected = []
                if thresh_text:
                    thresh = float(thresh_text)
                    if thresh < 0:
                        raise ValueError("Threshold on |r| must be >= 0.")
                    thresh_selected = [item for item in results if item[2] >= thresh]

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                        describe_target_matrix("y", y.reshape(-1, 1)),
                        f"Candidate feature parameters = {X.shape[1]}",
                    ),
                    "Feature correlations with target y:"
                ]
                for j, r, abs_r in results:
                    lines.append(f"Feature {j + 1}: r = {r:.6f}, |r| = {abs_r:.6f}")

                lines.append("")
                lines.append("Top selected features:")
                for rank, (j, r, abs_r) in enumerate(selected, start=1):
                    lines.append(f"{rank}. Feature {j + 1}  (r = {r:.6f}, |r| = {abs_r:.6f})")

                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Rank Features", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_model_order_compare(self):
        tab = self._create_page("18. Model Order / Overfitting")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "For each polynomial order p, build Φ_p(X), fit the model, then compare train and test MSE across orders.",
            "Enter raw training and test features and targets. The toolkit expands them into polynomial features internally for each tested order.",
            "Use this to spot underfitting, overfitting, and the order with the best test performance."
        )

        ttk.Label(
            left,
            text=(
                "Paste raw training and test data, then compare polynomial model orders. "
                "This is useful for Tutorial 7 style questions about train/test MSE, best order, and overfitting."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        frame_xtr, self.mo_xtr = labeled_scrolled_text(left, "Training X (one sample per row)")
        frame_xtr.pack(fill="both", expand=True)
        set_text(self.mo_xtr, "-3\n-2\n-1\n0\n1\n2\n3")

        frame_ytr, self.mo_ytr = labeled_scrolled_text(left, "Training y")
        frame_ytr.pack(fill="both", expand=True)
        set_text(self.mo_ytr, "4.3\n1.9\n0.8\n0.2\n0.9\n2.1\n4.6")

        frame_xte, self.mo_xte = labeled_scrolled_text(left, "Test X (one sample per row)")
        frame_xte.pack(fill="both", expand=True)
        set_text(self.mo_xte, "-2.5\n-1.5\n-0.5\n0.5\n1.5\n2.5")

        frame_yte, self.mo_yte = labeled_scrolled_text(left, "Test y")
        frame_yte.pack(fill="both", expand=True)
        set_text(self.mo_yte, "2.8\n1.3\n0.3\n0.4\n1.5\n3.1")

        options = ttk.Frame(left)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Order start").pack(side="left")
        self.mo_start = tk.StringVar(value="1")
        ttk.Entry(options, textvariable=self.mo_start, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(options, text="Order end").pack(side="left")
        self.mo_end = tk.StringVar(value="6")
        ttk.Entry(options, textvariable=self.mo_end, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(options, text="Ridge λ").pack(side="left")
        self.mo_lambda = tk.StringVar(value="0")
        ttk.Entry(options, textvariable=self.mo_lambda, width=8).pack(side="left", padx=4)

        def run():
            try:
                Xtr = parse_numeric_matrix(self.mo_xtr.get("1.0", tk.END))
                Ytr = parse_numeric_matrix(self.mo_ytr.get("1.0", tk.END))
                Xte = parse_numeric_matrix(self.mo_xte.get("1.0", tk.END))
                Yte = parse_numeric_matrix(self.mo_yte.get("1.0", tk.END))

                ensure_no_missing_or_infinite(Xtr, "Training X")
                ensure_no_missing_or_infinite(Ytr, "Training y")
                ensure_no_missing_or_infinite(Xte, "Test X")
                ensure_no_missing_or_infinite(Yte, "Test y")

                if Xtr.shape[0] != Ytr.shape[0]:
                    raise ValueError("Training X and y row counts must match.")
                if Xte.shape[0] != Yte.shape[0]:
                    raise ValueError("Test X and y row counts must match.")
                if Xtr.shape[1] != Xte.shape[1]:
                    raise ValueError("Training and test X must have the same number of feature columns.")

                start_order = int(self.mo_start.get().strip())
                end_order = int(self.mo_end.get().strip())
                lam = float(self.mo_lambda.get().strip())
                if start_order < 1 or end_order < start_order:
                    raise ValueError("Use valid orders with 1 <= start <= end.")
                if lam < 0:
                    raise ValueError("Ridge lambda must be non-negative.")

                rows = []
                for degree in range(start_order, end_order + 1):
                    Ptr, _ = make_polynomial_features(Xtr, degree)
                    Pte, _ = make_polynomial_features(Xte, degree)
                    W, method = solve_least_squares(Ptr, Ytr, ridge_lambda=lam)
                    train_mse = mse(Ytr, Ptr @ W)
                    test_mse = mse(Yte, Pte @ W)
                    rows.append((degree, Ptr.shape[1], train_mse, test_mse, method))

                best_train = min(rows, key=lambda row: row[2])
                best_test = min(rows, key=lambda row: row[3])

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Training X", Xtr),
                        describe_target_matrix("Training y", Ytr),
                        describe_design_matrix("Test X", Xte),
                        describe_target_matrix("Test y", Yte),
                    ),
                    "degree | parameters | train MSE | test MSE"
                ]
                lines.append("-" * 46)
                for degree, n_params, train_mse, test_mse, _ in rows:
                    lines.append(f"{degree:>6} | {n_params:>10} | {train_mse:>9.6f} | {test_mse:>8.6f}")

                lines.append("")
                lines.append(f"Best training MSE at order {best_train[0]} (MSE = {best_train[2]:.6f})")
                lines.append(f"Best test MSE at order {best_test[0]} (MSE = {best_test[3]:.6f})")

                if best_train[0] > best_test[0]:
                    lines.append("Interpretation: higher-order models fit training data better, but the test set suggests overfitting.")
                elif best_train[0] == best_test[0]:
                    lines.append("Interpretation: the same order is best on both train and test for this dataset.")
                else:
                    lines.append("Interpretation: the test set prefers a higher order than the train set minimum.")

                if lam > 0:
                    lines.append(f"Regularization was enabled with λ = {lam}, which can reduce overfitting by shrinking weights.")

                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Compare Orders", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)



    def _tab_gradient_descent_scalar(self):
        tab = self._create_page("19. Gradient Descent (scalar)")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Minimize a scalar function g(x) = x^p using the update x_(k+1) = x_k - η g'(x_k).",
            "Enter the power p, initial x, learning rate η, and the number of gradient-descent iterations.",
            "This directly covers Tutorial 8 / Lecture 8 style questions such as minimizing x^2 or x^4 and checking the first few iterations."
        )

        controls = ttk.LabelFrame(left, text="Scalar gradient descent inputs", padding=10)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="Power p in g(x)=x^p").grid(row=0, column=0, sticky="w")
        self.gd_power = tk.StringVar(value="4")
        ttk.Entry(controls, textvariable=self.gd_power, width=10).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(controls, text="Initial x₀").grid(row=1, column=0, sticky="w")
        self.gd_x0 = tk.StringVar(value="2")
        ttk.Entry(controls, textvariable=self.gd_x0, width=10).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(controls, text="Learning rate η").grid(row=2, column=0, sticky="w")
        self.gd_eta = tk.StringVar(value="0.1")
        ttk.Entry(controls, textvariable=self.gd_eta, width=10).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(controls, text="Iterations").grid(row=3, column=0, sticky="w")
        self.gd_steps = tk.StringVar(value="4")
        ttk.Entry(controls, textvariable=self.gd_steps, width=10).grid(row=3, column=1, sticky="w", padx=6)

        output_holder = ttk.Frame(right)
        output_holder.pack(fill="both", expand=True)
        out = self._new_result_box(output_holder)
        plot_box = self._new_plot_box(right, "Gradient Descent Graph Preview")

        def run():
            try:
                power = int(self.gd_power.get().strip())
                x0 = float(self.gd_x0.get().strip())
                eta = float(self.gd_eta.get().strip())
                steps = int(self.gd_steps.get().strip())
                result = scalar_power_gradient_descent(power, x0, eta, steps)
                xs = result["x_values"]
                grads = result["grad_values"]
                costs = result["cost_values"]
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        "Unknown variable x is a scalar, so the number of unknown parameters = 1.",
                        "Gradient g'(x) is also a scalar in this one-variable setting.",
                        f"Requested iteration count = {steps}",
                    ),
                    f"Objective: g(x) = x^{power}",
                    f"Gradient: g'(x) = {power}x^{power - 1}" if power != 1 else "Gradient: g'(x) = 1",
                    f"Update rule: x_(k+1) = x_k - η g'(x_k)",
                    f"Initial x₀ = {x0}",
                    f"Learning rate η = {eta}",
                    "",
                    "Iteration table:",
                ]
                for k in range(len(xs)):
                    lines.append(f"k={k}: x={xs[k]:.10g}, g(x)={costs[k]:.10g}, gradient={grads[k]:.10g}")
                fig = self._build_scalar_gd_figure(xs, costs, power)
                self._render_matplotlib_figure(plot_box, fig)
                append_result(out, "\n".join(lines))
            except Exception as exc:
                self._set_plot_message(plot_box, "Graph unavailable because the calculation failed.")
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Calculate", command=run).pack(anchor="w")

    def _tab_exponential_gd(self):
        tab = self._create_page("20. Exponential GD Regression")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Fit the exponential model f(x, w) = exp(-x^T w) by gradient descent on the squared-error objective C(w) = Σ_i (f(x_i, w) - y_i)^2.",
            "Enter either paired x,y rows or separate x and y columns. This tab can normalize x and y, auto-add a bias term, run gradient descent, plot the cost curve, and plot the fitted exponential curve.",
            "This directly covers Tutorial 8 Question 2. Use comparison learning rates to see how the cost history changes when η is too large or too small."
        )

        frame_pairs, self.exp_pairs = labeled_scrolled_text(left, "Paired data rows x, y (optional; overrides separate x/y below)", height=6)
        frame_pairs.pack(fill="both", expand=True)
        set_text(self.exp_pairs, "")

        frame_x, self.exp_x = labeled_scrolled_text(left, "x / year values")
        frame_x.pack(fill="both", expand=True)
        set_text(self.exp_x, "1981\n1985\n1990\n1995\n2000\n2005\n2010\n2015\n2018")
        frame_y, self.exp_y = labeled_scrolled_text(left, "y / expenditure values")
        frame_y.pack(fill="both", expand=True)
        set_text(self.exp_y, "2.2\n2.6\n3.1\n3.7\n4.2\n5.0\n6.0\n7.2\n7.9")

        options = ttk.LabelFrame(left, text="Gradient descent options", padding=10)
        options.pack(fill="x", pady=6)
        ttk.Label(options, text="Main learning rate η").grid(row=0, column=0, sticky="w")
        self.exp_eta = tk.StringVar(value="0.03")
        ttk.Entry(options, textvariable=self.exp_eta, width=10).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(options, text="Compare η values (comma-separated; optional)").grid(row=1, column=0, sticky="w")
        self.exp_compare_etas = tk.StringVar(value="0.1, 0.001")
        ttk.Entry(options, textvariable=self.exp_compare_etas, width=20).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(options, text="Max iterations").grid(row=2, column=0, sticky="w")
        self.exp_steps = tk.StringVar(value="200000")
        ttk.Entry(options, textvariable=self.exp_steps, width=12).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(options, text="Initial w (optional, one row)").grid(row=3, column=0, sticky="w")
        self.exp_initw = tk.StringVar(value="")
        ttk.Entry(options, textvariable=self.exp_initw, width=20).grid(row=3, column=1, sticky="w", padx=6)
        self.exp_use_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Automatically add bias / offset term", variable=self.exp_use_bias).grid(row=4, column=0, columnspan=2, sticky="w")
        self.exp_norm_x = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Normalize x by its max absolute value", variable=self.exp_norm_x).grid(row=5, column=0, columnspan=2, sticky="w")
        self.exp_norm_y = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Normalize y by its max absolute value", variable=self.exp_norm_y).grid(row=6, column=0, columnspan=2, sticky="w")

        range_box = ttk.LabelFrame(left, text="Prediction / plotting range", padding=10)
        range_box.pack(fill="x", pady=(0, 6))
        ttk.Label(range_box, text="x start").grid(row=0, column=0, sticky="w")
        self.exp_xstart = tk.StringVar(value="1981")
        ttk.Entry(range_box, textvariable=self.exp_xstart, width=10).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(range_box, text="x end").grid(row=0, column=2, sticky="w")
        self.exp_xend = tk.StringVar(value="2023")
        ttk.Entry(range_box, textvariable=self.exp_xend, width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(range_box, text="step").grid(row=0, column=4, sticky="w")
        self.exp_xstep = tk.StringVar(value="1")
        ttk.Entry(range_box, textvariable=self.exp_xstep, width=8).grid(row=0, column=5, sticky="w", padx=6)

        output_holder = ttk.Frame(right)
        output_holder.pack(fill="both", expand=True)
        out = self._new_result_box(output_holder)
        cost_plot_box = self._new_plot_box(right, "Exponential GD Cost Plot")
        fit_plot_box = self._new_plot_box(right, "Exponential GD Fit Plot")

        def parse_training_data():
            pairs_text = clean_text(self.exp_pairs.get("1.0", tk.END))
            if pairs_text:
                pairs = parse_numeric_matrix(pairs_text)
                if pairs.shape[1] != 2:
                    raise ValueError("Paired x,y input must have exactly 2 columns: x and y.")
                x_vals = pairs[:, 0]
                y_vals = pairs[:, 1]
            else:
                x_vals = parse_numeric_matrix(self.exp_x.get("1.0", tk.END)).reshape(-1)
                y_vals = parse_numeric_matrix(self.exp_y.get("1.0", tk.END)).reshape(-1)
            ensure_no_missing_or_infinite(x_vals, "x")
            ensure_no_missing_or_infinite(y_vals, "y")
            if x_vals.shape[0] != y_vals.shape[0]:
                raise ValueError("x and y must have the same number of samples.")
            return x_vals, y_vals

        def run():
            try:
                x_vals, y_vals = parse_training_data()
                eta = float(self.exp_eta.get().strip())
                steps = int(self.exp_steps.get().strip())
                init_w = None
                init_w_text = clean_text(self.exp_initw.get())
                if init_w_text:
                    init_w = np.asarray(parse_float_list(init_w_text), dtype=float)
                result = fit_exponential_regression_gd(
                    x_vals,
                    y_vals,
                    eta=eta,
                    num_steps=steps,
                    add_bias=self.exp_use_bias.get(),
                    normalize_x=self.exp_norm_x.get(),
                    normalize_y=self.exp_norm_y.get(),
                    init_w=init_w,
                )
                x_start = float(self.exp_xstart.get().strip())
                x_end = float(self.exp_xend.get().strip())
                x_step = float(self.exp_xstep.get().strip())
                if x_step <= 0:
                    raise ValueError("Plotting step must be > 0.")
                x_plot = np.arange(x_start, x_end + 0.5 * x_step, x_step)
                pred = predict_exponential_regression(result, x_plot)

                histories = [(f"η={eta}", result["iterations"], result["cost_history"])]
                compare_eta_text = clean_text(self.exp_compare_etas.get())
                compare_summaries = []
                if compare_eta_text:
                    for eta_other in parse_float_list(compare_eta_text):
                        compare_res = fit_exponential_regression_gd(
                            x_vals,
                            y_vals,
                            eta=eta_other,
                            num_steps=steps,
                            add_bias=self.exp_use_bias.get(),
                            normalize_x=self.exp_norm_x.get(),
                            normalize_y=self.exp_norm_y.get(),
                            init_w=init_w,
                        )
                        histories.append((f"η={eta_other}", compare_res["iterations"], compare_res["cost_history"]))
                        compare_summaries.append(
                            f"η={eta_other}: final cost={compare_res['cost_history'][-1]:.8g}, steps run={compare_res['num_steps_run']}, diverged={compare_res['diverged']}"
                        )

                self._render_matplotlib_figure(cost_plot_box, self._build_exponential_cost_figure(histories))
                self._render_matplotlib_figure(fit_plot_box, self._build_exponential_fit_figure(x_vals, y_vals, x_plot, pred["yhat_raw"].reshape(-1)))

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_array("Raw x", x_vals.reshape(-1, 1)),
                        describe_target_matrix("Raw y", y_vals.reshape(-1, 1)),
                        describe_design_matrix("Effective X used in GD", result["X_used"]),
                        describe_weight_matrix("w", result["w"]),
                        describe_target_matrix("Training prediction", result["yhat_raw"]),
                        describe_design_matrix("Plot / prediction X", pred["X_query_used"]),
                        describe_target_matrix("Plot / prediction ŷ", pred["yhat_raw"]),
                    ),
                    "Model: f(x, w) = exp(-x^T w)",
                    "Objective: C(w) = Σ_i (f(x_i, w) - y_i)^2",
                    gradient_formula_text("exponential_squared"),
                    f"\nMain η = {eta}",
                    f"Requested max iterations = {steps}",
                    f"Actual iterations run = {result['num_steps_run']}",
                    f"Stop reason = {result['stop_reason']}",
                    f"Used bias term = {result['add_bias']}",
                    f"Normalized x = {result['normalize_x']} with scale {result['x_scale']}",
                    f"Normalized y = {result['normalize_y']} with scale {result['y_scale']}",
                    f"Final w =\n{format_array(result['w'])}",
                    f"Training prediction =\n{format_array(result['yhat_raw'])}",
                    f"Training residual =\n{format_array(result['residual_raw'])}",
                    f"Training SSE = {result['training_sse_raw']}",
                    f"Training MSE = {result['training_mse_raw']}",
                    f"Primary learning-rate cost history starts at {result['cost_history'][0]:.8g} and ends at {result['cost_history'][-1]:.8g}.",
                    f"Predicted values on the requested x-range =\n{format_array(pred['yhat_raw'])}",
                ]
                if compare_summaries:
                    lines.append("\nComparison learning rates:")
                    lines.extend(compare_summaries)
                append_result(out, "\n".join(lines))
            except Exception as exc:
                self._set_plot_message(cost_plot_box, "Plot unavailable because the calculation failed.")
                self._set_plot_message(fit_plot_box, "Plot unavailable because the calculation failed.")
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Fit Exponential Model", command=run).pack(anchor="w")

    def _tab_gradient_formula_builder(self):
        tab = self._create_page("21. Gradient / Loss Builder")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Show the gradient formula for a selected model / loss combination, and optionally evaluate the cost and gradient at a user-supplied X, y, and w.",
            "Enter X, y, and w using the shapes from the formula. For sigmoid, you may also enter β. Optionally enter η to see one gradient-descent update.",
            "This directly covers Tutorial 8 Questions 3 to 5, and it also includes the exponential squared-error gradient from Question 2 for checking your code or manual derivation."
        )

        top = ttk.LabelFrame(left, text="Model / loss selection", padding=10)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Template").grid(row=0, column=0, sticky="w")
        self.grad_model = tk.StringVar(value="linear_quartic")
        ttk.Combobox(top, textvariable=self.grad_model, values=["linear_quartic", "sigmoid_quartic", "relu_squared_quartic", "exponential_squared"], width=24, state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(top, text="β (sigmoid only)").grid(row=1, column=0, sticky="w")
        self.grad_beta = tk.StringVar(value="1")
        ttk.Entry(top, textvariable=self.grad_beta, width=10).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(top, text="η for one GD update (optional)").grid(row=2, column=0, sticky="w")
        self.grad_eta = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.grad_eta, width=10).grid(row=2, column=1, sticky="w", padx=6)

        frame_x, self.grad_X = labeled_scrolled_text(left, "X (rows = samples, cols = parameters)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.grad_X, "1, 1\n1, -1\n1, 0")
        frame_y, self.grad_y = labeled_scrolled_text(left, "y (single target column)")
        frame_y.pack(fill="both", expand=True)
        set_text(self.grad_y, "1\n0\n2")
        frame_w, self.grad_w = labeled_scrolled_text(left, "w (single weight vector)")
        frame_w.pack(fill="both", expand=True)
        set_text(self.grad_w, "0\n0")

        def run():
            try:
                model_kind = self.grad_model.get().strip()
                beta = float(self.grad_beta.get().strip())
                X = parse_numeric_matrix(self.grad_X.get("1.0", tk.END))
                y = parse_numeric_matrix(self.grad_y.get("1.0", tk.END))
                w = parse_numeric_matrix(self.grad_w.get("1.0", tk.END))
                details = evaluate_model_gradient(X, y, w, model_kind, beta=beta)
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                        describe_target_matrix("y", y),
                        describe_weight_matrix("w", w),
                        describe_target_matrix("Model output f(X, w)", details["f"]),
                        describe_weight_matrix("Gradient ∇C(w)", details["gradient"]),
                    ),
                    gradient_formula_text(model_kind),
                    f"\nLinear score z = Xw =\n{format_array(details['z'])}",
                    f"\nModel output f(X, w) =\n{format_array(details['f'])}",
                    f"\nResidual f(X, w) - y =\n{format_array(details['residual'])}",
                    f"\nCost C(w) = {details['cost']}",
                    f"\nGradient ∇C(w) =\n{format_array(details['gradient'])}",
                ]
                eta_text = clean_text(self.grad_eta.get())
                if eta_text:
                    eta = float(eta_text)
                    w_new = np.asarray(w, dtype=float) - eta * np.asarray(details['gradient'], dtype=float)
                    new_details = evaluate_model_gradient(X, y, w_new, model_kind, beta=beta)
                    lines += [
                        f"\nOne-step GD update with η = {eta}:",
                        f"w_new = w - η∇C(w) =\n{format_array(w_new)}",
                        f"\nNew cost C(w_new) = {new_details['cost']}",
                    ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Show Formula / Evaluate Gradient", command=run).pack(anchor="w")
        out = self._new_result_box(right)


    def _tab_exam_style_solver(self):
        tab = self._create_page("22. Exam-Style Solver")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "This tab is designed around the examiner's midterm style: count parameters, check under/even/over-determined status, fit the requested model, extract bias terms, compute training MSE, and optionally predict on Xnew.",
            "Enter raw X, choose linear or full polynomial order, choose regression or multiclass one-hot classification, then decide whether ridge regularization is used.",
            "This directly supports questions like: total parameters across all classes, bias term for class2, training MSE, and the predicted label for a new point."
        )

        frame_x, self.exam_X = labeled_scrolled_text(left, "Raw X (rows = samples)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.exam_X, "2, 1, 0\n0, 3, 1\n1, 0, 3\n3, 1, 4\n-1, 2, 1")

        frame_y, self.exam_y = labeled_scrolled_text(left, "Targets: numeric y or class labels")
        frame_y.pack(fill="both", expand=True)
        set_text(self.exam_y, "class1\nclass3\nclass2\nclass1\nclass2")

        frame_xnew, self.exam_Xnew = labeled_scrolled_text(left, "Xnew (optional)")
        frame_xnew.pack(fill="both", expand=True)
        set_text(self.exam_Xnew, "1, 1, 2")

        options = ttk.LabelFrame(left, text="Exam-style model settings", padding=10)
        options.pack(fill="x", pady=6)

        ttk.Label(options, text="Target mode").grid(row=0, column=0, sticky="w")
        self.exam_target_mode = tk.StringVar(value="multiclass")
        ttk.Combobox(
            options,
            textvariable=self.exam_target_mode,
            values=["regression", "multiclass"],
            width=14,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(options, text="Model family").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.exam_model_family = tk.StringVar(value="polynomial")
        ttk.Combobox(
            options,
            textvariable=self.exam_model_family,
            values=["linear", "polynomial"],
            width=14,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(options, text="Polynomial degree").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.exam_degree = tk.StringVar(value="2")
        ttk.Entry(options, textvariable=self.exam_degree, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        self.exam_use_ridge = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Use ridge regularization", variable=self.exam_use_ridge).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Label(options, text="Lambda").grid(row=1, column=3, sticky="w", pady=(6, 0))
        self.exam_lambda = tk.StringVar(value="0.01")
        ttk.Entry(options, textvariable=self.exam_lambda, width=10).grid(row=1, column=4, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(options, text="Class to inspect bias for (multiclass only)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.exam_bias_class = tk.StringVar(value="class2")
        ttk.Entry(options, textvariable=self.exam_bias_class, width=14).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(
            left,
            text=(
                "Tips:\n"
                "- Linear means a bias column plus the raw features.\n"
                "- Polynomial means the full feature expansion including the constant term.\n"
                "- For multiclass, the toolkit uses one-hot targets and predicts with argmax.\n"
                "- The first row of W is always the bias / constant-term row for this tab."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        out = self._new_result_box(right)

        def run():
            try:
                X = parse_numeric_matrix(self.exam_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")

                model_family = self.exam_model_family.get().strip()
                degree = 1 if model_family == "linear" else int(self.exam_degree.get().strip())
                target_mode = self.exam_target_mode.get().strip()
                use_ridge = self.exam_use_ridge.get()
                lam = float(self.exam_lambda.get().strip()) if use_ridge else 0.0
                if lam < 0:
                    raise ValueError("Lambda must be non-negative.")
                if use_ridge and lam == 0:
                    raise ValueError("When ridge is enabled, lambda should be > 0.")

                if model_family == "linear":
                    P = add_bias_column(X)
                    names = ["1"] + [f"x{i+1}" for i in range(X.shape[1])]
                else:
                    P, names = make_polynomial_features(X, degree)

                xnew_text = clean_text(self.exam_Xnew.get("1.0", tk.END))
                Xnew = None
                Pnew = None
                if xnew_text:
                    Xnew = parse_numeric_matrix(xnew_text)
                    ensure_no_missing_or_infinite(Xnew, "Xnew")
                    if Xnew.shape[1] != X.shape[1]:
                        raise ValueError(f"Xnew must have {X.shape[1]} raw feature columns to match X.")
                    if model_family == "linear":
                        Pnew = add_bias_column(Xnew)
                    else:
                        Pnew, _ = make_polynomial_features(Xnew, degree)

                if target_mode == "regression":
                    Y = parse_numeric_matrix(self.exam_y.get("1.0", tk.END))
                    ensure_no_missing_or_infinite(Y, "y")
                    if Y.shape[0] != X.shape[0]:
                        raise ValueError("Number of target rows must match the number of samples.")
                    W, method = solve_least_squares(P, Y, ridge_lambda=lam)
                    train_scores = P @ W
                    summary = polynomial_model_summary(X, degree, n_outputs=Y.shape[1] if Y.ndim == 2 else 1)
                    lines = [
                        section_block(
                            "Parameter / Dimension Summary",
                            describe_design_matrix("Raw X", X),
                            describe_design_matrix("Effective feature matrix", P),
                            describe_target_matrix("Y", Y),
                            describe_weight_matrix("W", W),
                            describe_target_matrix("Training prediction", train_scores),
                        ),
                        f"Model family = {model_family}",
                        f"Fitted feature names = {names}",
                        f"Method used = {method}",
                        f"Samples N = {summary['n_samples']}",
                        f"Parameters per output = {summary['params_per_output']}",
                        f"Number of outputs = {summary['n_outputs']}",
                        f"Total parameters across all outputs = {summary['total_params']}",
                        f"System type before any regularization = {summary['determination']}",
                        f"Bias / constant-term row =\n{format_array(W[0:1, :])}",
                        f"Training prediction =\n{format_array(train_scores)}",
                        f"Training MSE = {mse(Y, train_scores)}",
                    ]
                    if use_ridge:
                        lines.append(f"Ridge regularization enabled with λ = {lam}. All parameters, including the bias row, are regularized.")
                    else:
                        lines.append("No ridge regularization used.")
                        if summary['determination'] == "Under-determined":
                            lines.append("Because the system is under-determined, the unregularized model may have infinitely many exact solutions depending on rank / consistency.")

                    if Pnew is not None:
                        Ynew = Pnew @ W
                        lines += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Effective feature matrix for Xnew", Pnew),
                                describe_target_matrix("Prediction for Xnew", Ynew),
                            ),
                            f"Prediction for Xnew =\n{format_array(Ynew)}",
                        ]
                    else:
                        lines.append("No Xnew provided.")

                    append_result(out, "\n".join(lines))

                else:
                    labels = parse_label_list(self.exam_y.get("1.0", tk.END))
                    if len(labels) != X.shape[0]:
                        raise ValueError("Number of labels must match the number of samples.")
                    Y, classes = one_hot_encode(labels)
                    W, method = solve_least_squares(P, Y, ridge_lambda=lam)
                    train_scores = P @ W
                    train_idx = np.argmax(train_scores, axis=1)
                    train_labels = [classes[i] for i in train_idx]
                    summary = polynomial_model_summary(X, degree, n_outputs=len(classes))
                    bias_class = self.exam_bias_class.get().strip()

                    lines = [
                        section_block(
                            "Parameter / Dimension Summary",
                            describe_design_matrix("Raw X", X),
                            describe_design_matrix("Effective feature matrix", P),
                            describe_target_matrix("One-hot Y", Y),
                            describe_weight_matrix("W", W),
                            describe_target_matrix("Training raw class scores", train_scores),
                        ),
                        f"Model family = {model_family}",
                        f"Fitted feature names = {names}",
                        f"Classes = {classes}",
                        f"Method used = {method}",
                        f"Samples N = {summary['n_samples']}",
                        f"Parameters per class = {summary['params_per_output']}",
                        f"Number of classes = {summary['n_outputs']}",
                        f"Total parameters across all classes = {summary['total_params']}",
                        f"System type before any regularization = {summary['determination']}",
                        f"Bias / constant-term row for all classes =\n{format_array(W[0:1, :])}",
                        f"Training raw class scores =\n{format_array(train_scores)}",
                        f"Training predicted labels via argmax = {train_labels}",
                        f"Training MSE on one-hot targets = {mse(Y, train_scores)}",
                    ]

                    if bias_class:
                        if bias_class not in classes:
                            lines.append(f"Requested bias class '{bias_class}' was not found in the discovered classes.")
                        else:
                            class_idx = classes.index(bias_class)
                            lines.append(f"Bias / constant-term coefficient for {bias_class} = {float(W[0, class_idx])}")

                    if use_ridge:
                        lines.append(f"Ridge regularization enabled with λ = {lam}. All parameters, including the bias row, are regularized.")
                    else:
                        lines.append("No ridge regularization used.")
                        if summary['determination'] == "Under-determined":
                            lines.append("Because the system is under-determined, the unregularized model may have infinitely many exact solutions. That is why exam questions often add ridge here.")

                    if Pnew is not None:
                        Ynew = Pnew @ W
                        idx = np.argmax(Ynew, axis=1)
                        pred_labels = [classes[i] for i in idx]
                        pred_onehot = np.zeros_like(Ynew)
                        pred_onehot[np.arange(len(idx)), idx] = 1.0
                        lines += [
                            section_block(
                                "Xnew parameter summary",
                                describe_design_matrix("Raw Xnew", Xnew),
                                describe_design_matrix("Effective feature matrix for Xnew", Pnew),
                                describe_target_matrix("Raw class scores for Xnew", Ynew),
                                describe_target_matrix("Predicted one-hot via argmax", pred_onehot),
                            ),
                            f"Raw class scores for Xnew =\n{format_array(Ynew)}",
                            f"Predicted labels via argmax = {pred_labels}",
                            f"Predicted one-hot via argmax =\n{format_array(pred_onehot)}",
                        ]
                    else:
                        lines.append("No Xnew provided.")

                    append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Solve Exam-Style Question", command=run).pack(anchor="w", pady=6)

    # ------------------------------------------------------------------
    # Phase 2 new tabs
    # ------------------------------------------------------------------

    def _tab_metrics_cv(self):
        tab = self._create_page("23. Metrics / Cross-Validation")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute binary confusion-matrix metrics directly from TP/FN/FP/TN, and count / select cross-validation runs.",
            "Use the top section when the question gives confusion counts. Use the bottom section when the question gives fold counts or candidate training/validation scores.",
            "This targets Lecture 10 and Tutorial 10 style questions, especially imbalanced-class traps and k-fold bookkeeping."
        )

        cm_box = ttk.LabelFrame(left, text="Binary confusion-matrix counts", padding=10)
        cm_box.pack(fill="x", pady=(0, 10))
        self.cv_tp = tk.StringVar(value="7")
        self.cv_fn = tk.StringVar(value="7")
        self.cv_fp = tk.StringVar(value="2")
        self.cv_tn = tk.StringVar(value="25")
        for row, (label, var) in enumerate([
            ("TP", self.cv_tp),
            ("FN", self.cv_fn),
            ("FP", self.cv_fp),
            ("TN", self.cv_tn),
        ]):
            ttk.Label(cm_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(cm_box, textvariable=var, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=2)

        cv_box = ttk.LabelFrame(left, text="Cross-validation helper", padding=10)
        cv_box.pack(fill="x", pady=(0, 10))
        ttk.Label(cv_box, text="Number of candidate models").grid(row=0, column=0, sticky="w")
        self.cv_candidates = tk.StringVar(value="3")
        ttk.Entry(cv_box, textvariable=self.cv_candidates, width=10).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(cv_box, text="Number of folds").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.cv_folds = tk.StringVar(value="5")
        ttk.Entry(cv_box, textvariable=self.cv_folds, width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(cv_box, text="Metric direction").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cv_metric_direction = tk.StringVar(value="lower is better")
        ttk.Combobox(
            cv_box,
            textvariable=self.cv_metric_direction,
            values=["lower is better", "higher is better"],
            width=16,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        frame_rows, self.cv_rows = labeled_scrolled_text(
            left,
            "Candidate rows: parameter, training_metric, validation_metric",
            height=7,
        )
        frame_rows.pack(fill="both", expand=True)
        set_text(
            self.cv_rows,
            "10, 0.10, 0.25\n"
            "9, 0.30, 0.35\n"
            "8, 0.22, 0.15\n"
            "7, 0.15, 0.25\n"
            "6, 0.18, 0.15"
        )

        ttk.Label(
            left,
            text=(
                "Exam traps:\n"
                "- For imbalanced classes, accuracy alone can be misleading.\n"
                "- If two candidates have the same validation metric, the simpler model is usually preferred.\n"
                "- Total fitted models in k-fold CV = (number of candidates) x (number of folds)."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        out = self._new_result_box(right)

        def run():
            try:
                tp = int(self.cv_tp.get().strip())
                fn = int(self.cv_fn.get().strip())
                fp = int(self.cv_fp.get().strip())
                tn = int(self.cv_tn.get().strip())
                if min(tp, fn, fp, tn) < 0:
                    raise ValueError("TP, FN, FP, and TN must be non-negative integers.")

                counts_summary = binary_confusion_summary(
                    ["P"] * (tp + fn) + ["N"] * (fp + tn),
                    ["P"] * tp + ["N"] * fn + ["P"] * fp + ["N"] * tn,
                    positive_label="P",
                )

                candidates = int(self.cv_candidates.get().strip())
                folds = int(self.cv_folds.get().strip())
                total_fits = count_cross_validation_fits(candidates, folds)
                rows = []
                for raw_line in self.cv_rows.get("1.0", tk.END).replace(";", "\n").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 3:
                        raise ValueError(f"Expected 3 comma-separated values per row, got: {raw_line}")
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                lower_is_better = self.cv_metric_direction.get().strip().lower().startswith("lower")
                best = choose_best_validation_candidate(rows, lower_is_better=lower_is_better)

                lines = [
                    section_block(
                        "Binary Metric Summary",
                        f"TP = {tp}, FN = {fn}, FP = {fp}, TN = {tn}",
                        f"Total samples = {tp + fn + fp + tn}",
                    ),
                    f"Accuracy = (TP + TN) / total = ({tp} + {tn}) / {tp + fn + fp + tn} = {counts_summary['accuracy']:.6f}",
                    f"Precision = TP / (TP + FP) = {tp} / {tp + fp} = {counts_summary['precision']:.6f}",
                    f"Recall = TP / (TP + FN) = {tp} / {tp + fn} = {counts_summary['recall']:.6f}",
                    f"Specificity = TN / (TN + FP) = {tn} / {tn + fp} = {counts_summary['specificity']:.6f}",
                    f"F1 = 2PR / (P + R) = {counts_summary['f1']:.6f}",
                    f"Balanced accuracy = (Recall + Specificity) / 2 = {counts_summary['balanced_accuracy']:.6f}",
                    "",
                    section_block(
                        "Cross-Validation Summary",
                        f"Candidates = {candidates}",
                        f"Folds = {folds}",
                        f"Total model fits = candidates x folds = {total_fits}",
                    ),
                    "parameter | train metric | validation metric",
                    "-" * 42,
                ]
                for param, train_metric, val_metric in rows:
                    marker = "  <-- selected" if param == best["best_parameter"] else ""
                    lines.append(f"{param:>9g} | {train_metric:>12.6f} | {val_metric:>17.6f}{marker}")
                lines += [
                    "",
                    (
                        f"Best parameter = {best['best_parameter']:g} based on validation metric "
                        f"({best['best_validation_metric']:.6f})."
                    ),
                    "Tie-break used: prefer the simpler / smaller-parameter candidate when validation scores tie.",
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Compute CV & Metrics", command=run).pack(anchor="w", pady=6)

    def _tab_kmeans(self):
        tab = self._create_page("24. K-Means Clustering")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Run k-means clustering on a data matrix X (unsupervised).",
            "Enter X as rows of data points. Optionally enter initial centroids (k rows). If not provided, the first k rows of X are used.",
            "WCSS = Within-Cluster Sum of Squares (the k-means objective). Lower is better. Lecture 11 covers this algorithm."
        )

        frame_x, self.km_X = labeled_scrolled_text(left, "Data matrix X (rows = samples)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.km_X, "1, 2\n1.5, 1.8\n5, 8\n8, 8\n1, 0.6\n9, 11")

        frame_c, self.km_centroids = labeled_scrolled_text(left, "Initial centroids (optional, k rows)")
        frame_c.pack(fill="both", expand=True)
        set_text(self.km_centroids, "")

        frame_labels, self.km_true_labels = labeled_scrolled_text(
            left,
            "True labels (optional, for post-hoc clustering accuracy check)",
            height=4,
        )
        frame_labels.pack(fill="both", expand=True)
        set_text(self.km_true_labels, "")

        opts = ttk.Frame(left)
        opts.pack(fill="x", pady=6)
        ttk.Label(opts, text="k (number of clusters):").pack(side="left")
        self.km_k = tk.StringVar(value="2")
        ttk.Entry(opts, textvariable=self.km_k, width=8).pack(side="left", padx=(4, 12))
        ttk.Label(opts, text="Max iterations:").pack(side="left")
        self.km_maxiter = tk.StringVar(value="100")
        ttk.Entry(opts, textvariable=self.km_maxiter, width=8).pack(side="left", padx=4)

        def run():
            try:
                X = parse_numeric_matrix(self.km_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")
                k = int(self.km_k.get().strip())
                max_iter = int(self.km_maxiter.get().strip())

                init_c = None
                c_text = clean_text(self.km_centroids.get("1.0", tk.END))
                if c_text:
                    init_c = parse_numeric_matrix(c_text)
                    ensure_no_missing_or_infinite(init_c, "Initial centroids")

                result = kmeans_cluster(X, k, max_iter=max_iter, init_centroids=init_c)
                labels = result["labels"]
                centroids = result["centroids"]
                true_label_text = clean_text(self.km_true_labels.get("1.0", tk.END))
                cluster_accuracy = None
                if true_label_text:
                    true_labels = parse_label_list(self.km_true_labels.get("1.0", tk.END))
                    cluster_accuracy = best_clustering_accuracy(true_labels, labels)

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("X", X),
                        f"k (number of clusters) = {k}",
                        f"Iterations run = {result['n_iter']}",
                        f"Final WCSS = {result['wcss']:.6f}",
                    ),
                    f"Final cluster assignments (0-indexed):\n{labels.tolist()}",
                    f"\nFinal centroids =\n{format_array(centroids)}",
                    "",
                    "Step-by-step iteration details:"
                ]
                for step in result["history"]:
                    lines.append(
                        f"Iter {step['iter']}: WCSS={step['wcss']:.6f}  labels={step['labels'].tolist()}"
                    )
                    lines.append(f"  Centroids at start of iter {step['iter']}:\n  {format_array(step['centroids']).replace(chr(10), chr(10)+'  ')}")

                # Per-cluster summary
                lines.append("")
                lines.append("Cluster membership summary:")
                for j in range(k):
                    members = np.where(labels == j)[0].tolist()
                    lines.append(
                        f"Cluster {j}: members = {members}, centroid = {format_array(centroids[j:j+1, :])}"
                    )

                if cluster_accuracy is not None:
                    lines += [
                        "",
                        "Post-hoc clustering accuracy against supplied true labels:",
                        f"Best permutation mapping = {cluster_accuracy['mapping']}",
                        f"Best accuracy = {cluster_accuracy['accuracy']:.6f}",
                        f"Mapped predicted labels = {cluster_accuracy['predicted_labels']}",
                    ]

                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Run K-Means", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_neural_network(self):
        tab = self._create_page("26. Neural Network Forward")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Run a forward pass through 1 to 3 fully connected layers with optional bias augmentation and ReLU / sigmoid / linear activations.",
            "Enter X as rows of samples. Each weight matrix W_l should have one row per incoming feature, plus one extra row if bias augmentation is enabled.",
            "This targets Tutorial 12 style questions asking for layer outputs, final network output, and parameter counts."
        )

        frame_x, self.nn_X = labeled_scrolled_text(left, "Input matrix X (rows = samples, cols = features)")
        frame_x.pack(fill="both", expand=True)
        set_text(self.nn_X, "1, 0\n0, 1\n1, 1")

        frame_w1, self.nn_W1 = labeled_scrolled_text(left, "W1 (required)", height=5)
        frame_w1.pack(fill="both", expand=True)
        set_text(self.nn_W1, "0, 0\n1, 0\n0, 1")

        frame_w2, self.nn_W2 = labeled_scrolled_text(left, "W2 (optional)", height=5)
        frame_w2.pack(fill="both", expand=True)
        set_text(self.nn_W2, "0\n1\n1")

        frame_w3, self.nn_W3 = labeled_scrolled_text(left, "W3 (optional)", height=4)
        frame_w3.pack(fill="both", expand=True)
        set_text(self.nn_W3, "")

        opts = ttk.LabelFrame(left, text="Forward-pass options", padding=10)
        opts.pack(fill="x", pady=6)
        ttk.Label(opts, text="Hidden activation").grid(row=0, column=0, sticky="w")
        self.nn_hidden_activation = tk.StringVar(value="relu")
        ttk.Combobox(
            opts,
            textvariable=self.nn_hidden_activation,
            values=["relu", "sigmoid", "linear"],
            width=12,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(opts, text="Output activation").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.nn_output_activation = tk.StringVar(value="linear")
        ttk.Combobox(
            opts,
            textvariable=self.nn_output_activation,
            values=["linear", "relu", "sigmoid"],
            width=12,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=6)
        self.nn_add_bias = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Auto-add bias column before every layer", variable=self.nn_add_bias).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

        ttk.Label(
            left,
            text=(
                "Shape rule:\n"
                "- If a layer input has shape N x d and bias is enabled, the effective input becomes N x (d+1).\n"
                "- Then W must have shape (d+1) x h for h output units.\n"
                "- Total parameters = sum of all entries across W1, W2, W3."
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        out = self._new_result_box(right)

        def parse_weight_box(widget: tk.Text, name: str) -> Optional[np.ndarray]:
            text = clean_text(widget.get("1.0", tk.END))
            if not text:
                return None
            W = parse_numeric_matrix(text)
            ensure_no_missing_or_infinite(W, name)
            return W

        def run():
            try:
                X = parse_numeric_matrix(self.nn_X.get("1.0", tk.END))
                ensure_no_missing_or_infinite(X, "X")
                weights = []
                for name, widget in [("W1", self.nn_W1), ("W2", self.nn_W2), ("W3", self.nn_W3)]:
                    W = parse_weight_box(widget, name)
                    if W is not None:
                        weights.append(W)
                if not weights:
                    raise ValueError("At least W1 must be provided.")

                result = forward_neural_network(
                    X,
                    weights,
                    hidden_activation=self.nn_hidden_activation.get().strip(),
                    output_activation=self.nn_output_activation.get().strip(),
                    add_bias=self.nn_add_bias.get(),
                )

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        describe_design_matrix("Input X", X),
                        f"Number of layers = {len(result['layers'])}",
                        f"Total parameters across all layers = {result['total_parameters']}",
                    ),
                    f"Hidden activation = {self.nn_hidden_activation.get().strip()}",
                    f"Output activation = {self.nn_output_activation.get().strip()}",
                    f"Bias augmentation enabled = {result['add_bias']}",
                ]

                for layer in result["layers"]:
                    lines += [
                        "",
                        section_block(
                            f"Layer {layer['layer_index']}",
                            describe_design_matrix("Effective input", layer["input_with_bias"]),
                            describe_weight_matrix("W", layer["weights"]),
                            describe_target_matrix("Pre-activation Z", layer["pre_activation"]),
                            describe_target_matrix(f"Output after {layer['activation_name']}", layer["output"]),
                            f"Layer parameter count = {layer['parameter_count']}",
                        ),
                        f"Pre-activation Z{layer['layer_index']} =\n{format_array(layer['pre_activation'])}",
                        f"Output A{layer['layer_index']} =\n{format_array(layer['output'])}",
                    ]

                lines += [
                    "",
                    f"Final network output =\n{format_array(result['final_output'])}",
                    "Prediction note: for multiclass outputs, exam questions often choose the predicted class by argmax of the final score vector.",
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Forward Pass", command=run).pack(anchor="w", pady=6)

    def _tab_decision_tree_mse(self):
        tab = self._create_page("25. Decision Tree Metrics")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute both classification-tree impurity metrics and regression-tree split MSE from Lecture 9 / Tutorial 9 style inputs.",
            "Use the top section for class counts in the root and child nodes. Use the bottom section for a regression-tree x,y split threshold.",
            "This page covers Gini, entropy, misclassification, weighted child impurity, root MSE, child MSE, and split improvement."
        )

        frame_root, self.dt_root_counts = labeled_scrolled_text(
            left,
            "Root node class counts (comma-separated)",
            height=3,
        )
        frame_root.pack(fill="both", expand=True)
        set_text(self.dt_root_counts, "5, 5, 8")

        frame_children, self.dt_child_counts = labeled_scrolled_text(
            left,
            "Child node class counts (one child per line, comma-separated)",
            height=4,
        )
        frame_children.pack(fill="both", expand=True)
        set_text(self.dt_child_counts, "4, 0, 6\n1, 5, 2")

        frame_x, self.dt_x = labeled_scrolled_text(left, "Regression-tree x values", height=4)
        frame_x.pack(fill="both", expand=True)
        set_text(self.dt_x, "1\n0.8\n2\n2.5\n3\n4\n4.2\n6\n6.3\n7\n8\n8.2\n9")

        frame_y, self.dt_y = labeled_scrolled_text(left, "Regression-tree y values", height=4)
        frame_y.pack(fill="both", expand=True)
        set_text(self.dt_y, "2\n3\n2.5\n1\n2.3\n2.8\n1.5\n2.6\n3.5\n4\n3.5\n5\n4.5")

        threshold_row = ttk.Frame(left)
        threshold_row.pack(fill="x", pady=6)
        ttk.Label(threshold_row, text="Split threshold x <= ").pack(side="left")
        self.dt_threshold = tk.StringVar(value="5.0")
        ttk.Entry(threshold_row, textvariable=self.dt_threshold, width=10).pack(side="left", padx=6)

        def run():
            try:
                root_counts = parse_float_list(self.dt_root_counts.get("1.0", tk.END))
                child_counts = []
                for raw_line in self.dt_child_counts.get("1.0", tk.END).replace(";", "\n").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    child_counts.append(parse_float_list(line))

                root_summary = class_impurity_summary(root_counts)
                child_summary = weighted_child_impurity_summaries(child_counts)

                x = parse_numeric_matrix(self.dt_x.get("1.0", tk.END)).reshape(-1)
                y = parse_numeric_matrix(self.dt_y.get("1.0", tk.END)).reshape(-1)
                threshold = float(self.dt_threshold.get().strip())
                reg_summary = regression_tree_split_summary(x, y, threshold)

                lines = [
                    section_block(
                        "Classification-Tree Impurity",
                        f"Root counts = {root_counts}",
                        f"Child node count = {len(child_summary['children'])}",
                        f"Root total = {int(root_summary['total'])}",
                    ),
                    f"Root probabilities = {format_array(root_summary['probabilities'])}",
                    f"Root Gini = {root_summary['gini']:.6f}",
                    f"Root entropy = {root_summary['entropy']:.6f}",
                    f"Root misclassification = {root_summary['misclassification']:.6f}",
                    "",
                    "Child-node details:",
                ]
                for idx, child in enumerate(child_summary["children"], start=1):
                    lines.append(
                        f"Child {idx}: counts = {child['counts'].astype(int).tolist()}, "
                        f"probs = {format_array(child['probabilities'])}, "
                        f"Gini = {child['gini']:.6f}, entropy = {child['entropy']:.6f}, "
                        f"misclassification = {child['misclassification']:.6f}"
                    )
                lines += [
                    "",
                    f"Weighted child Gini = {child_summary['weighted_gini']:.6f}",
                    f"Weighted child entropy = {child_summary['weighted_entropy']:.6f}",
                    f"Weighted child misclassification = {child_summary['weighted_misclassification']:.6f}",
                    "",
                    section_block(
                        "Regression-Tree Split",
                        f"Threshold = {threshold}",
                        f"Left count = {reg_summary['left_count']}",
                        f"Right count = {reg_summary['right_count']}",
                    ),
                    f"Root mean = {reg_summary['root_mean']:.6f}",
                    f"Left mean = {reg_summary['left_mean']:.6f}",
                    f"Right mean = {reg_summary['right_mean']:.6f}",
                    f"Root MSE = {reg_summary['root_mse']:.6f}",
                    f"Left MSE = {reg_summary['left_mse']:.6f}",
                    f"Right MSE = {reg_summary['right_mse']:.6f}",
                    f"Weighted child MSE = {reg_summary['weighted_mse']:.6f}",
                    f"MSE improvement = root MSE - weighted child MSE = {reg_summary['improvement']:.6f}",
                    "",
                    "Interpretation: good tree splits reduce weighted impurity or weighted MSE relative to the parent node.",
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Compute Tree Metrics", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_classification_metrics(self):
        tab = self._create_page("27. Classification Metrics")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute accuracy, error rate, and confusion matrix from true and predicted class labels.",
            "Enter true labels and predicted labels, one per row (or comma-separated). Labels must match in count.",
            "This supports Lecture 10-12 style questions about model evaluation. Rows = true class, columns = predicted class."
        )

        frame_t, self.cm_true = labeled_scrolled_text(left, "True labels (one per row or comma-separated)")
        frame_t.pack(fill="both", expand=True)
        set_text(self.cm_true, "cat\ndog\ncat\nbird\ndog\ncat")

        frame_p, self.cm_pred = labeled_scrolled_text(left, "Predicted labels (one per row or comma-separated)")
        frame_p.pack(fill="both", expand=True)
        set_text(self.cm_pred, "cat\ndog\ndog\nbird\ncat\ncat")

        def run():
            try:
                y_true = parse_label_list(self.cm_true.get("1.0", tk.END))
                y_pred = parse_label_list(self.cm_pred.get("1.0", tk.END))
                result = classification_metrics(y_true, y_pred)
                classes = result["classes"]
                mat = result["confusion_mat"]

                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"Number of samples = {result['n_total']}",
                        f"Number of classes = {len(classes)}",
                        f"Classes = {classes}",
                    ),
                    f"Accuracy  = {result['n_correct']} / {result['n_total']} = {result['accuracy']:.6f}",
                    f"Error rate = {result['n_total'] - result['n_correct']} / {result['n_total']} = {result['error_rate']:.6f}",
                    "",
                    "Confusion matrix (rows = true class, columns = predicted class):",
                ]
                header = "True \\ Pred  " + "  ".join(f"{c:>12}" for c in classes)
                lines.append(header)
                for i, c_true in enumerate(classes):
                    row_vals = "  ".join(f"{mat[i, j]:>12}" for j in range(len(classes)))
                    lines.append(f"{c_true:>12}  " + row_vals)

                lines += [
                    "",
                    "Per-class precision (TP / column sum = correctly predicted for that class / all predicted as that class):",
                ]
                for j, c in enumerate(classes):
                    col_sum = int(mat[:, j].sum())
                    tp = int(mat[j, j])
                    prec = tp / col_sum if col_sum > 0 else float("nan")
                    lines.append(f"  {c}: precision = {tp}/{col_sum} = {prec:.4f}")

                lines.append("")
                lines.append("Per-class recall (TP / row sum = correctly predicted for that class / all true members of that class):")
                for i, c in enumerate(classes):
                    row_sum = int(mat[i, :].sum())
                    tp = int(mat[i, i])
                    rec = tp / row_sum if row_sum > 0 else float("nan")
                    lines.append(f"  {c}: recall = {tp}/{row_sum} = {rec:.4f}")

                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Compute Metrics", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_regression_metrics(self):
        tab = self._create_page("28. Regression Metrics")
        left, right = self._make_two_col_layout(tab)
        self._add_equation_guide(
            left,
            "Compute MSE, RMSE, MAE, and R² from true and predicted regression values.",
            "Enter true y values and predicted ŷ values, one per row or comma-separated.",
            "These metrics are used in Lectures 10-12 style questions on model evaluation."
        )

        frame_t, self.rm_true = labeled_scrolled_text(left, "True y values")
        frame_t.pack(fill="both", expand=True)
        set_text(self.rm_true, "1.0\n2.0\n3.0\n4.0\n5.0")

        frame_p, self.rm_pred = labeled_scrolled_text(left, "Predicted ŷ values")
        frame_p.pack(fill="both", expand=True)
        set_text(self.rm_pred, "1.1\n1.9\n3.2\n4.1\n4.8")

        def run():
            try:
                y_true = parse_numeric_matrix(self.rm_true.get("1.0", tk.END)).reshape(-1)
                y_pred = parse_numeric_matrix(self.rm_pred.get("1.0", tk.END)).reshape(-1)
                ensure_no_missing_or_infinite(y_true, "y_true")
                ensure_no_missing_or_infinite(y_pred, "y_pred")
                m = regression_metrics(y_true, y_pred)
                lines = [
                    section_block(
                        "Parameter / Dimension Summary",
                        f"N = {len(y_true)} samples",
                    ),
                    f"MSE  (Mean Squared Error)      = {m['mse']:.8g}",
                    f"RMSE (Root Mean Squared Error) = {m['rmse']:.8g}",
                    f"MAE  (Mean Absolute Error)     = {m['mae']:.8g}",
                    f"R²   (Coefficient of Det.)     = {m['r2']:.8g}",
                    f"SS_res (residual sum of sq.)   = {m['ss_res']:.8g}",
                    f"SS_tot (total sum of sq.)       = {m['ss_tot']:.8g}",
                    "",
                    "Interpretation:",
                    f"  R² = {m['r2']:.4f}" + (
                        "  (perfect fit)" if abs(m['r2'] - 1.0) < 1e-6 else
                        "  (model explains no variance, as good as predicting mean)" if abs(m['r2']) < 1e-6 else
                        "  (positive: model explains part of the variance)" if m['r2'] > 0 else
                        "  (negative: model is worse than predicting the mean)"
                    ),
                ]
                append_result(out, "\n".join(lines))
            except Exception as exc:
                append_result(out, f"Error: {exc}")

        ttk.Button(left, text="Execute / Compute Metrics", command=run).pack(anchor="w", pady=6)
        out = self._new_result_box(right)

    def _tab_self_test(self):
        tab = self._create_page("29. Toolkit Self-Test")
        left, right = self._make_two_col_layout(tab)
    
        ttk.Label(
            left,
            text=(
                "Runs built-in sanity checks across the core helper functions and the newly added exam helpers. "
                "Use this before the exam to confirm the toolkit file was not accidentally broken."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
    
        def run():
            checks = []
    
            def record(name: str, ok: bool, details: str = ""):
                checks.append((name, ok, details))
    
            try:
                A = parse_numeric_matrix("1,2\n3,4")
                record("parse_numeric_matrix", A.shape == (2, 2), f"shape={A.shape}")
            except Exception as exc:
                record("parse_numeric_matrix", False, str(exc))
    
            try:
                labels = parse_label_list("Cat\nDog\nBird")
                Y, classes = one_hot_encode(labels)
                ok = classes == ["Cat", "Dog", "Bird"] and Y.shape == (3, 3)
                record("parse_label_list + one_hot_encode", ok, f"classes={classes}, shape={Y.shape}")
            except Exception as exc:
                record("parse_label_list + one_hot_encode", False, str(exc))
    
            try:
                X = parse_numeric_matrix("1,1\n1,-1\n1,0")
                y = parse_numeric_matrix("1\n0\n2")
                w, _ = solve_least_squares(X, y)
                ok = w.shape == (2, 1)
                record("solve_least_squares", ok, f"shape={w.shape}")
            except Exception as exc:
                record("solve_least_squares", False, str(exc))
    
            try:
                P, names = make_polynomial_features(parse_numeric_matrix("0,0\n1,1"), 2)
                ok = P.shape[0] == 2 and len(names) == P.shape[1]
                record("make_polynomial_features", ok, f"shape={P.shape}, terms={len(names)}")
            except Exception as exc:
                record("make_polynomial_features", False, str(exc))
    
            try:
                stats = descriptive_stats([1, 3, 4, 6, 6, 7, 8])
                ok = "std_population" in stats and "quartiles_percentile_linear" in stats
                record("descriptive_stats", ok, "population/sample std + quartile methods available")
            except Exception as exc:
                record("descriptive_stats", False, str(exc))
    
            try:
                r = pearson_r(np.array([1, 2, 3]), np.array([2, 4, 6]))
                record("pearson_r", abs(r - 1.0) < 1e-12, f"r={r}")
            except Exception as exc:
                record("pearson_r", False, str(exc))
    
            try:
                probs, unknown_idx = solve_pmf_unknowns([1, 2, 3, 4, 5], ["0.1", "?", "0.2", "0.4", "?"], expected_value=3.5)
                ok = len(unknown_idx) == 2 and abs(probs[1] - 0.1) < 1e-9 and abs(probs[4] - 0.2) < 1e-9
                record("solve_pmf_unknowns (2 unknowns + E[X])", ok, f"probs={probs}")
            except Exception as exc:
                record("solve_pmf_unknowns (2 unknowns + E[X])", False, str(exc))
    
            try:
                inv_summary = inverse_status_summary(parse_numeric_matrix("2,0,0\n0,-1,1"))
                ok = (not inv_summary["has_left_inverse"]) and inv_summary["has_right_inverse"]
                record("inverse_status_summary", ok, f"left={inv_summary['has_left_inverse']}, right={inv_summary['has_right_inverse']}")
            except Exception as exc:
                record("inverse_status_summary", False, str(exc))
    
            try:
                shape, _ = derivative_shape_summary("vector", "vector", 2, 3)
                record("derivative_shape_summary", shape == "2 x 3 matrix", f"shape={shape}")
            except Exception as exc:
                record("derivative_shape_summary", False, str(exc))
    
            try:
                p, _ = exact_sequence_probability([("Queen", 4), ("Not Queen", 48)], ["Queen", "Queen"], with_replacement=False)
                expected = (4 / 52) * (3 / 51)
                record("exact_sequence_probability", abs(p - expected) < 1e-12, f"p={p}")
            except Exception as exc:
                record("exact_sequence_probability", False, str(exc))
    
            try:
                records = parse_group_success_records(
                    "Department A, Team A-1, 200, 150\n"
                    "Department A, Team A-2, 100, 50\n"
                    "Department B, Team B, 200, 80\n"
                    "Department C, Team C-1, 150, 90\n"
                    "Department C, Team C-2, 150, 60"
                )
                overall_success = sum(s for _, _, _, s in records)
                overall_total = sum(t for _, _, t, _ in records)
                ok = overall_total == 800 and overall_success == 430
                record("parse_group_success_records", ok, f"overall_total={overall_total}, overall_success={overall_success}")
            except Exception as exc:
                record("parse_group_success_records", False, str(exc))
    

            try:
                gd = scalar_power_gradient_descent(2, 1.0, 0.4, 1)
                ok = abs(gd["x_values"][1] - 0.2) < 1e-12
                record("scalar_power_gradient_descent", ok, f"x1={gd['x_values'][1]}")
            except Exception as exc:
                record("scalar_power_gradient_descent", False, str(exc))

            try:
                X = parse_numeric_matrix("1, 0\n1, 1\n1, 2")
                y = parse_numeric_matrix("1\n0.7\n0.5")
                details = evaluate_model_gradient(X, y, parse_numeric_matrix("0\n0"), "exponential_squared")
                ok = details["gradient"].shape == (2, 1) and np.isfinite(details["cost"])
                record("evaluate_model_gradient (exponential)", ok, f"cost={details['cost']}, grad_shape={details['gradient'].shape}")
            except Exception as exc:
                record("evaluate_model_gradient (exponential)", False, str(exc))

            try:
                fit = fit_exponential_regression_gd(
                    np.array([1981, 1990, 2000, 2010, 2018], dtype=float),
                    np.array([2.0, 2.6, 3.5, 5.0, 6.2], dtype=float),
                    eta=0.03,
                    num_steps=2000,
                    add_bias=True,
                    normalize_x=True,
                    normalize_y=True,
                )
                ok = fit["w"].shape == (2, 1) and np.isfinite(fit["training_mse_raw"]) and fit["cost_history"][0] >= fit["cost_history"][-1]
                record("fit_exponential_regression_gd", ok, f"final_mse={fit['training_mse_raw']}, steps={fit['num_steps_run']}")
            except Exception as exc:
                record("fit_exponential_regression_gd", False, str(exc))

            try:
                X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]], dtype=float)
                y = np.array([[2.0], [3.1], [3.5], [4.0]], dtype=float)
                Xb = add_bias_column(X)
                w, _ = solve_least_squares(Xb, y)
                P, _ = make_polynomial_features(X, 2)
                w_ridge, _ = solve_least_squares(P, y, ridge_lambda=0.1)
                ok = abs(float(w[0, 0]) - 1.29) < 1e-9 and abs(mse(y, Xb @ w) - 0.11025) < 1e-9 and abs(mse(y, P @ w_ridge) - 0.024568696698726827) < 1e-12
                record("midterm_q17_regression_numbers", ok, f"w0={float(w[0,0])}, mse_linear={mse(y, Xb @ w)}, mse_ridge_poly={mse(y, P @ w_ridge)}")
            except Exception as exc:
                record("midterm_q17_regression_numbers", False, str(exc))

            try:
                X = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 3], [3, 1, 4], [-1, 2, 1]], dtype=float)
                labels = ["class1", "class3", "class2", "class1", "class2"]
                Y, classes = one_hot_encode(labels)
                P, _ = make_polynomial_features(X, 2)
                W, _ = solve_least_squares(P, Y, ridge_lambda=0.01)
                Pnew, _ = make_polynomial_features(np.array([[1, 1, 2]], dtype=float), 2)
                pred = classes[int(np.argmax((Pnew @ W).reshape(-1)))]
                ok = P.shape[1] == 10 and W.size == 30 and abs(float(W[0, classes.index("class2")]) - 0.11092343297234747) < 1e-12 and pred == "class2"
                record("midterm_q18_multiclass_ridge", ok, f"params_total={W.size}, bias_class2={float(W[0, classes.index('class2')])}, pred={pred}")
            except Exception as exc:
                record("midterm_q18_multiclass_ridge", False, str(exc))

            try:
                metrics = binary_confusion_summary(
                    ["P"] * 14 + ["N"] * 27,
                    ["P"] * 7 + ["N"] * 7 + ["P"] * 2 + ["N"] * 25,
                    positive_label="P",
                )
                ok = (
                    abs(metrics["accuracy"] - (32 / 41)) < 1e-12
                    and abs(metrics["precision"] - (7 / 9)) < 1e-12
                    and abs(metrics["recall"] - 0.5) < 1e-12
                )
                record("binary_confusion_summary", ok, f"acc={metrics['accuracy']}, prec={metrics['precision']}, recall={metrics['recall']}")
            except Exception as exc:
                record("binary_confusion_summary", False, str(exc))

            try:
                best = choose_best_validation_candidate(
                    [(10, 0.10, 0.25), (9, 0.30, 0.35), (8, 0.22, 0.15), (7, 0.15, 0.25), (6, 0.18, 0.15)],
                    lower_is_better=True,
                )
                total_fits = count_cross_validation_fits(3, 5)
                ok = best["best_parameter"] == 6 and total_fits == 15
                record("cross_validation_helper", ok, f"best_param={best['best_parameter']}, total_fits={total_fits}")
            except Exception as exc:
                record("cross_validation_helper", False, str(exc))

            try:
                root = class_impurity_summary([5, 5, 8])
                child = weighted_child_impurity_summaries([[4, 0, 6], [1, 5, 2]])
                ok = (
                    abs(root["gini"] - 0.6481481481481481) < 1e-12
                    and abs(child["weighted_entropy"] - 1.116659192783882) < 1e-12
                    and abs(child["weighted_misclassification"] - (7 / 18)) < 1e-12
                )
                record("decision_tree_impurity", ok, f"root_gini={root['gini']}, weighted_entropy={child['weighted_entropy']}")
            except Exception as exc:
                record("decision_tree_impurity", False, str(exc))

            try:
                reg = regression_tree_split_summary(
                    np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9], dtype=float),
                    np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5], dtype=float),
                    5.0,
                )
                ok = (
                    abs(reg["root_mse"] - 1.2223668639053253) < 1e-12
                    and abs(reg["weighted_mse"] - 0.5101648351648351) < 1e-12
                    and abs(reg["right_mean"] - 3.85) < 1e-12
                )
                record("regression_tree_split_summary", ok, f"root_mse={reg['root_mse']}, weighted_mse={reg['weighted_mse']}")
            except Exception as exc:
                record("regression_tree_split_summary", False, str(exc))

            try:
                kmeans = run_kmeans(
                    np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]], dtype=float),
                    np.array([[0, 0], [3, 0]], dtype=float),
                    max_iter=10,
                )
                ok = (
                    kmeans["converged"]
                    and np.allclose(kmeans["final_centroids"], np.array([[0.5, 0.5], [3.5, 0.5]]))
                )
                record("run_kmeans", ok, f"centroids={format_array(kmeans['final_centroids'])}")
            except Exception as exc:
                record("run_kmeans", False, str(exc))

            try:
                nn = forward_neural_network(
                    np.array([[1, 0], [0, 1], [1, 1]], dtype=float),
                    [
                        np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
                        np.array([[0], [1], [1]], dtype=float),
                    ],
                    hidden_activation="relu",
                    output_activation="linear",
                    add_bias=True,
                )
                ok = nn["total_parameters"] == 9 and np.allclose(nn["final_output"], np.array([[1], [1], [2]], dtype=float))
                record("forward_neural_network", ok, f"output={format_array(nn['final_output'])}")
            except Exception as exc:
                record("forward_neural_network", False, str(exc))

            passed = sum(ok for _, ok, _ in checks)
            lines = [f"Passed {passed} / {len(checks)} checks", ""]
            for name, ok, details in checks:
                status = "PASS" if ok else "FAIL"
                lines.append(f"[{status}] {name}")
                if details:
                    lines.append(f"    {details}")
            append_result(out, "\n".join(lines))
    
        ttk.Button(left, text="Run Self-Test", command=run).pack(anchor="w")
        out = self._new_result_box(right)


if __name__ == "__main__":
    app = EE2211ToolkitApp()
    app.mainloop()
