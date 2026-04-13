import numpy as np

from ee2211_calc.core import (
    bayes_posterior,
    classify_from_linear_scores,
    classify_linear_system,
    correlation,
    gradient_descent_step,
    lambda_effect_summary,
    mse,
    normal_cdf,
    normal_pdf,
    one_hot,
    parameter_count,
    pmf_valid,
    polynomial_features,
    predict_linear,
    residuals,
    ridge_regression,
    with_bias_column,
)


def test_bias_column():
    X = np.array([[2, 3], [4, 5]])
    out = with_bias_column(X)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, 0], 1.0)


def test_system_types_and_inverses():
    under = classify_linear_system(np.array([[1, 0, 0], [0, 1, 0]]))
    even = classify_linear_system(np.eye(2))
    over = classify_linear_system(np.array([[1, 0], [0, 1], [1, 1]]))

    assert under.system_type == "underdetermined"
    assert even.system_type == "even-determined"
    assert over.system_type == "overdetermined"
    assert even.has_inverse is True


def test_parameter_count_multioutput_with_bias():
    assert parameter_count(5, n_outputs=3, include_bias=True) == 18


def test_polynomial_features_degree_three():
    X = np.array([2, 3])
    Phi = polynomial_features(X, 3)
    assert np.allclose(Phi, np.array([[2, 4, 8], [3, 9, 27]]))


def test_ridge_regression_lambda_shrinks_weights():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([1, 3, 5, 7], dtype=float)
    w0 = ridge_regression(X, y, lam=0)
    w1 = ridge_regression(X, y, lam=100)
    assert np.linalg.norm(w1[1:]) < np.linalg.norm(w0[1:])


def test_predict_residual_mse_pipeline():
    X = np.array([[0], [1]], dtype=float)
    W = np.array([[1.0], [2.0]])
    y = np.array([[1.0], [3.0]])
    yp = predict_linear(X, W)
    assert np.allclose(yp, y)
    assert np.allclose(residuals(y, yp), np.zeros_like(y))
    assert mse(y, yp) == 0.0


def test_one_hot_and_multiclass_argmax():
    y = [0, 2, 1]
    Y = one_hot(y, n_classes=3)
    assert Y.shape == (3, 3)

    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    labels = classify_from_linear_scores(X, W)
    assert np.array_equal(labels, np.array([0, 1]))


def test_binary_threshold_classification():
    X = np.array([[0.0], [1.0], [2.0]])
    W = np.array([[0.1], [0.3]])
    labels = classify_from_linear_scores(X, W)
    assert np.array_equal(labels, np.array([0, 0, 1]))


def test_probability_helpers():
    assert abs(bayes_posterior(0.8, 0.2, 0.4) - 0.4) < 1e-12
    assert pmf_valid([0.2, 0.3, 0.5])
    assert not pmf_valid([0.2, 0.3, 0.6])
    assert abs(normal_cdf(0, 0, 1) - 0.5) < 1e-12
    assert abs(normal_pdf(0, 0, 1) - 0.3989422804) < 1e-9


def test_misc_helpers():
    assert abs(correlation(np.array([1, 2, 3]), np.array([2, 4, 6])) - 1.0) < 1e-12
    assert np.allclose(gradient_descent_step(np.array([1.0, 2.0]), np.array([0.5, -1.0]), 0.1), [0.95, 2.1])
    assert "regularization" in lambda_effect_summary(0.1, 1.0)
