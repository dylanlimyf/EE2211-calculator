import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "EE2211_Exam_Toolkit_GUI_v16_midterm_patched.py"
SPEC = importlib.util.spec_from_file_location("ee2211_toolkit", MODULE_PATH)
toolkit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(toolkit)


class ExamToolkitTests(unittest.TestCase):
    def test_midterm_q17_regression_numbers(self):
        X = np.array([[1, 1], [2, 1], [1, 2], [2, 3]], dtype=float)
        y = np.array([[2.0], [3.1], [3.5], [4.0]], dtype=float)
        Xb = toolkit.add_bias_column(X)
        w, _ = toolkit.solve_least_squares(Xb, y)
        P, _ = toolkit.make_polynomial_features(X, 2)
        w_ridge, _ = toolkit.solve_least_squares(P, y, ridge_lambda=0.1)

        self.assertAlmostEqual(float(w[0, 0]), 1.29, places=9)
        self.assertAlmostEqual(toolkit.mse(y, Xb @ w), 0.11025, places=9)
        self.assertAlmostEqual(toolkit.mse(y, P @ w_ridge), 0.024568696698726827, places=12)

    def test_midterm_q18_multiclass_ridge(self):
        X = np.array([[2, 1, 0], [0, 3, 1], [1, 0, 3], [3, 1, 4], [-1, 2, 1]], dtype=float)
        labels = ["class1", "class3", "class2", "class1", "class2"]
        Y, classes = toolkit.one_hot_encode(labels)
        P, _ = toolkit.make_polynomial_features(X, 2)
        W, _ = toolkit.solve_least_squares(P, Y, ridge_lambda=0.01)
        Pnew, _ = toolkit.make_polynomial_features(np.array([[1, 1, 2]], dtype=float), 2)
        pred = classes[int(np.argmax((Pnew @ W).reshape(-1)))]

        self.assertEqual(P.shape[1], 10)
        self.assertEqual(W.size, 30)
        self.assertAlmostEqual(float(W[0, classes.index("class2")]), 0.11092343297234747, places=12)
        self.assertEqual(pred, "class2")

    def test_binary_confusion_summary(self):
        summary = toolkit.binary_confusion_summary(
            ["P"] * 14 + ["N"] * 27,
            ["P"] * 7 + ["N"] * 7 + ["P"] * 2 + ["N"] * 25,
            positive_label="P",
        )

        self.assertAlmostEqual(summary["accuracy"], 32 / 41, places=12)
        self.assertAlmostEqual(summary["precision"], 7 / 9, places=12)
        self.assertAlmostEqual(summary["recall"], 0.5, places=12)
        self.assertAlmostEqual(summary["specificity"], 25 / 27, places=12)

    def test_cross_validation_helper(self):
        best = toolkit.choose_best_validation_candidate(
            [(10, 0.10, 0.25), (9, 0.30, 0.35), (8, 0.22, 0.15), (7, 0.15, 0.25), (6, 0.18, 0.15)],
            lower_is_better=True,
        )
        self.assertEqual(toolkit.count_cross_validation_fits(3, 5), 15)
        self.assertEqual(best["best_parameter"], 6)

    def test_decision_tree_impurity_from_tutorial_9(self):
        root = toolkit.class_impurity_summary([5, 5, 8])
        child = toolkit.weighted_child_impurity_summaries([[4, 0, 6], [1, 5, 2]])

        self.assertAlmostEqual(root["gini"], 0.6481481481481481, places=12)
        self.assertAlmostEqual(root["entropy"], 1.5466316186596222, places=8)
        self.assertAlmostEqual(child["weighted_gini"], 0.5027777777777778, places=12)
        self.assertAlmostEqual(child["weighted_entropy"], 1.116659192783882, places=12)
        self.assertAlmostEqual(child["weighted_misclassification"], 7 / 18, places=12)

    def test_regression_tree_split_summary_from_tutorial_9(self):
        result = toolkit.regression_tree_split_summary(
            np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9], dtype=float),
            np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5], dtype=float),
            5.0,
        )

        self.assertAlmostEqual(result["root_mean"], 2.9384615384615387, places=12)
        self.assertAlmostEqual(result["right_mean"], 3.85, places=12)
        self.assertAlmostEqual(result["root_mse"], 1.2223668639053253, places=12)
        self.assertAlmostEqual(result["weighted_mse"], 0.5101648351648351, places=12)

    def test_kmeans_converges_to_expected_centroids(self):
        result = toolkit.run_kmeans(
            np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]], dtype=float),
            np.array([[0, 0], [3, 0]], dtype=float),
            max_iter=10,
        )

        self.assertTrue(result["converged"])
        np.testing.assert_allclose(result["final_centroids"], np.array([[0.5, 0.5], [3.5, 0.5]]))

    def test_clustering_accuracy_permutation(self):
        result = toolkit.best_clustering_accuracy(
            ["A", "A", "B", "B"],
            np.array([1, 1, 0, 0]),
        )
        self.assertAlmostEqual(result["accuracy"], 1.0, places=12)

    def test_forward_neural_network(self):
        result = toolkit.forward_neural_network(
            np.array([[1, 0], [0, 1], [1, 1]], dtype=float),
            [
                np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
                np.array([[0], [1], [1]], dtype=float),
            ],
            hidden_activation="relu",
            output_activation="linear",
            add_bias=True,
        )

        self.assertEqual(result["total_parameters"], 9)
        np.testing.assert_allclose(result["final_output"], np.array([[1.0], [1.0], [2.0]]))

    def test_probability_solver_still_handles_constraints(self):
        probs, unknown_idx = toolkit.solve_pmf_unknowns(
            [1, 2, 3, 4, 5],
            ["0.1", "?", "0.2", "0.4", "?"],
            expected_value=3.5,
        )

        self.assertEqual(unknown_idx, [1, 4])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=12)

    def test_midterm_style_guide_contains_core_exam_patterns(self):
        self.assertIn("Polynomial parameter count + system type", toolkit.MIDTERM_STYLE_GUIDE)
        self.assertIn("Linear multiclass dimensions with bias", toolkit.MIDTERM_STYLE_GUIDE)
        self.assertIn("One-hot regression prediction by argmax", toolkit.MIDTERM_STYLE_GUIDE)
        self.assertIn("Ridge lambda increases", toolkit.MIDTERM_STYLE_GUIDE)
        self.assertIn("C(d+p, p)", toolkit.MIDTERM_STYLE_GUIDE["Polynomial parameter count + system type"])
        self.assertIn("(d+1) x C", toolkit.MIDTERM_STYLE_GUIDE["Linear multiclass dimensions with bias"])

    def test_midterm_open_ended_coverage_contains_story_questions(self):
        self.assertIn("Sem2 Midterm Q15 student data story", toolkit.MIDTERM_PAPER_COVERAGE)
        self.assertIn("Sem2 Midterm Q16 factory defect Bayes", toolkit.MIDTERM_PAPER_COVERAGE)
        self.assertIn("PYP Midterm Q23 AutoDrive open-ended", toolkit.MIDTERM_PAPER_COVERAGE)
        self.assertIn("PYP Midterm Q24 department / team promotion", toolkit.MIDTERM_PAPER_COVERAGE)
        self.assertIn("Unsupervised learning", toolkit.MIDTERM_PAPER_COVERAGE["Sem2 Midterm Q15 student data story"])
        self.assertIn("15c. Count / Conditional Probability", toolkit.MIDTERM_PAPER_COVERAGE["PYP Midterm Q24 department / team promotion"])

    def test_open_ended_guides_cover_learning_type_variable_type_and_prep(self):
        self.assertIn("Labeled category prediction", toolkit.SCENARIO_LEARNING_GUIDE)
        self.assertIn("City / Major / hand-gesture class", toolkit.VARIABLE_ENCODING_GUIDE)
        self.assertIn("Expert removes poor-quality samples", toolkit.PREPROCESSING_SCENARIO_GUIDE)
        self.assertIn("Supervised learning", toolkit.SCENARIO_LEARNING_GUIDE["Labeled category prediction"])
        self.assertIn("Nominal", toolkit.VARIABLE_ENCODING_GUIDE["City / Major / hand-gesture class"])
        self.assertIn("Data cleaning", toolkit.PREPROCESSING_SCENARIO_GUIDE["Expert removes poor-quality samples"])


if __name__ == "__main__":
    unittest.main()
