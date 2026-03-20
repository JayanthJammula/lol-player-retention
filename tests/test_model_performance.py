"""
test_model_performance.py
Regression tests for trained model performance. Ensures models meet
minimum performance thresholds and catches catastrophic degradation.
"""
import unittest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
COMPARISON_PATH = os.path.join(MODEL_DIR, 'model_comparison.json')
CV_PATH = os.path.join(MODEL_DIR, 'cv_results.json')

SKIP_MSG = "Model comparison data not found. Run train.py first."


@unittest.skipUnless(os.path.exists(COMPARISON_PATH), SKIP_MSG)
class TestModelPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(COMPARISON_PATH) as f:
            cls.comparison = json.load(f)

        cls.cv_results = None
        if os.path.exists(CV_PATH):
            with open(CV_PATH) as f:
                cls.cv_results = json.load(f)

    def _non_baseline_models(self):
        return {k: v for k, v in self.comparison.items()
                if 'baseline' not in k.lower()}

    def test_models_beat_baseline_f1(self):
        """Every non-baseline model should have F1 >= baseline F1."""
        baseline_f1 = self.comparison.get('Baseline (Majority)', {}).get('f1', 0)
        for name, metrics in self._non_baseline_models().items():
            with self.subTest(model=name):
                self.assertGreaterEqual(
                    metrics['f1'], baseline_f1,
                    f"{name} F1 ({metrics['f1']:.4f}) is below baseline ({baseline_f1:.4f})"
                )

    def test_minimum_auc(self):
        """Every non-baseline model should have ROC-AUC >= 0.45.
        Threshold is 0.45 (not 0.50) because LSTM on small datasets can
        underperform random due to high variance. This catches only
        catastrophic failure, not marginal underperformance."""
        for name, metrics in self._non_baseline_models().items():
            with self.subTest(model=name):
                self.assertGreaterEqual(
                    metrics['roc_auc'], 0.45,
                    f"{name} ROC-AUC ({metrics['roc_auc']:.4f}) indicates catastrophic failure"
                )

    def test_confidence_intervals_valid(self):
        """Bootstrap CI lower bound <= point estimate <= upper bound."""
        for name, metrics in self.comparison.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                ci = metrics.get(f'{metric}_ci')
                if ci:
                    with self.subTest(model=name, metric=metric):
                        self.assertLessEqual(
                            ci[0], metrics[metric] + 0.01,
                            f"{name} {metric}: CI lower ({ci[0]:.4f}) > "
                            f"point estimate ({metrics[metric]:.4f})"
                        )
                        self.assertGreaterEqual(
                            ci[1], metrics[metric] - 0.01,
                            f"{name} {metric}: CI upper ({ci[1]:.4f}) < "
                            f"point estimate ({metrics[metric]:.4f})"
                        )

    @unittest.skipUnless(os.path.exists(CV_PATH), "CV results not found")
    def test_cv_consistency(self):
        """Cross-validation standard deviation < 0.20 for each model (stability)."""
        for name, res in self.cv_results.items():
            if 'baseline' in name.lower():
                continue
            for metric in ['f1', 'roc_auc']:
                with self.subTest(model=name, metric=metric):
                    std = res['std'][metric]
                    self.assertLess(
                        std, 0.20,
                        f"{name} {metric} CV std ({std:.4f}) indicates high instability"
                    )

    def test_probability_ranges(self):
        """All stored FPR/TPR values are in [0, 1]."""
        for name, metrics in self.comparison.items():
            with self.subTest(model=name):
                for fpr_val in metrics.get('fpr', []):
                    self.assertGreaterEqual(fpr_val, 0.0)
                    self.assertLessEqual(fpr_val, 1.0)
                for tpr_val in metrics.get('tpr', []):
                    self.assertGreaterEqual(tpr_val, 0.0)
                    self.assertLessEqual(tpr_val, 1.0)


if __name__ == '__main__':
    unittest.main()
