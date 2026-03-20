"""
test_data_quality.py
Regression tests for data quality. Ensures the dataset meets minimum
standards before any model training proceeds.
"""
import unittest
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering import load_and_clean, get_training_features, run_data_quality_checks

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'player_features_temporal.csv')
SKIP_MSG = "Dataset not found. Run feature_extraction_temporal.py first."


@unittest.skipUnless(os.path.exists(DATA_PATH), SKIP_MSG)
class TestDataQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_and_clean()
        cls.features = get_training_features()

    def test_no_missing_values(self):
        """No NaN values in training features."""
        missing = self.df[self.features].isnull().sum()
        missing_any = missing[missing > 0]
        self.assertEqual(len(missing_any), 0,
                         f"Missing values found: {missing_any.to_dict()}")

    def test_feature_ranges(self):
        """All features within expected valid ranges."""
        range_checks = {
            'win_rate': (0, 1),
            'gold_efficiency': (0, 5),
            'kda': (0, 50),
            'kill_death_ratio': (0, 50),
            'total_games_played': (3, 500),
            'avg_time_between_games_hrs': (0, 1000),
            'play_frequency': (0, 200),
            'feature_window_days': (0, 365),
            'last_gap_days': (0, 365),
        }
        for feat, (low, high) in range_checks.items():
            if feat in self.df.columns:
                violations = ((self.df[feat] < low) | (self.df[feat] > high)).sum()
                self.assertEqual(violations, 0,
                                 f"{feat}: {violations} values outside [{low}, {high}]")

    def test_churn_rate_reasonable(self):
        """Churn rate is between 5% and 50% (not degenerate)."""
        churn_rate = self.df['churn'].mean()
        self.assertGreaterEqual(churn_rate, 0.05,
                                f"Churn rate too low: {churn_rate:.1%}")
        self.assertLessEqual(churn_rate, 0.50,
                             f"Churn rate too high: {churn_rate:.1%}")

    def test_no_duplicate_puuids(self):
        """Each player appears exactly once."""
        n_dupes = self.df['puuid'].duplicated().sum()
        self.assertEqual(n_dupes, 0, f"Found {n_dupes} duplicate PUUIDs")

    def test_no_infinite_values(self):
        """No infinite values in numeric features."""
        numeric = self.df[self.features].select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric).sum().sum()
        self.assertEqual(inf_count, 0, f"Found {inf_count} infinite values")

    def test_minimum_dataset_size(self):
        """Dataset has at least 100 players."""
        self.assertGreaterEqual(len(self.df), 100,
                                f"Only {len(self.df)} players, possible data corruption")

    def test_all_quality_checks_pass(self):
        """The full data quality report shows all checks passed."""
        report = run_data_quality_checks(self.df, save_path=None)
        failed = [c['name'] for c in report['checks'] if not c['passed']]
        self.assertTrue(report['passed'],
                        f"Failed checks: {failed}")


if __name__ == '__main__':
    unittest.main()
