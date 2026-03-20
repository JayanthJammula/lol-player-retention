"""
test_feature_engineering.py
Tests for feature engineering functions. Ensures data loading and feature
definitions remain consistent and correct.
"""
import unittest
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering import load_and_clean, get_training_features, get_all_features

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'player_features_temporal.csv')
SKIP_MSG = "Dataset not found. Run feature_extraction_temporal.py first."


class TestFeatureDefinitions(unittest.TestCase):
    """Tests for feature name lists."""

    def test_feature_count(self):
        """get_training_features() should return exactly 20 features."""
        features = get_training_features()
        self.assertEqual(len(features), 20)

    def test_no_duplicate_features(self):
        """No duplicate feature names."""
        features = get_training_features()
        self.assertEqual(len(features), len(set(features)))

    def test_all_features_matches_training(self):
        """get_all_features() should return the same list as get_training_features()."""
        self.assertEqual(get_training_features(), get_all_features())


@unittest.skipUnless(os.path.exists(DATA_PATH), SKIP_MSG)
class TestDataLoading(unittest.TestCase):
    """Tests for load_and_clean()."""

    @classmethod
    def setUpClass(cls):
        cls.df = load_and_clean()
        cls.features = get_training_features()

    def test_returns_dataframe(self):
        """load_and_clean() should return a pandas DataFrame."""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_all_features_in_dataframe(self):
        """All training features should exist as columns."""
        for feat in self.features:
            self.assertIn(feat, self.df.columns,
                          f"Feature {feat} missing from dataframe")

    def test_feature_types_numeric(self):
        """All features should be numeric (int or float)."""
        for feat in self.features:
            self.assertTrue(
                np.issubdtype(self.df[feat].dtype, np.number),
                f"Feature {feat} is not numeric: {self.df[feat].dtype}"
            )

    def test_has_churn_column(self):
        """Dataset should have a 'churn' column."""
        self.assertIn('churn', self.df.columns)

    def test_has_puuid_column(self):
        """Dataset should have a 'puuid' column."""
        self.assertIn('puuid', self.df.columns)

    def test_churn_values_binary(self):
        """Churn column should only contain 0 and 1."""
        unique_vals = set(self.df['churn'].unique())
        self.assertTrue(unique_vals.issubset({0, 1}),
                        f"Unexpected churn values: {unique_vals}")

    def test_clamped_ratios(self):
        """KDA and kill_death_ratio should be clamped at 50."""
        for col in ['kda', 'kill_death_ratio']:
            if col in self.df.columns:
                self.assertLessEqual(self.df[col].max(), 50.0,
                                     f"{col} not properly clamped")

    def test_non_negative_time_features(self):
        """Time-based features should be non-negative after cleaning."""
        time_feats = ['avg_time_between_games_hrs',
                      'median_time_between_games_hrs',
                      'last_gap_days', 'feature_window_days']
        for feat in time_feats:
            if feat in self.df.columns:
                self.assertGreaterEqual(self.df[feat].min(), 0,
                                        f"{feat} has negative values")


if __name__ == '__main__':
    unittest.main()
