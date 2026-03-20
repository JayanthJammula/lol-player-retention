"""
test_predictions.py
Tests for the structured prediction pipeline. Ensures predictions return
valid, typed, traceable outputs and that validation catches bad input.
"""
import unittest
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from schemas import (
    validate_feature_input, make_prediction, FeatureVector,
    PredictionResult, ValidationError, TRAINING_FEATURES, FEATURE_RANGES
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LR_PATH = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'player_features_temporal.csv')

SKIP_MSG = "Model artifacts not found. Run train.py first."


class TestInputValidation(unittest.TestCase):
    """Tests for validate_feature_input()."""

    def _make_valid_input(self):
        """Return a dict of feature values within valid ranges."""
        return {
            'win_rate': 0.5, 'kda': 3.0, 'kill_death_ratio': 2.0,
            'avg_damage': 15000, 'avg_cs': 120, 'avg_vision_score': 20,
            'avg_game_duration': 1800, 'avg_champion_level': 12,
            'avg_gold_earned': 10000, 'gold_efficiency': 0.85,
            'unique_champions': 5, 'total_games_played': 20,
            'unique_play_days': 10, 'avg_time_between_games_hrs': 5.0,
            'median_time_between_games_hrs': 3.0, 'play_frequency': 2.0,
            'feature_window_days': 30.0, 'kda_trend': 0.1,
            'winrate_trend': 0.05, 'last_gap_days': 1.0,
        }

    def test_valid_input_passes(self):
        """Valid feature input should return a FeatureVector."""
        fv = validate_feature_input(self._make_valid_input())
        self.assertIsInstance(fv, FeatureVector)
        self.assertEqual(len(fv.values), 20)

    def test_missing_features_raises_error(self):
        """Missing features should raise ValidationError."""
        incomplete = {'win_rate': 0.5}
        with self.assertRaises(ValidationError) as ctx:
            validate_feature_input(incomplete)
        self.assertIn("Missing features", str(ctx.exception))

    def test_out_of_range_raises_error(self):
        """Out-of-range values should raise ValidationError."""
        bad_input = self._make_valid_input()
        bad_input['win_rate'] = 2.0  # max is 1.0
        with self.assertRaises(ValidationError):
            validate_feature_input(bad_input)

    def test_nan_raises_error(self):
        """NaN values should raise ValidationError."""
        bad_input = self._make_valid_input()
        bad_input['kda'] = float('nan')
        with self.assertRaises(ValidationError):
            validate_feature_input(bad_input)

    def test_inf_raises_error(self):
        """Infinite values should raise ValidationError."""
        bad_input = self._make_valid_input()
        bad_input['avg_damage'] = float('inf')
        with self.assertRaises(ValidationError):
            validate_feature_input(bad_input)

    def test_feature_vector_to_array(self):
        """FeatureVector.to_array() should return correct shape."""
        fv = validate_feature_input(self._make_valid_input())
        arr = fv.to_array(TRAINING_FEATURES)
        self.assertEqual(arr.shape, (1, 20))


@unittest.skipUnless(
    os.path.exists(SCALER_PATH) and os.path.exists(LR_PATH), SKIP_MSG
)
class TestPredictionPipeline(unittest.TestCase):
    """Tests for make_prediction() with real models."""

    @classmethod
    def setUpClass(cls):
        import joblib
        cls.scaler = joblib.load(SCALER_PATH)
        cls.model = joblib.load(LR_PATH)

    def _make_valid_fv(self):
        valid_input = {
            'win_rate': 0.5, 'kda': 3.0, 'kill_death_ratio': 2.0,
            'avg_damage': 15000, 'avg_cs': 120, 'avg_vision_score': 20,
            'avg_game_duration': 1800, 'avg_champion_level': 12,
            'avg_gold_earned': 10000, 'gold_efficiency': 0.85,
            'unique_champions': 5, 'total_games_played': 20,
            'unique_play_days': 10, 'avg_time_between_games_hrs': 5.0,
            'median_time_between_games_hrs': 3.0, 'play_frequency': 2.0,
            'feature_window_days': 30.0, 'kda_trend': 0.1,
            'winrate_trend': 0.05, 'last_gap_days': 1.0,
        }
        return validate_feature_input(valid_input)

    def test_returns_structured_output(self):
        """make_prediction() should return a PredictionResult."""
        fv = self._make_valid_fv()
        result = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        self.assertIsInstance(result, PredictionResult)

    def test_probability_in_valid_range(self):
        """Prediction probability should be in [0, 1]."""
        fv = self._make_valid_fv()
        result = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_label_matches_probability(self):
        """Label should match the probability threshold."""
        fv = self._make_valid_fv()
        result = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        if result.probability >= 0.5:
            self.assertEqual(result.predicted_label, "Churned")
        else:
            self.assertEqual(result.predicted_label, "Active")

    def test_has_timestamp(self):
        """Result should include an ISO timestamp."""
        fv = self._make_valid_fv()
        result = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        self.assertIsNotNone(result.timestamp)
        self.assertIn('T', result.timestamp)  # ISO format contains 'T'

    def test_determinism(self):
        """Same input should produce the same output (sklearn models are deterministic)."""
        fv = self._make_valid_fv()
        r1 = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        r2 = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        self.assertAlmostEqual(r1.probability, r2.probability, places=10)
        self.assertEqual(r1.predicted_label, r2.predicted_label)

    def test_to_dict_complete(self):
        """PredictionResult.to_dict() should include all required fields."""
        fv = self._make_valid_fv()
        result = make_prediction(fv, self.model, self.scaler, "Logistic Regression")
        d = result.to_dict()
        required_keys = [
            'probability', 'predicted_label', 'confidence',
            'model_name', 'model_version', 'timestamp', 'feature_values',
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key in to_dict(): {key}")


if __name__ == '__main__':
    unittest.main()
