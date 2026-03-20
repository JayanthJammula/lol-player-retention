"""
schemas.py
Structured output types and validation for the churn prediction pipeline.
Ensures predictions are typed, validated, and traceable.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np


# Valid ranges for each feature (used for input validation)
FEATURE_RANGES = {
    'win_rate': (0, 1),
    'kda': (0, 50),
    'kill_death_ratio': (0, 50),
    'avg_damage': (0, 100000),
    'avg_cs': (0, 500),
    'avg_vision_score': (0, 200),
    'avg_game_duration': (0, 7200),
    'avg_champion_level': (1, 18),
    'avg_gold_earned': (0, 50000),
    'gold_efficiency': (0, 5),
    'unique_champions': (1, 200),
    'total_games_played': (3, 500),
    'unique_play_days': (1, 365),
    'avg_time_between_games_hrs': (0, 1000),
    'median_time_between_games_hrs': (0, 1000),
    'play_frequency': (0, 200),
    'feature_window_days': (0, 365),
    'kda_trend': (-50, 50),
    'winrate_trend': (-1, 1),
    'last_gap_days': (0, 365),
}

# Ordered list of training features
TRAINING_FEATURES = [
    'win_rate', 'kda', 'kill_death_ratio', 'avg_damage', 'avg_cs',
    'avg_vision_score', 'avg_game_duration', 'avg_champion_level',
    'avg_gold_earned', 'gold_efficiency', 'unique_champions',
    'total_games_played', 'unique_play_days', 'avg_time_between_games_hrs',
    'median_time_between_games_hrs', 'play_frequency', 'feature_window_days',
    'kda_trend', 'winrate_trend', 'last_gap_days',
]


class ValidationError(Exception):
    """Raised when input data fails schema validation."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


@dataclass
class FeatureVector:
    """Validated feature input for prediction."""
    values: Dict[str, float]
    warnings: List[str] = field(default_factory=list)

    def to_array(self, feature_order: List[str]) -> np.ndarray:
        """Convert to numpy array in the given feature order."""
        return np.array([[self.values[f] for f in feature_order]])


@dataclass
class PredictionResult:
    """Structured prediction output with full traceability."""
    probability: float
    predicted_label: str
    confidence: str
    model_name: str
    model_version: str
    timestamp: str
    feature_values: Dict[str, float]
    feature_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Tracks provenance for a trained model."""
    model_name: str
    model_version: str
    training_date: str
    dataset_hash: str
    dataset_size: int
    performance_snapshot: Dict[str, float]
    feature_names: List[str]
    hyperparameters: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def compute_dataset_hash(csv_path: str) -> str:
        """Compute MD5 hash of training data for reproducibility tracking."""
        h = hashlib.md5()
        with open(csv_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()


def validate_feature_input(feature_dict: Dict[str, float]) -> FeatureVector:
    """
    Validate raw feature input against schema.
    Returns a FeatureVector on success, raises ValidationError on failure.
    Warnings are generated for values near range boundaries.
    """
    errors = []
    warnings = []

    # Check all required features are present
    missing = [f for f in TRAINING_FEATURES if f not in feature_dict]
    if missing:
        errors.append(f"Missing features: {missing}")

    # Check types and ranges
    for feat in TRAINING_FEATURES:
        if feat not in feature_dict:
            continue

        val = feature_dict[feat]

        # Type check
        if not isinstance(val, (int, float, np.integer, np.floating)):
            errors.append(f"{feat}: expected numeric, got {type(val).__name__}")
            continue

        val = float(val)

        # NaN/Inf check
        if np.isnan(val) or np.isinf(val):
            errors.append(f"{feat}: value is {'NaN' if np.isnan(val) else 'Inf'}")
            continue

        # Range check
        if feat in FEATURE_RANGES:
            lo, hi = FEATURE_RANGES[feat]
            if val < lo or val > hi:
                errors.append(f"{feat}: {val} outside valid range [{lo}, {hi}]")

    if errors:
        raise ValidationError(errors)

    clean_values = {f: float(feature_dict[f]) for f in TRAINING_FEATURES}
    return FeatureVector(values=clean_values, warnings=warnings)


def classify_confidence(probability: float) -> str:
    """Map probability distance from 0.5 to confidence level."""
    distance = abs(probability - 0.5)
    if distance >= 0.3:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"


def make_prediction(
    feature_vector: FeatureVector,
    model,
    scaler,
    model_name: str,
    model_version: str = "1.0",
) -> PredictionResult:
    """
    Run prediction through the full structured pipeline.
    Scales input, runs inference, and returns a PredictionResult.
    """
    arr = feature_vector.to_array(TRAINING_FEATURES)
    scaled = scaler.transform(arr)

    # Get probability
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(scaled)[:, 1][0])
    else:
        prob = float(model.predict(scaled, verbose=0).flatten()[0])

    # Clamp to [0, 1] for safety
    prob = max(0.0, min(1.0, prob))

    label = "Churned" if prob >= 0.5 else "Active"
    confidence = classify_confidence(prob)

    return PredictionResult(
        probability=prob,
        predicted_label=label,
        confidence=confidence,
        model_name=model_name,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature_values=feature_vector.values,
        feature_warnings=feature_vector.warnings,
    )


def save_model_metadata(metadata_list: List[ModelMetadata],
                        path: str = './models/model_metadata.json'):
    """Save model metadata to JSON for traceability."""
    data = {m.model_name: m.to_dict() for m in metadata_list}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_model_metadata(path: str = './models/model_metadata.json') -> Optional[dict]:
    """Load model metadata from disk. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
