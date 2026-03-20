"""
infrastructure.py
Production infrastructure utilities: safe model loading, retry logic,
data quality gates, and feature drift detection.
"""
import logging
import time
import os
import json
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger('churn_predictor.infra')


# ---------------------------------------------------------------------------
# Safe model loading
# ---------------------------------------------------------------------------
def safe_load_model(path: str, model_type: str = 'joblib') -> Tuple[Any, Optional[str]]:
    """
    Load a model file with error handling.
    Returns (model, None) on success, (None, error_message) on failure.
    """
    if not os.path.exists(path):
        msg = f"Model file not found: {path}"
        logger.error(msg)
        return None, msg

    try:
        if model_type == 'joblib':
            import joblib
            model = joblib.load(path)
        elif model_type == 'keras':
            from tensorflow.keras.models import load_model
            model = load_model(path)
        else:
            msg = f"Unknown model type: {model_type}"
            logger.error(msg)
            return None, msg

        logger.info(f"Loaded model: {os.path.basename(path)}")
        return model, None

    except Exception as e:
        msg = f"Failed to load {path}: {str(e)}"
        logger.error(msg)
        return None, msg


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0,
                       backoff_factor: float = 2.0,
                       exceptions: tuple = (Exception,)):
    """
    Decorator that retries a function with exponential backoff.

    Usage:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def flaky_api_call():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{fn.__name__} attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{fn.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

            raise last_exception
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Data quality gate
# ---------------------------------------------------------------------------
class DataQualityError(Exception):
    """Raised when data quality checks fail and training should not proceed."""

    def __init__(self, failed_checks: List[str]):
        self.failed_checks = failed_checks
        super().__init__(
            f"Data quality gate failed on {len(failed_checks)} check(s): "
            + "; ".join(failed_checks)
        )


class DataQualityGate:
    """
    Wraps data quality checks with a hard pass/fail gate.
    If critical checks fail, training is halted with a clear error.
    """

    def __init__(self, dq_report: dict):
        self.report = dq_report

    def enforce(self):
        """Raise DataQualityError if any check failed."""
        failed = [
            c['name'] for c in self.report.get('checks', [])
            if not c.get('passed', False)
        ]

        if failed:
            logger.error(f"Data quality gate FAILED: {failed}")
            raise DataQualityError(failed)

        n_checks = len(self.report.get('checks', []))
        logger.info(f"Data quality gate PASSED: {n_checks}/{n_checks} checks OK")


# ---------------------------------------------------------------------------
# Feature drift detection
# ---------------------------------------------------------------------------
def compute_training_stats(X_train, feature_names: List[str]) -> dict:
    """
    Compute per-feature statistics from training data.
    Saved alongside models so drift can be detected at prediction time.
    """
    if hasattr(X_train, 'values'):
        arr = X_train.values
    else:
        arr = np.array(X_train)

    stats = {}
    for i, feat in enumerate(feature_names):
        col = arr[:, i]
        stats[feat] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'p25': float(np.percentile(col, 25)),
            'p50': float(np.percentile(col, 50)),
            'p75': float(np.percentile(col, 75)),
        }

    return stats


def save_training_stats(stats: dict, path: str = './models/training_stats.json'):
    """Save training statistics to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Training stats saved to {path}")


def load_training_stats(path: str = './models/training_stats.json') -> Optional[dict]:
    """Load training statistics from disk."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def detect_feature_drift(training_stats: dict,
                         input_values: dict,
                         threshold: float = 2.0) -> List[dict]:
    """
    Check if input feature values are far from training distribution.
    Returns a list of drift warnings for features exceeding threshold
    standard deviations from the training mean.
    """
    warnings = []

    for feat, val in input_values.items():
        if feat not in training_stats:
            continue

        stats = training_stats[feat]
        mean = stats['mean']
        std = stats['std']

        if std == 0:
            continue

        z_score = abs(val - mean) / std
        if z_score > threshold:
            warnings.append({
                'feature': feat,
                'value': val,
                'training_mean': round(mean, 4),
                'training_std': round(std, 4),
                'z_score': round(z_score, 2),
            })

    if warnings:
        logger.warning(
            f"Feature drift detected in {len(warnings)} feature(s): "
            + ", ".join(w['feature'] for w in warnings)
        )

    return warnings
