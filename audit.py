"""
audit.py
Prediction audit trail for traceability and observability.
Every prediction is logged as a JSON line with full context:
model, version, inputs, outputs, and timestamp.
"""
import json
import os
import logging
from typing import List, Optional

logger = logging.getLogger('churn_predictor.predict')

AUDIT_LOG_PATH = './logs/prediction_audit.jsonl'


def log_prediction(prediction_result, source: str = "unknown"):
    """
    Append a structured prediction record to the audit log.

    Args:
        prediction_result: A PredictionResult instance from schemas.py
        source: Where the prediction originated (e.g. "dashboard_live",
                "dashboard_explorer", "api")
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)

    record = {
        'timestamp': prediction_result.timestamp,
        'model_name': prediction_result.model_name,
        'model_version': prediction_result.model_version,
        'probability': round(prediction_result.probability, 6),
        'predicted_label': prediction_result.predicted_label,
        'confidence': prediction_result.confidence,
        'source': source,
        'input_features': {
            k: round(v, 4) for k, v in prediction_result.feature_values.items()
        },
    }

    if prediction_result.feature_warnings:
        record['warnings'] = prediction_result.feature_warnings

    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        logger.debug(
            f"Prediction logged: {prediction_result.model_name} -> "
            f"{prediction_result.predicted_label} ({prediction_result.probability:.3f})"
        )
    except IOError as e:
        logger.error(f"Failed to write audit log: {e}")


def get_recent_predictions(n: int = 50) -> List[dict]:
    """
    Read the last N predictions from the audit log.
    Returns newest-first.
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return []

    try:
        with open(AUDIT_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                records.append(json.loads(line))

        records.reverse()
        return records

    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read audit log: {e}")
        return []


def get_prediction_count() -> int:
    """Return total number of logged predictions."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return 0

    try:
        with open(AUDIT_LOG_PATH, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except IOError:
        return 0
