"""
logging_config.py
Centralized logging configuration for the churn prediction pipeline.
Replaces scattered print() calls with structured, leveled logging.
"""
import logging
import os
from datetime import datetime


def setup_logging(log_dir='./logs', level=logging.INFO):
    """
    Configure project-wide logging with console and file handlers.

    Logger hierarchy:
        churn_predictor           (root for this project)
        churn_predictor.train     (training pipeline)
        churn_predictor.predict   (prediction / inference)
        churn_predictor.data      (data loading, feature engineering)
        churn_predictor.collect   (API data collection)
    """
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger('churn_predictor')
    root_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls (e.g. Streamlit reruns)
    if root_logger.handlers:
        return root_logger

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (INFO and above)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # Persistent file handler (DEBUG and above)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'churn_predictor.log'),
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return root_logger


def get_training_logger(log_dir='./logs'):
    """
    Return a logger that also writes to a timestamped training log file.
    Call this at the start of a training run for per-run log files.
    """
    logger = logging.getLogger('churn_predictor.train')

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(
        os.path.join(log_dir, f'training_{timestamp}.log'),
        encoding='utf-8',
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    return logger
