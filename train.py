"""
train.py
Trains multiple models for player churn prediction using temporal-split
features, evaluates them, and saves all artifacts to the models/ directory.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Masking
)
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2

from feature_engineering import load_and_clean, get_training_features, run_data_quality_checks
from feature_extraction_temporal import build_lstm_sequences, validate_temporal_integrity

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
MODEL_DIR = './models'
PLOT_DIR = os.path.join(MODEL_DIR, 'plots')
RAW_CSV = './data/raw_matches.csv'


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = load_and_clean()
    features = get_training_features()
    X = df[features]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, features, df


# ---------------------------------------------------------------------------
# 2. Feature scaling
# ---------------------------------------------------------------------------
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 3. Model training functions
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, 'logistic_regression.joblib'))
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight='balanced', random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, 'random_forest.joblib'))
    return model


def train_xgboost(X_train, y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(1, n_pos)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='logloss', use_label_encoder=False
    )
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgboost_model.joblib'))
    return model


def train_neural_network(X_train, y_train, X_test, y_test):
    """Feedforward Dense network for tabular features (20 inputs)."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    cw = {0: 1.0, 1: n_neg / max(1, n_pos)}

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=cw,
        verbose=0
    )
    model.save(os.path.join(MODEL_DIR, 'neural_network.keras'))

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, 'nn_history.json'), 'w') as f:
        json.dump(hist, f)

    return model, history


def train_lstm_sequential(X_train_seq, y_train, X_test_seq, y_test):
    """
    LSTM with proper sequential input — per-player game sequences.
    Input shape: (N, max_seq_len, n_per_game_features)
    This LSTM can genuinely learn temporal patterns (performance trends,
    gap patterns, declining engagement over recent games).
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    cw = {0: 1.0, 1: n_neg / max(1, n_pos)}

    seq_len = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Masking(mask_value=0.0),  # Ignore zero-padded timesteps
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.005))),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_seq, y_train,
        epochs=50, batch_size=32,
        validation_data=(X_test_seq, y_test),
        class_weight=cw,
        verbose=0
    )
    model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODEL_DIR, 'lstm_history.json'), 'w') as f:
        json.dump(hist, f)

    return model, history


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name, is_lstm_seq=False):
    """Evaluate a model and return metrics dict."""
    if is_lstm_seq:
        y_prob = model.predict(X_test, verbose=0).flatten()
    elif hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test, verbose=0).flatten()

    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Precision-recall curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'avg_precision': float(avg_prec),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'pr_precision': [float(x) for x in prec_curve],
        'pr_recall': [float(x) for x in rec_curve],
    }

    for metric_name, metric_fn in [
        ('accuracy', accuracy_score),
        ('precision', lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        ('recall', lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        ('f1', lambda yt, yp: f1_score(yt, yp, zero_division=0)),
    ]:
        ci = bootstrap_ci(y_test.values, y_pred, metric_fn)
        metrics[f'{metric_name}_ci'] = ci

    return metrics


def bootstrap_ci(y_true, y_pred, metric_fn, n_resamples=1000, alpha=0.05):
    np.random.seed(42)
    scores = []
    for _ in range(n_resamples):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return [lower, upper]


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
def compute_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    result = dict(zip(feature_names, [float(x) for x in importances]))
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(MODEL_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# 6. Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_models(X, y):
    """Run stratified 5-fold CV on sklearn models + baseline. Returns per-fold metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_configs = {
        'Baseline (Majority)': lambda _yt: DummyClassifier(
            strategy='most_frequent', random_state=42),
        'Logistic Regression': lambda _yt: LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': lambda _yt: RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
        'XGBoost': lambda yt: XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=(yt == 0).sum() / max(1, (yt == 1).sum()),
            random_state=42, eval_metric='logloss', use_label_encoder=False),
    }

    cv_results = {}

    for model_name, model_factory in model_configs.items():
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_fold)
            X_test_s = scaler.transform(X_test_fold)

            model = model_factory(y_train_fold)
            model.fit(X_train_s, y_train_fold)

            y_prob = model.predict_proba(X_test_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            fold_metrics.append({
                'fold': fold_idx,
                'accuracy': float(accuracy_score(y_test_fold, y_pred)),
                'precision': float(precision_score(y_test_fold, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test_fold, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test_fold, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(y_test_fold, y_prob)),
            })

        metrics_arrays = {
            metric: [f[metric] for f in fold_metrics]
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        }

        cv_results[model_name] = {
            'per_fold': fold_metrics,
            'mean': {m: float(np.mean(v)) for m, v in metrics_arrays.items()},
            'std': {m: float(np.std(v)) for m, v in metrics_arrays.items()},
        }

    with open(os.path.join(MODEL_DIR, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    return cv_results


# ---------------------------------------------------------------------------
# 7. Learning curves
# ---------------------------------------------------------------------------
def compute_learning_curves(X, y):
    """Compute learning curves for LR and RF (fast, representative)."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lc_results = {}

    for name, model in models.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_scaled, y,
            train_sizes=np.linspace(0.2, 1.0, 8),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
        )
        lc_results[name] = {
            'train_sizes': train_sizes.tolist(),
            'train_mean': np.mean(train_scores, axis=1).tolist(),
            'train_std': np.std(train_scores, axis=1).tolist(),
            'test_mean': np.mean(test_scores, axis=1).tolist(),
            'test_std': np.std(test_scores, axis=1).tolist(),
        }

    with open(os.path.join(MODEL_DIR, 'learning_curves.json'), 'w') as f:
        json.dump(lc_results, f, indent=2)

    return lc_results


# ---------------------------------------------------------------------------
# 8. Comparison + plots
# ---------------------------------------------------------------------------
def save_comparison(all_results, feature_names):
    """Save model comparison metrics and generate plots."""
    with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(MODEL_DIR, 'training_features.json'), 'w') as f:
        json.dump(feature_names, f)

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, metrics in all_results.items():
        ax.plot(metrics['fpr'], metrics['tpr'],
                label=f"{name} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Model Comparison')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'roc_curves.png'), dpi=150)
    plt.close(fig)

    # Confusion matrices
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, metrics) in zip(axes, all_results.items()):
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Active', 'Churned'],
                    yticklabels=['Active', 'Churned'])
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'confusion_matrices.png'), dpi=150)
    plt.close(fig)

    # Summary table
    summary_rows = []
    for name, metrics in all_results.items():
        summary_rows.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


def save_training_plots(nn_hist_path, lstm_hist_path):
    """Plot training history for NN and LSTM side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (hist_path, label) in enumerate([
        (nn_hist_path, 'Dense NN'),
        (lstm_hist_path, 'LSTM (Sequential)'),
    ]):
        with open(hist_path) as f:
            hist = json.load(f)

        axes[0][col].plot(hist['accuracy'], label='Train')
        axes[0][col].plot(hist['val_accuracy'], label='Validation')
        axes[0][col].set_title(f'{label} — Accuracy')
        axes[0][col].set_xlabel('Epoch')
        axes[0][col].set_ylabel('Accuracy')
        axes[0][col].legend()

        axes[1][col].plot(hist['loss'], label='Train')
        axes[1][col].plot(hist['val_loss'], label='Validation')
        axes[1][col].set_title(f'{label} — Loss')
        axes[1][col].set_xlabel('Epoch')
        axes[1][col].set_ylabel('Loss')
        axes[1][col].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, 'training_history.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ensure_dirs()

    print("Loading data...")
    X_train, X_test, y_train, y_test, feature_names, df = load_data()

    print(f"Dataset: {len(df)} players, {len(feature_names)} features")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Churn rate: {df['churn'].mean():.1%}")

    # --- Data quality checks ---
    print("Running data quality checks...")
    dq_report = run_data_quality_checks(df)
    n_passed = sum(1 for c in dq_report['checks'] if c['passed'])
    print(f"  Data quality: {n_passed}/{len(dq_report['checks'])} checks passed")

    # --- Leakage detection ---
    print("Running temporal integrity check...")
    integrity = validate_temporal_integrity()
    print(f"  Leakage check: {'PASSED' if integrity['passed'] else 'FAILED'} "
          f"({integrity['n_players_checked']} players verified)")

    print("Scaling features...")
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    all_results = {}

    # --- Baseline ---
    print("Training Baseline (Majority Class)...")
    baseline = DummyClassifier(strategy='most_frequent', random_state=42)
    baseline.fit(X_train_s, y_train)
    all_results['Baseline (Majority)'] = evaluate_model(
        baseline, X_test_s, y_test, 'Baseline (Majority)'
    )

    # --- Sklearn models ---
    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_train_s, y_train)
    all_results['Logistic Regression'] = evaluate_model(
        lr, X_test_s, y_test, 'Logistic Regression'
    )

    print("Training Random Forest...")
    rf = train_random_forest(X_train_s, y_train)
    all_results['Random Forest'] = evaluate_model(
        rf, X_test_s, y_test, 'Random Forest'
    )

    print("Training XGBoost...")
    xgb = train_xgboost(X_train_s, y_train)
    all_results['XGBoost'] = evaluate_model(
        xgb, X_test_s, y_test, 'XGBoost'
    )

    # --- Dense NN ---
    print("Training Dense Neural Network (50 epochs)...")
    nn, nn_hist = train_neural_network(X_train_s, y_train, X_test_s, y_test)
    all_results['Dense NN'] = evaluate_model(
        nn, X_test_s, y_test, 'Dense NN'
    )

    # --- LSTM with proper sequences ---
    print("Building LSTM sequences from raw match data...")
    X_seq_all, seq_feature_names = build_lstm_sequences(RAW_CSV, df)

    # Split sequences using the same train/test indices
    train_idx = X_train.index
    test_idx = X_test.index
    # Map to positional indices in df
    train_pos = [df.index.get_loc(i) for i in train_idx]
    test_pos = [df.index.get_loc(i) for i in test_idx]
    X_train_seq = X_seq_all[train_pos]
    X_test_seq = X_seq_all[test_pos]

    # Scale sequences per feature
    seq_scaler = StandardScaler()
    n_samples, seq_len, n_feats = X_train_seq.shape
    X_train_seq_flat = X_train_seq.reshape(-1, n_feats)
    X_test_seq_flat = X_test_seq.reshape(-1, n_feats)
    seq_scaler.fit(X_train_seq_flat[X_train_seq_flat.sum(axis=1) != 0])  # Fit on non-padded
    X_train_seq = seq_scaler.transform(X_train_seq_flat).reshape(n_samples, seq_len, n_feats)
    X_test_seq = seq_scaler.transform(X_test_seq_flat).reshape(len(test_pos), seq_len, n_feats)
    # Re-zero the padded rows
    for i in range(len(train_pos)):
        mask = X_seq_all[train_pos[i]].sum(axis=1) == 0
        X_train_seq[i][mask] = 0
    for i in range(len(test_pos)):
        mask = X_seq_all[test_pos[i]].sum(axis=1) == 0
        X_test_seq[i][mask] = 0

    joblib.dump(seq_scaler, os.path.join(MODEL_DIR, 'seq_scaler.joblib'))

    print(f"LSTM input shape: {X_train_seq.shape}")
    print("Training LSTM with sequential input (50 epochs)...")
    lstm, lstm_hist = train_lstm_sequential(
        X_train_seq, y_train, X_test_seq, y_test
    )
    all_results['LSTM (Sequential)'] = evaluate_model(
        lstm, X_test_seq, y_test, 'LSTM (Sequential)', is_lstm_seq=True
    )

    # --- Feature importance ---
    print("Computing feature importance...")
    compute_feature_importance(rf, feature_names)

    # --- Save comparison + plots ---
    save_comparison(all_results, feature_names)
    save_training_plots(
        os.path.join(MODEL_DIR, 'nn_history.json'),
        os.path.join(MODEL_DIR, 'lstm_history.json'),
    )

    # --- Cross-validation (sklearn models only — NN/LSTM too slow for 5x) ---
    print("\nRunning 5-fold cross-validation on sklearn models...")
    cv_results = cross_validate_models(df[feature_names], df['churn'])
    for name, res in cv_results.items():
        print(f"  {name}: F1={res['mean']['f1']:.3f} +/- {res['std']['f1']:.3f}, "
              f"AUC={res['mean']['roc_auc']:.3f} +/- {res['std']['roc_auc']:.3f}")

    # --- Learning curves ---
    print("\nComputing learning curves...")
    lc_results = compute_learning_curves(df[feature_names], df['churn'])
    for name in lc_results:
        print(f"  {name}: done")

    print("\nAll models and artifacts saved to ./models/")
    print("Run: streamlit run app.py")


if __name__ == '__main__':
    main()
