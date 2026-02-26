"""
feature_engineering.py
Loads the temporal-split player features CSV and provides helper functions
for loading data and defining training features.
"""
import pandas as pd
import numpy as np
import json
import os


def load_and_clean(path='./data/player_features_temporal.csv'):
    """Load the temporal-split features CSV and apply cleaning."""
    df = pd.read_csv(path)

    # Clamp time-based features to non-negative
    for col in ['avg_time_between_games_hrs', 'median_time_between_games_hrs',
                'last_gap_days', 'feature_window_days']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Clamp extreme ratios
    for col in ['kda', 'kill_death_ratio']:
        if col in df.columns:
            df[col] = df[col].clip(upper=50)

    return df


def get_training_features():
    """
    Return feature columns for model training.
    All features are computed from the historical (feature) window only —
    no data leakage.
    """
    return [
        # Performance
        'win_rate',
        'kda',
        'kill_death_ratio',
        'avg_damage',
        'avg_cs',
        'avg_vision_score',
        # Game context
        'avg_game_duration',
        'avg_champion_level',
        'avg_gold_earned',
        'gold_efficiency',
        'unique_champions',
        # Engagement
        'total_games_played',
        'unique_play_days',
        'avg_time_between_games_hrs',
        'median_time_between_games_hrs',
        'play_frequency',
        'feature_window_days',
        # Trends
        'kda_trend',
        'winrate_trend',
        'last_gap_days',
    ]


def get_all_features():
    """Return all feature columns (same as training — no leakage exclusions)."""
    return get_training_features()


def run_data_quality_checks(df=None, save_path='./models/data_quality.json'):
    """Run data quality assertions and return a report dict."""
    if df is None:
        df = load_and_clean()

    features = get_training_features()
    report = {'passed': True, 'total_players': len(df), 'checks': []}

    # Check 1: No missing values in features
    missing = df[features].isnull().sum()
    missing_any = missing[missing > 0]
    report['checks'].append({
        'name': 'No missing values in features',
        'passed': len(missing_any) == 0,
        'detail': missing_any.to_dict() if len(missing_any) > 0 else {},
    })

    # Check 2: Feature range checks
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
        if feat in df.columns:
            out_of_range = int(((df[feat] < low) | (df[feat] > high)).sum())
            check_passed = out_of_range == 0
            report['checks'].append({
                'name': f'{feat} in [{low}, {high}]',
                'passed': check_passed,
                'violations': out_of_range,
            })
            if not check_passed:
                report['passed'] = False

    # Check 3: Churn rate between 5% and 50%
    churn_rate = float(df['churn'].mean())
    report['checks'].append({
        'name': 'Churn rate between 5% and 50%',
        'passed': 0.05 <= churn_rate <= 0.50,
        'churn_rate': round(churn_rate, 4),
    })
    if not (0.05 <= churn_rate <= 0.50):
        report['passed'] = False

    # Check 4: No duplicate PUUIDs
    n_dupes = int(df['puuid'].duplicated().sum())
    report['checks'].append({
        'name': 'No duplicate player IDs',
        'passed': n_dupes == 0,
        'duplicates': n_dupes,
    })
    if n_dupes > 0:
        report['passed'] = False

    # Check 5: No infinite values
    inf_count = int(np.isinf(df[features].select_dtypes(include=[np.number])).sum().sum())
    report['checks'].append({
        'name': 'No infinite values',
        'passed': inf_count == 0,
        'count': inf_count,
    })
    if inf_count > 0:
        report['passed'] = False

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == '__main__':
    df = load_and_clean()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Training features ({len(get_training_features())}): {get_training_features()}")
    print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
    print(f"Churn rate: {df['churn'].mean():.1%}")
