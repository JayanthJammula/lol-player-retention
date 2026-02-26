"""
feature_extraction_temporal.py
Reads raw match-level data and produces player-level features using a
temporal split: features from the first 70% of games, churn label from
the remaining 30%.

Usage: python feature_extraction_temporal.py [--input data/raw_matches.csv]
"""
import pandas as pd
import numpy as np
import os
import argparse

# Configuration
FEATURE_SPLIT_RATIO = 0.7   # First 70% of games → features
MIN_GAMES = 10               # Minimum total games per player
CHURN_THRESHOLD_DAYS = 0.25  # Gap > 6 hours between windows → disengagement

OUTPUT_DIR = './data'
DEFAULT_INPUT = os.path.join(OUTPUT_DIR, 'raw_matches.csv')
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, 'player_features_temporal.csv')


# Feature extraction
def extract_temporal_features(input_csv, output_csv):
    """Extract player-level temporal features from raw match data."""
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)

    # Convert timestamps
    df['game_start_dt'] = pd.to_datetime(df['gameStartTimestamp'], unit='ms')
    df['game_end_dt'] = pd.to_datetime(df['gameEndTimestamp'], unit='ms')

    # Ensure boolean win column
    if df['win'].dtype == object:
        df['win'] = df['win'].str.strip().str.lower() == 'true'
    df['win'] = df['win'].astype(int)

    # Sort chronologically per player
    df = df.sort_values(['puuid', 'gameStartTimestamp']).reset_index(drop=True)

    # Group by player
    grouped = df.groupby('puuid')
    print(f"Total players: {grouped.ngroups}")
    print(f"Players with >= {MIN_GAMES} games: "
          f"{(grouped.size() >= MIN_GAMES).sum()}")

    player_features = []
    skipped = 0

    for puuid, games in grouped:
        games = games.reset_index(drop=True)
        n = len(games)

        if n < MIN_GAMES:
            skipped += 1
            continue

        # --- Temporal split ---
        split_idx = int(n * FEATURE_SPLIT_RATIO)
        split_idx = max(split_idx, 3)  # At least 3 games in feature window
        feature_games = games.iloc[:split_idx]
        label_games = games.iloc[split_idx:]

        # --- Churn label (forward-looking) ---
        last_feature_time = feature_games['game_end_dt'].iloc[-1]
        if len(label_games) == 0:
            churn = 1
        else:
            first_label_time = label_games['game_start_dt'].iloc[0]
            gap_days = (first_label_time - last_feature_time).total_seconds() / 86400
            churn = 1 if gap_days > CHURN_THRESHOLD_DAYS else 0

        # --- Compute features from feature_games ONLY ---
        fg = feature_games
        n_feat = len(fg)

        # Performance
        win_rate = fg['win'].mean()
        total_kills = fg['kills'].sum()
        total_deaths = fg['deaths'].sum()
        total_assists = fg['assists'].sum()
        kda = (total_kills + total_assists) / max(1, total_deaths)
        kill_death_ratio = total_kills / max(1, total_deaths)

        # Game context
        avg_game_duration = fg['gameDuration'].mean()
        avg_champion_level = fg['champLevel'].mean()
        avg_gold_earned = fg['goldEarned'].mean()
        gold_earned_total = fg['goldEarned'].sum()
        gold_spent_total = fg['goldSpent'].sum()
        gold_efficiency = gold_spent_total / max(1, gold_earned_total)
        unique_champions = fg['championName'].nunique()

        # Damage/CS/Vision
        avg_damage = fg['totalDamageDealtToChampions'].mean()
        avg_cs = fg['totalMinionsKilled'].mean()
        avg_vision = fg['visionScore'].mean()

        # Engagement
        total_games_played = n_feat
        unique_play_days = fg['game_start_dt'].dt.date.nunique()

        time_diffs = fg['game_start_dt'].diff().dt.total_seconds().dropna()
        time_diffs = time_diffs.clip(lower=0)  # Clamp negatives

        avg_time_between = time_diffs.mean() / 3600 if len(time_diffs) > 0 else 0
        median_time_between = time_diffs.median() / 3600 if len(time_diffs) > 0 else 0

        window_start = fg['game_start_dt'].iloc[0]
        window_end = fg['game_end_dt'].iloc[-1]
        feature_window_days = max(
            (window_end - window_start).total_seconds() / 86400, 0.01
        )
        play_frequency = n_feat / feature_window_days

        # Trends: first half vs second half of feature window
        mid = n_feat // 2
        if mid >= 2:
            first_half = fg.iloc[:mid]
            second_half = fg.iloc[mid:]

            kda_first = ((first_half['kills'].sum() + first_half['assists'].sum())
                         / max(1, first_half['deaths'].sum()))
            kda_second = ((second_half['kills'].sum() + second_half['assists'].sum())
                          / max(1, second_half['deaths'].sum()))
            kda_trend = kda_second - kda_first

            wr_first = first_half['win'].mean()
            wr_second = second_half['win'].mean()
            winrate_trend = wr_second - wr_first
        else:
            kda_trend = 0.0
            winrate_trend = 0.0

        # Recency: gap (in days) between the last two games in feature window
        if len(time_diffs) > 0:
            last_gap_days = time_diffs.iloc[-1] / 86400
        else:
            last_gap_days = 0.0

        player_features.append({
            'puuid': puuid,
            # Performance
            'win_rate': round(win_rate, 4),
            'kda': round(kda, 4),
            'kill_death_ratio': round(kill_death_ratio, 4),
            'avg_damage': round(avg_damage, 2),
            'avg_cs': round(avg_cs, 2),
            'avg_vision_score': round(avg_vision, 2),
            # Game context
            'avg_game_duration': round(avg_game_duration, 2),
            'avg_champion_level': round(avg_champion_level, 2),
            'avg_gold_earned': round(avg_gold_earned, 2),
            'gold_efficiency': round(gold_efficiency, 4),
            'unique_champions': unique_champions,
            # Engagement
            'total_games_played': total_games_played,
            'unique_play_days': unique_play_days,
            'avg_time_between_games_hrs': round(avg_time_between, 4),
            'median_time_between_games_hrs': round(median_time_between, 4),
            'play_frequency': round(play_frequency, 4),
            'feature_window_days': round(feature_window_days, 2),
            # Trends
            'kda_trend': round(kda_trend, 4),
            'winrate_trend': round(winrate_trend, 4),
            'last_gap_days': round(last_gap_days, 4),
            # Label
            'churn': churn,
        })

    result_df = pd.DataFrame(player_features)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print(f"\nResults:")
    print(f"  Players processed: {len(player_features)}")
    print(f"  Players skipped (< {MIN_GAMES} games): {skipped}")
    print(f"  Churn rate: {result_df['churn'].mean():.1%}")
    print(f"  Churn distribution:\n{result_df['churn'].value_counts().to_string()}")
    print(f"  Features: {len(result_df.columns) - 2} (excluding puuid & churn)")
    print(f"  Output: {output_csv}")

    return result_df


# ---------------------------------------------------------------------------
# Temporal integrity validation (leakage detection)
# ---------------------------------------------------------------------------
def validate_temporal_integrity(input_csv=None, features_csv=None,
                                n_sample=20, save_path='./models/leakage_check.json'):
    """
    Programmatic check that no label-window data leaked into features.
    Samples n_sample players and verifies:
      1. Timestamps are monotonically increasing
      2. Feature window ends before label window starts
      3. Recomputed churn label matches stored label
      4. Feature game count matches total_games_played
    """
    import json as _json

    if input_csv is None:
        input_csv = DEFAULT_INPUT
    if features_csv is None:
        features_csv = DEFAULT_OUTPUT

    raw = pd.read_csv(input_csv, low_memory=False)
    features_df = pd.read_csv(features_csv)

    raw['game_start_dt'] = pd.to_datetime(raw['gameStartTimestamp'], unit='ms')
    raw['game_end_dt'] = pd.to_datetime(raw['gameEndTimestamp'], unit='ms')
    raw = raw.sort_values(['puuid', 'gameStartTimestamp']).reset_index(drop=True)

    results = {'passed': True, 'checks': [], 'n_players_checked': 0}

    sample_puuids = features_df['puuid'].sample(
        n=min(n_sample, len(features_df)), random_state=42
    ).values

    for puuid in sample_puuids:
        games = raw[raw['puuid'] == puuid].sort_values('gameStartTimestamp')
        n = len(games)
        if n < MIN_GAMES:
            continue

        split_idx = max(int(n * FEATURE_SPLIT_RATIO), 3)
        feat_games = games.iloc[:split_idx]
        label_games = games.iloc[split_idx:]

        # Check 1: Timestamps monotonically increasing
        timestamps = games['gameStartTimestamp'].values
        monotonic = bool(all(
            timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
        ))

        # Check 2: Feature window ends before label window starts
        if len(label_games) > 0:
            feat_end = feat_games['game_end_dt'].iloc[-1]
            label_start = label_games['game_start_dt'].iloc[0]
            no_overlap = bool(feat_end <= label_start)
        else:
            no_overlap = True

        # Check 3: Recompute churn label independently
        player_row = features_df[features_df['puuid'] == puuid].iloc[0]
        if len(label_games) == 0:
            expected_churn = 1
        else:
            gap = (label_games['game_start_dt'].iloc[0] -
                   feat_games['game_end_dt'].iloc[-1]).total_seconds() / 86400
            expected_churn = 1 if gap > CHURN_THRESHOLD_DAYS else 0
        churn_matches = bool(int(player_row['churn']) == expected_churn)

        # Check 4: Feature game count matches total_games_played
        games_match = bool(int(player_row['total_games_played']) == split_idx)

        check = {
            'puuid_prefix': puuid[:16] + '...',
            'total_games': int(n),
            'feature_games': int(split_idx),
            'monotonic_timestamps': monotonic,
            'no_temporal_overlap': no_overlap,
            'churn_label_correct': churn_matches,
            'game_count_correct': games_match,
            'all_passed': monotonic and no_overlap and churn_matches and games_match,
        }
        results['checks'].append(check)
        results['n_players_checked'] += 1

        if not check['all_passed']:
            results['passed'] = False

    # Save
    if save_path:
        import os as _os
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            _json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------
def build_lstm_sequences(input_csv, player_df, max_seq_len=20):
    """
    Build per-player game sequences for LSTM input.
    Uses the same temporal split — only feature-window games.
    Returns: X_seq array of shape (n_players, max_seq_len, n_seq_features)
    """
    raw_df = pd.read_csv(input_csv, low_memory=False)

    # Ensure boolean win column
    if raw_df['win'].dtype == object:
        raw_df['win'] = raw_df['win'].str.strip().str.lower() == 'true'
    raw_df['win'] = raw_df['win'].astype(int)

    raw_df = raw_df.sort_values(['puuid', 'gameStartTimestamp']).reset_index(drop=True)

    # Per-game features for the sequence
    seq_feature_cols = [
        'gameDuration', 'kills', 'deaths', 'assists',
        'champLevel', 'goldEarned', 'goldSpent',
        'totalDamageDealtToChampions', 'totalMinionsKilled',
        'visionScore', 'win',
    ]

    sequences = {}
    for puuid, games in raw_df.groupby('puuid'):
        n = len(games)
        if n < MIN_GAMES:
            continue
        split_idx = max(int(n * FEATURE_SPLIT_RATIO), 3)
        feature_games = games.iloc[:split_idx]

        # Take last max_seq_len games from feature window
        seq_data = feature_games[seq_feature_cols].tail(max_seq_len).values
        sequences[puuid] = seq_data

    # Build aligned array matching player_df row order
    puuids = player_df['puuid'].values
    n_seq_features = len(seq_feature_cols)
    X_seq = np.zeros((len(puuids), max_seq_len, n_seq_features))

    for i, puuid in enumerate(puuids):
        if puuid in sequences:
            seq = sequences[puuid]
            # Right-align: pad zeros on left, actual games on right
            X_seq[i, -len(seq):, :] = seq

    return X_seq, seq_feature_cols


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='Path to raw matches CSV')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Path for output features CSV')
    args = parser.parse_args()

    extract_temporal_features(args.input, args.output)
