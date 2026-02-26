import csv
import pandas as pd

# Load CSV data manually without Pandas
file_path = './updated_players_match_data.csv'
data = []

# Read the CSV file with UTF-8 encoding
with open(file_path, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            # Convert specific columns to datetime or numeric as needed
            row['gameStartTimestamp'] = pd.to_datetime(int(row['gameStartTimestamp']) // 1000, unit='s')
            row['gameEndTimestamp'] = pd.to_datetime(int(row['gameEndTimestamp']) // 1000, unit='s')
            row['gameDuration'] = int(row['gameDuration'])
        
            # Handle 'win' column: convert 'True'/'False' to 1/0
            row['win'] = 1 if row['win'].strip().lower() == 'true' else 0
        
            row['kills'] = int(row['kills'])
            row['deaths'] = int(row['deaths'])
            row['champLevel'] = int(row['champLevel'])
            row['goldEarned'] = int(row['goldEarned'])
            row['goldSpent'] = int(row['goldSpent'])
            data.append(row)
        except Exception as e:
            print(f"Error processing row: {row}\nError: {e}")

# Group data manually by player
players_data = {}
for row in data:
    player_id = row['puuid']
    if player_id not in players_data:
        players_data[player_id] = []
    players_data[player_id].append(row)

# Compute features for each player
player_features = []

for player_id, games in players_data.items():
    # Sort games by start time
    games.sort(key=lambda x: x['gameStartTimestamp'])
    
    total_game_duration = 0
    total_kills = 0
    total_deaths = 0
    total_gold_usage = 0
    total_wins = 0
    time_differences = []
    unique_champions = set()
    dates_played = set()
    churn_flag = 0  # Used to track if the player churned at least once
    
    for i in range(len(games)):
        game = games[i]
        total_game_duration += game['gameDuration']
        total_kills += game['kills']
        total_deaths += game['deaths']
        total_gold_usage += game['goldEarned'] + game['goldSpent']
        total_wins += game['win']
        unique_champions.add(game['championName'])
        
        # Extract the start and end date for total_active_days calculation
        dates_played.add(game['gameStartTimestamp'].date())
        dates_played.add(game['gameEndTimestamp'].date())
        
        if i < len(games) - 1:  # Calculate the gap to the next game
            current_game_end = game['gameEndTimestamp']
            next_game_start = games[i + 1]['gameStartTimestamp']
            time_difference = (next_game_start - current_game_end).total_seconds()
            churn = 1 if time_difference > 24 * 3600 else 0
            games[i]['churn'] = churn
            time_differences.append(time_difference)
            if churn == 1:  # If churn is detected, set the churn flag to 1
                churn_flag = 1
        else:
            games[i]['churn'] = 0  # The last game for each player doesn't have a next game, so no churn is possible
    
    # Calculate total number of games
    total_games = len(games)
    num_gaps = len(time_differences)
    
    # Calculate each feature manually
    avg_game_duration = total_game_duration / total_games if total_games > 0 else 0
    avg_time_between_games = sum(time_differences) / num_gaps if num_gaps > 0 else 0
    
    # Calculate win/loss ratio
    total_losses = total_games - total_wins
    win_loss_ratio = total_wins / max(1, total_losses)
    
    unique_champions_count = len(unique_champions)
    kill_death_ratio = total_kills / max(1, total_deaths)
    avg_champion_level = round(sum([game['champLevel'] for game in games]) / total_games, 2) if total_games > 0 else 0
    avg_gold_usage = total_gold_usage / total_games if total_games > 0 else 0
    avg_daily_streak = len(dates_played)  # Number of distinct days played
    total_active_days = len(dates_played)  # Total distinct days where player was active
    
    player_features.append({
        'puuid': player_id,
        'total_game_duration': avg_game_duration,
        'avg_time_between_games': avg_time_between_games,
        'win_loss_ratio': win_loss_ratio,
        'unique_champions': unique_champions_count,
        'kill_death_ratio': kill_death_ratio,
        'avg_champion_level': avg_champion_level,
        'avg_gold_usage': avg_gold_usage,
        'avg_daily_streak': avg_daily_streak,
        'churn': churn_flag,  # New binary churn flag
        'total_active_days': total_active_days
    })

# Write player-level features to a CSV file
output_file_path = './player_features_manual_x.csv'
with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
    fieldnames = [
        'puuid', 'total_game_duration', 'avg_time_between_games', 'win_loss_ratio', 
        'unique_champions', 'kill_death_ratio', 'avg_champion_level', 'avg_gold_usage', 
        'avg_daily_streak', 'churn', 'total_active_days'
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for player_feature in player_features:
        writer.writerow(player_feature)

print(f"Feature extraction complete. Player features have been saved to '{output_file_path}'.")
