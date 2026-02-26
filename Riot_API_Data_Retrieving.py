import requests
import time
import random
import json
import os
from urllib.parse import quote
import pandas as pd

# Set your Riot API key
API_KEY = ''  # Replace with your Riot API Key
HEADERS = {'X-Riot-Token': API_KEY}
REGION = 'americas'

def get_match_ids_by_puuid(puuid, start=0, count=100):
    """ Get match IDs for a given player by their PUUID. """
    url = f'https://{REGION}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start={start}&count={count}'
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting match IDs for PUUID {puuid}: {e}")
        return []
    
def get_match_details(match_id):
    """ Get match details by match ID. """
    url = f'https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}'
    retries = 3
    sleep_time = 2
    while retries > 0:
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 429:  # Too Many Requests
                print("Rate limit hit. Retrying after delay...")
                time.sleep(sleep_time)
                sleep_time *= 2  # Exponential backoff
                retries -= 1
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting match details for match ID {match_id}: {e}")
            retries -= 1
            time.sleep(2)
    return {}

def collect_player_match_data(seed_puuid, max_players=10000, matches_per_player=100):
    """ Collect match data starting from a seed PUUID. """
    collected_puuids = set([seed_puuid])
    processed_matches = set()
    rows = []
    players_to_process = [seed_puuid]
    i=0

    start = 0
    while len(collected_puuids) < max_players and players_to_process:
        current_puuid = players_to_process.pop(0)
        print(f"Processing player {len(collected_puuids)} / {max_players} - {current_puuid}")
        
        while start<10000:
            match_ids = get_match_ids_by_puuid(current_puuid, start=start, count=matches_per_player)
            #print(match_ids)
            if not match_ids:
                break

            match_num=1
            for match_id in match_ids:
                #print("Match: ",match_num)
                if match_id in processed_matches:
                    continue
                processed_matches.add(match_id)

                # Rate limiting delay
                time.sleep(random.uniform(1, 2))
                match_data = get_match_details(match_id)
                if not match_data:
                    continue

                # Match-level metadata
                match_info = match_data.get("info", {})
                match_metadata = {
                    "endOfGameResult": match_info.get("endOfGameResult"),
                    "gameCreation": match_info.get("gameCreation"),
                    "gameDuration": match_info.get("gameDuration"),
                    "gameEndTimestamp": match_info.get("gameEndTimestamp"),
                    "gameId": match_info.get("gameId"),
                    "gameMode": match_info.get("gameMode"),
                    "gameName": match_info.get("gameName"),
                    "gameStartTimestamp": match_info.get("gameStartTimestamp"),
                    "gameType": match_info.get("gameType"),
                    "gameVersion": match_info.get("gameVersion"),
                    "mapId": match_info.get("mapId")
                }

                # Participant-level data
                for participant in match_info.get("participants", []):
                    row = {**match_metadata, **participant}
                    rows.append(row)

                # Discover new players
                competitor_puuids = extract_competitor_puuids(match_data, current_puuid)
                for puuid in competitor_puuids:
                    if puuid not in collected_puuids and len(collected_puuids) < max_players:
                        players_to_process.append(puuid)
                        collected_puuids.add(puuid)
                match_num+=1
            start += matches_per_player
            print("Simulation: ", start, f" Total players: {len(collected_puuids)}")
            i=i+1

        print(f"Finished processing {current_puuid}. Total players: {len(collected_puuids)}")
    return rows

def extract_competitor_puuids(match_data, exclude_puuid):
    """ Extract competitor PUUIDs from match data, excluding the provided PUUID. """
    if 'info' not in match_data or 'participants' not in match_data['info']:
        return []
    return [p['puuid'] for p in match_data['info']['participants'] if p['puuid'] != exclude_puuid]

# Replace this with the PUUID of a known player
seed_puuid = ''

# Collect match data
all_participant_data = collect_player_match_data(seed_puuid, max_players=10000, matches_per_player=100)

# Ensure the directory exists
output_dir = os.path.join(os.getcwd(), 'data')
os.makedirs(output_dir, exist_ok=True)

# Define file paths
json_output_path = os.path.join(output_dir, 'players_match_data_5000_100.json')
csv_output_path = os.path.join(output_dir, 'players_match_data_5000_100.csv')

# Save to JSON
try:
    with open(json_output_path, 'w') as f:
        json.dump(all_participant_data, f, indent=4)
    print(f"Data successfully saved to {json_output_path}")
except Exception as e:
    print(f"Failed to save JSON file: {e}")

# Save to CSV
try:
    match_data_df = pd.DataFrame(all_participant_data)
    match_data_df.to_csv(csv_output_path, index=False)
    print(f"Data successfully saved to {csv_output_path}")
except Exception as e:
    print(f"Failed to save CSV file: {e}")
