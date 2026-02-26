"""
collect_data.py
Collects League of Legends match data from the Riot API with a focus on
getting players with >= MIN_MATCHES games for meaningful temporal analysis.

Usage:
    1. Paste your Riot API key into .env file: RIOT_API_KEY=your_key_here
    2. Run: python collect_data.py
    3. The script saves progress and can be resumed if interrupted (Ctrl+C)

Output: data/raw_matches.csv
"""
import requests
import time
import json
import csv
import os
import sys
from dotenv import load_dotenv

# CONFIGURATION — API key loaded from .env file
load_dotenv()
API_KEY = os.getenv('RIOT_API_KEY', '')
REGION = 'americas'
HEADERS = {'X-Riot-Token': API_KEY}

# Collection parameters
MAX_QUALIFIED_PLAYERS = 250   # Stop after collecting this many qualified players
MIN_MATCHES = 10              # Skip players with fewer matches
MATCHES_TO_FETCH = 50         # Fetch up to this many match details per player

# File paths
DATA_DIR = './data'
RAW_CSV = os.path.join(DATA_DIR, 'raw_matches.csv')
PROGRESS_FILE = os.path.join(DATA_DIR, 'collection_progress.json')

# Seed player (most active player from existing dataset)
SEED_PUUID = 'pSh6OFJ0vWaw88tYI6SimBJm8FgIZe_lyOLwdhj1441kzLFYSFvvBRBRJa3aUv8fYy-USRHptsoy0w'

# CSV columns to save per match
CSV_FIELDS = [
    'puuid', 'matchId', 'gameStartTimestamp', 'gameEndTimestamp',
    'gameDuration', 'gameMode', 'gameType',
    'kills', 'deaths', 'assists', 'champLevel',
    'championName', 'goldEarned', 'goldSpent',
    'totalDamageDealtToChampions', 'totalMinionsKilled',
    'visionScore', 'win',
]


class RateLimiter:
    def __init__(self, max_calls=95, period=120):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = []

    def wait(self):
        now = time.time()
        # Remove timestamps older than the period
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self.timestamps[0]) + 1
            print(f"  Rate limit: sleeping {sleep_time:.0f}s...")
            time.sleep(max(sleep_time, 0))
        self.timestamps.append(time.time())
        
rate_limiter = RateLimiter()


def api_get(url, retries=3):
    """Make a rate-limited GET request with retry on 429."""
    for attempt in range(retries):
        rate_limiter.wait()
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get('Retry-After', 10))
                print(f"  429 Too Many Requests. Waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                continue
            if resp.status_code == 403:
                print("ERROR: 403 Forbidden. Your API key may be expired.")
                print("Get a new key at https://developer.riotgames.com/")
                sys.exit(1)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


def get_match_ids(puuid, count=100):
    """Get match IDs for a player. Returns list of match ID strings."""
    url = (f'https://{REGION}.api.riotgames.com/lol/match/v5/matches/'
           f'by-puuid/{puuid}/ids?start=0&count={count}')
    result = api_get(url)
    return result if result else []


def get_match_detail(match_id):
    """Get full match detail by match ID."""
    url = f'https://{REGION}.api.riotgames.com/lol/match/v5/matches/{match_id}'
    return api_get(url)


def extract_player_row(match_data, puuid):
    """Extract the target player's participant row from match data."""
    info = match_data.get('info', {})
    match_id = match_data.get('metadata', {}).get('matchId', '')

    for participant in info.get('participants', []):
        if participant.get('puuid') == puuid:
            return {
                'puuid': puuid,
                'matchId': match_id,
                'gameStartTimestamp': info.get('gameStartTimestamp'),
                'gameEndTimestamp': info.get('gameEndTimestamp'),
                'gameDuration': info.get('gameDuration'),
                'gameMode': info.get('gameMode'),
                'gameType': info.get('gameType'),
                'kills': participant.get('kills'),
                'deaths': participant.get('deaths'),
                'assists': participant.get('assists'),
                'champLevel': participant.get('champLevel'),
                'championName': participant.get('championName'),
                'goldEarned': participant.get('goldEarned'),
                'goldSpent': participant.get('goldSpent'),
                'totalDamageDealtToChampions': participant.get('totalDamageDealtToChampions'),
                'totalMinionsKilled': participant.get('totalMinionsKilled'),
                'visionScore': participant.get('visionScore'),
                'win': participant.get('win'),
            }
    return None


def extract_all_puuids(match_data, exclude_puuid):
    """Get all participant PUUIDs from a match except the given one."""
    participants = match_data.get('info', {}).get('participants', [])
    return [p['puuid'] for p in participants
            if p.get('puuid') and p['puuid'] != exclude_puuid]


# Progress Management
def save_progress(state):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(state, f)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return None


# Main Collection Loop
def collect():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not API_KEY:
        print("ERROR: Paste your Riot API key in the API_KEY variable at the top of this file.")
        sys.exit(1)

    # Check for resume
    state = load_progress()
    if state:
        queue = state['queue']
        seen_puuids = set(state['seen_puuids'])
        qualified_count = state['qualified_count']
        processed_matches = set(state.get('processed_matches', []))
        total_rows = state.get('total_rows', 0)
        print(f"Resuming: {qualified_count} qualified players, "
              f"{len(seen_puuids)} seen, {len(queue)} in queue")
    else:
        queue = [SEED_PUUID]
        seen_puuids = {SEED_PUUID}
        qualified_count = 0
        processed_matches = set()
        total_rows = 0

    # Initialize CSV if not resuming
    csv_exists = os.path.exists(RAW_CSV) and state is not None
    csv_file = open(RAW_CSV, 'a' if csv_exists else 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()

    players_checked = 0

    try:
        while queue and qualified_count < MAX_QUALIFIED_PLAYERS:
            puuid = queue.pop(0)
            players_checked += 1

            # Step 1: Get match IDs (1 API call)
            match_ids = get_match_ids(puuid, count=100)

            if len(match_ids) < MIN_MATCHES:
                # Not enough matches — skip but discover players from first match
                if match_ids:
                    first_match = get_match_detail(match_ids[0])
                    if first_match:
                        for new_puuid in extract_all_puuids(first_match, puuid):
                            if new_puuid not in seen_puuids:
                                seen_puuids.add(new_puuid)
                                queue.append(new_puuid)
                if players_checked % 20 == 0:
                    print(f"  Checked {players_checked} players, "
                          f"{qualified_count} qualified, {len(queue)} in queue")
                continue

            # Step 2: This player qualifies — fetch match details
            qualified_count += 1
            matches_to_get = match_ids[:MATCHES_TO_FETCH]
            player_rows = 0

            print(f"[{qualified_count}/{MAX_QUALIFIED_PLAYERS}] "
                  f"Player {puuid[:16]}... — {len(match_ids)} matches, "
                  f"fetching {len(matches_to_get)}")

            for match_id in matches_to_get:
                if match_id in processed_matches:
                    continue
                processed_matches.add(match_id)

                match_data = get_match_detail(match_id)
                if not match_data:
                    continue

                # Extract target player's row
                row = extract_player_row(match_data, puuid)
                if row:
                    writer.writerow(row)
                    player_rows += 1
                    total_rows += 1

                # Discover new players
                for new_puuid in extract_all_puuids(match_data, puuid):
                    if new_puuid not in seen_puuids:
                        seen_puuids.add(new_puuid)
                        queue.append(new_puuid)

            csv_file.flush()
            print(f"  Saved {player_rows} matches. Total rows: {total_rows}")

            # Save progress every 5 qualified players
            if qualified_count % 5 == 0:
                save_progress({
                    'queue': queue[:5000],  # Cap queue size in checkpoint
                    'seen_puuids': list(seen_puuids)[:50000],
                    'qualified_count': qualified_count,
                    'processed_matches': list(processed_matches),
                    'total_rows': total_rows,
                })
                print(f"  Progress saved. Queue: {len(queue)}")

    except KeyboardInterrupt:
        print("\nInterrupted! Saving progress...")
        save_progress({
            'queue': queue[:5000],
            'seen_puuids': list(seen_puuids)[:50000],
            'qualified_count': qualified_count,
            'processed_matches': list(processed_matches),
            'total_rows': total_rows,
        })
        print("Progress saved. Run again to resume.")
    finally:
        csv_file.close()

    print(f"\nCollection complete!")
    print(f"  Qualified players: {qualified_count}")
    print(f"  Total match rows: {total_rows}")
    print(f"  Players checked: {players_checked}")
    print(f"  Output: {RAW_CSV}")


if __name__ == '__main__':
    collect()
