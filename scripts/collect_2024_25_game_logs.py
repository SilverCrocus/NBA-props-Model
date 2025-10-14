"""
Collect 2024-25 NBA Season Game Logs

Fetches all game logs from the completed 2024-25 season (Oct 2024 - June 2025)
to use for backtesting the model on fresh, unseen data.
"""

import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time
from pathlib import Path
from tqdm import tqdm
import json

# Configuration
SEASON = "2024-25"
OUTPUT_DIR = Path("data/game_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"game_logs_{SEASON.replace('-', '_')}.csv"
PROGRESS_FILE = OUTPUT_DIR / f"collection_progress_{SEASON.replace('-', '_')}.json"

print("="*80)
print(f"NBA {SEASON} SEASON - GAME LOG COLLECTOR")
print("="*80)
print(f"\nOutput: {OUTPUT_FILE}")

# Load progress if exists
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_players": [], "total_games": 0}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

progress = load_progress()
print(f"\nProgress: {len(progress['completed_players'])} players already collected")
print(f"Total games so far: {progress['total_games']}")

# Get all NBA players
print("\nFetching active NBA players...")
all_players = players.get_active_players()
print(f"Found {len(all_players)} active players")

# Load existing game logs if they exist
all_game_logs = []
if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    all_game_logs = existing.to_dict('records')
    print(f"Loaded {len(all_game_logs):,} existing game logs")

# Collect game logs
print(f"\n" + "="*80)
print(f"COLLECTING {SEASON} GAME LOGS")
print("="*80)

failed_players = []

for player in tqdm(all_players, desc="Collecting game logs"):
    player_id = player['id']
    player_name = player['full_name']

    # Skip if already completed
    if player_id in progress['completed_players']:
        continue

    try:
        # Fetch game logs for this player in 2024-25 season
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=SEASON,
            season_type_all_star='Regular Season'
        )

        # Get the dataframe
        df = gamelog.get_data_frames()[0]

        if len(df) > 0:
            # Add player info
            df['PLAYER_ID'] = player_id
            df['PLAYER_NAME'] = player_name

            # Add to master list
            all_game_logs.extend(df.to_dict('records'))
            progress['total_games'] += len(df)

            # Mark as complete
            progress['completed_players'].append(player_id)
            save_progress(progress)

            # Save incrementally every 10 players
            if len(progress['completed_players']) % 10 == 0:
                df_all = pd.DataFrame(all_game_logs)
                df_all.to_csv(OUTPUT_FILE, index=False)

        # Rate limit: NBA API allows ~60 requests per minute
        time.sleep(1.0)

    except Exception as e:
        tqdm.write(f"❌ Error for {player_name}: {str(e)[:100]}")
        failed_players.append((player_id, player_name, str(e)))
        time.sleep(2)

# Final save
if all_game_logs:
    df_all = pd.DataFrame(all_game_logs)
    df_all.to_csv(OUTPUT_FILE, index=False)

    print(f"\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\n✅ Total game logs collected: {len(df_all):,}")
    print(f"✅ Unique players: {df_all['PLAYER_ID'].nunique()}")
    print(f"✅ Unique games: {df_all['GAME_ID'].nunique()}")
    print(f"✅ Date range: {df_all['GAME_DATE'].min()} to {df_all['GAME_DATE'].max()}")

    # Show sample
    print(f"\nSample game logs:")
    sample = df_all[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST', 'MIN']].head(10)
    print(sample.to_string())

    # Calculate PRA
    df_all['PRA'] = df_all['PTS'] + df_all['REB'] + df_all['AST']

    # Summary stats
    print(f"\nSeason Statistics:")
    print(f"  Average PRA: {df_all['PRA'].mean():.2f}")
    print(f"  Average PTS: {df_all['PTS'].mean():.2f}")
    print(f"  Average REB: {df_all['REB'].mean():.2f}")
    print(f"  Average AST: {df_all['AST'].mean():.2f}")
    print(f"  Average MIN: {df_all['MIN'].mean():.2f}")

    print(f"\n📁 Saved to: {OUTPUT_FILE}")

if failed_players:
    print(f"\n⚠️  {len(failed_players)} players failed:")
    for pid, name, error in failed_players[:10]:
        print(f"  - {name}: {error[:80]}")

print("="*80)
