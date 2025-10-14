"""
Preprocess 2024-25 game logs to add required columns for game_log_builder.
"""

import pandas as pd

print("="*80)
print("PREPROCESSING 2024-25 GAME LOGS")
print("="*80)

# Load game logs
print("\n1. Loading game logs...")
df = pd.read_csv('data/game_logs/game_logs_2024_25.csv')
print(f"✅ Loaded {len(df):,} game logs")

# Add SEASON column from SEASON_ID
print("\n2. Adding SEASON column...")
# SEASON_ID format: "22024" = 2024-25 season
df['SEASON'] = df['SEASON_ID'].astype(str).str[1:] + '-' + (df['SEASON_ID'].astype(str).str[1:].astype(int) + 1).astype(str).str[-2:]
print(f"✅ Added SEASON column: {df['SEASON'].unique()}")

# Add SEASON_TYPE column (Regular Season for 2024-25)
print("\n3. Adding SEASON_TYPE column...")
df['SEASON_TYPE'] = 'Regular Season'
print(f"✅ Added SEASON_TYPE: {df['SEASON_TYPE'].unique()}")

# Standardize column names
print("\n4. Standardizing column names...")
rename_map = {
    'Player_ID': 'PLAYER_ID',
    'Game_ID': 'GAME_ID'
}
df = df.rename(columns=rename_map)
print(f"✅ Standardized: {list(rename_map.keys())}")

# Save preprocessed file
output_file = 'data/game_logs/game_logs_2024_25_preprocessed.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved preprocessed game logs to {output_file}")

print(f"\nFinal columns: {len(df.columns)}")
print(f"Required columns present:")
print(f"  ✅ PLAYER_ID: {('PLAYER_ID' in df.columns)}")
print(f"  ✅ PLAYER_NAME: {('PLAYER_NAME' in df.columns)}")
print(f"  ✅ GAME_ID: {('GAME_ID' in df.columns)}")
print(f"  ✅ GAME_DATE: {('GAME_DATE' in df.columns)}")
print(f"  ✅ SEASON: {('SEASON' in df.columns)}")
print(f"  ✅ SEASON_TYPE: {('SEASON_TYPE' in df.columns)}")
print(f"  ✅ PRA: {('PRA' in df.columns)}")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE")
print("="*80)
