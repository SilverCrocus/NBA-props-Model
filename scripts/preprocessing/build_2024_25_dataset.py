"""
Build Clean 2024-25 Dataset with Fixed CTG Merge

Uses the corrected game_log_builder.py to create a dataset without duplicates.
"""

import sys
sys.path.append('src')

from data.game_log_builder import GameLogDatasetBuilder
import pandas as pd
from pathlib import Path

print("="*80)
print("BUILDING CLEAN 2024-25 DATASET (CTG BUG FIXED)")
print("="*80)

# Load 2024-25 game logs first to verify
print("\n1. Loading 2024-25 game logs...")
game_logs = pd.read_csv('data/game_logs/game_logs_2024_25_preprocessed.csv')
print(f"✅ Loaded {len(game_logs):,} game logs")
print(f"   Players: {game_logs['PLAYER_NAME'].nunique()}")
print(f"   Games: {game_logs['GAME_ID'].nunique()}")
print(f"   Date range: {game_logs['GAME_DATE'].min()} to {game_logs['GAME_DATE'].max()}")
print(f"   Season: {game_logs['SEASON'].unique()}")

# Initialize builder with 2024-25 game logs path
print("\n2. Initializing GameLogDatasetBuilder with 2024-25 data...")
builder = GameLogDatasetBuilder(
    game_logs_path='data/game_logs/game_logs_2024_25_preprocessed.csv',
    ctg_data_dir='data/ctg_data_organized/players'
)

# Build complete dataset
print("\n3. Building complete dataset with features...")
print("   This will:")
print("   ✅ Merge CTG stats (WITH DEDUPLICATION)")
print("   ✅ Create lag features (1, 2, 3 games)")
print("   ✅ Create rolling averages (3, 5, 10 games)")
print("   ✅ Create EWMA features (3, 5, 10 games)")
print("   ✅ Remove duplicate player-game combinations")

# Build the dataset
dataset = builder.build_complete_dataset(
    merge_ctg=True,
    min_minutes_per_game=10.0,
    min_games_played=5
)

print(f"\n✅ Dataset built successfully!")
print(f"   Total rows: {len(dataset):,}")
print(f"   Total columns: {len(dataset.columns)}")

# Check for duplicates
print("\n4. Verifying no duplicates...")
duplicates = dataset.groupby(['PLAYER_ID', 'GAME_DATE']).size()
duplicates = duplicates[duplicates > 1]

if len(duplicates) > 0:
    print(f"⚠️  WARNING: Found {len(duplicates)} duplicate player-game combinations!")
    print(f"\nExample duplicates:")
    print(duplicates.head(10))
else:
    print("✅ No duplicates found! Clean dataset confirmed.")

# Check for CTG columns
ctg_cols = [col for col in dataset.columns if any(x in col for x in ['USG%', 'TS%', 'AST%', 'TOV%', 'ORB%', 'DRB%'])]
print(f"\n5. CTG columns found: {len(ctg_cols)}")
if ctg_cols:
    print(f"   Sample CTG columns: {ctg_cols[:5]}")

# Split into train/val
print("\n6. Splitting into train/val sets...")

# Use date-based split: 80/20
dataset['GAME_DATE'] = pd.to_datetime(dataset['GAME_DATE'])
dataset = dataset.sort_values('GAME_DATE')

split_idx = int(len(dataset) * 0.8)
train = dataset.iloc[:split_idx]
val = dataset.iloc[split_idx:]

print(f"   Train: {len(train):,} rows ({train['GAME_DATE'].min()} to {train['GAME_DATE'].max()})")
print(f"   Val:   {len(val):,} rows ({val['GAME_DATE'].min()} to {val['GAME_DATE'].max()})")

# Save datasets
print("\n7. Saving datasets...")
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)

train_file = output_dir / 'train_2024_25.parquet'
val_file = output_dir / 'val_2024_25.parquet'
full_file = output_dir / 'full_2024_25.parquet'

train.to_parquet(train_file, index=False)
val.to_parquet(val_file, index=False)
dataset.to_parquet(full_file, index=False)

print(f"✅ Saved train to {train_file}")
print(f"✅ Saved val to {val_file}")
print(f"✅ Saved full dataset to {full_file}")

# Summary statistics
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)

print(f"\nBasic Stats:")
print(f"  Total player-games: {len(dataset):,}")
print(f"  Unique players: {dataset['PLAYER_NAME'].nunique()}")
game_id_col = 'Game_ID' if 'Game_ID' in dataset.columns else 'GAME_ID'
print(f"  Unique games: {dataset[game_id_col].nunique()}")
print(f"  Date range: {dataset['GAME_DATE'].min()} to {dataset['GAME_DATE'].max()}")

print(f"\nFeature Columns:")
print(f"  Total columns: {len(dataset.columns)}")
print(f"  CTG columns: {len(ctg_cols)}")

# Check for missing values in key columns
key_cols = ['PRA', 'MIN', 'PLUS_MINUS'] + ctg_cols[:5]
print(f"\nMissing Values in Key Columns:")
for col in key_cols:
    if col in dataset.columns:
        missing = dataset[col].isna().sum()
        missing_pct = missing / len(dataset) * 100
        print(f"  {col}: {missing:,} ({missing_pct:.1f}%)")

print("\n" + "="*80)
print("✅ DATASET BUILD COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Train model on 2003-2024 historical data")
print("  2. Generate predictions for 2024-25 dataset")
print("  3. Match predictions to betting odds")
print("  4. Run corrected backtest to validate performance")
print("="*80)
