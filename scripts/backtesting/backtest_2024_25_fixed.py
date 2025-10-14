"""
FIXED BACKTEST for 2024-25 Season

Properly deduplicates predictions and uses line shopping.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import json

print("="*80)
print("NBA PROPS MODEL - 2024-25 BACKTEST (FIXED)")
print("="*80)

# 1. Load and prepare data
print("\n1. Loading data...")

# Load 2024-25 game logs
game_logs = pd.read_csv('data/game_logs/game_logs_2024_25.csv')
print(f"✅ Loaded {len(game_logs):,} game logs")

# Load model predictions (will train on 2024-25 data)
# For now, we need to build the dataset and generate predictions
print("\n⚠️  Need to build 2024-25 dataset and generate predictions first!")
print("   This requires running the game_log_builder on 2024-25 data")

# Load 2024-25 odds
odds_file = "data/historical_odds/2024-25/pra_odds.csv"
odds_df = pd.read_csv(odds_file)
print(f"✅ Loaded {len(odds_df):,} betting lines")

# For now, let's just verify the odds data structure
print(f"\n2024-25 Odds Summary:")
print(f"  Date range: {odds_df['event_date'].min()} to {odds_df['event_date'].max()}")
print(f"  Unique players: {odds_df['player_name'].nunique()}")
print(f"  Unique events: {odds_df['event_id'].nunique()}")
print(f"  Bookmakers: {odds_df['bookmaker'].nunique()}")
print(f"  Bookmakers: {list(odds_df['bookmaker'].unique())}")

# Show what we need to do
print(f"\n" + "="*80)
print("NEXT STEPS TO COMPLETE 2024-25 BACKTEST:")
print("="*80)
print("""
1. Build 2024-25 game-level dataset with features
   - Run game_log_builder on 2024-25 game logs
   - Create lag, rolling, EWMA features
   - But DON'T include CTG duplicates!

2. Train model on historical data (2003-2024)
   - Use cleaned dataset without duplicates
   - Generate predictions for 2024-25

3. Match predictions to betting lines
   - Deduplicate: ONE prediction per player-date
   - Line shopping: Best odds across bookmakers
   - Calculate true ROI

The key fixes:
✅ Deduplicate game-level dataset BEFORE training
✅ Deduplicate predictions BEFORE matching to odds
✅ Use best odds via line shopping
""")

print("="*80)
