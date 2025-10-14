"""
REAL BACKTEST with Historical Betting Lines

Compare model predictions to actual PRA betting lines from 2023-24 season
and calculate TRUE ROI based on real odds.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("NBA PROPS MODEL - REAL BACKTEST WITH HISTORICAL ODDS")
print("="*80)

# Load model predictions from previous backtest
print("\n1. Loading model predictions...")
predictions_file = "data/results/baseline_predictions_2023-24.csv"
preds_df = pd.read_csv(predictions_file)
print(f"✅ Loaded {len(preds_df):,} predictions")
print(f"   Date range: {preds_df['GAME_DATE'].min()} to {preds_df['GAME_DATE'].max()}")

# Load historical odds
print("\n2. Loading historical betting lines...")
odds_file = "data/historical_odds/2023-24/pra_odds.csv"
odds_df = pd.read_csv(odds_file)
print(f"✅ Loaded {len(odds_df):,} prop lines")
print(f"   Date range: {odds_df['event_date'].min()} to {odds_df['event_date'].max()}")
print(f"   Bookmakers: {odds_df['bookmaker'].unique()}")

# Standardize dates
preds_df['game_date'] = pd.to_datetime(preds_df['GAME_DATE']).dt.date
odds_df['game_date'] = pd.to_datetime(odds_df['event_date']).dt.date

# Standardize player names (case-insensitive matching)
preds_df['player_lower'] = preds_df['PLAYER_NAME'].str.lower().str.strip()
odds_df['player_lower'] = odds_df['player_name'].str.lower().str.strip()

print("\n3. Matching predictions to betting lines...")

# Strategy: Use ALL bookmakers, take best available line (line shopping!)
print(f"   Available bookmakers:")
for bookie, count in odds_df['bookmaker'].value_counts().items():
    print(f"     {bookie}: {count:,} lines")

# For each prediction, find the best line across all bookmakers
# Group by player + date to get all available lines
odds_grouped = odds_df.groupby(['player_lower', 'game_date']).agg({
    'line': 'first',  # The line (should be same across bookies)
    'over_price': 'max',  # Best over price (highest)
    'under_price': 'max',  # Best under price (highest)
    'bookmaker': lambda x: ', '.join(x.unique()[:3])  # Which bookies had it
}).reset_index()

print(f"\n   Unique player-date combinations: {len(odds_grouped):,}")

# Match on player + date
merged = preds_df.merge(
    odds_grouped,
    left_on=['player_lower', 'game_date'],
    right_on=['player_lower', 'game_date'],
    how='inner',
    suffixes=('_pred', '_odds')
)

print(f"✅ Matched {len(merged):,} predictions to betting lines")
print(f"   Match rate: {len(merged)/len(preds_df)*100:.1f}%")

if len(merged) == 0:
    print("\n❌ No matches found! Check date/name formats")
    print("\nSample predictions:")
    print(preds_df[['PLAYER_NAME', 'game_date', 'predicted_PRA']].head())
    print("\nSample odds:")
    print(dk_odds[['player_name', 'game_date', 'line']].head())
    exit(1)

# Calculate betting metrics
print("\n" + "="*80)
print("4. CALCULATING REAL BETTING PERFORMANCE")
print("="*80)

merged['actual_pra'] = merged['PRA']
merged['predicted_pra'] = merged['predicted_PRA']
merged['betting_line'] = merged['line']
merged['over_odds'] = merged['over_price']
merged['under_odds'] = merged['under_price']

# Edge calculation
merged['edge'] = merged['predicted_pra'] - merged['betting_line']
merged['abs_edge'] = abs(merged['edge'])

# Convert American odds to implied probability
def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

merged['over_implied_prob'] = merged['over_odds'].apply(american_to_prob)
merged['under_implied_prob'] = merged['under_odds'].apply(american_to_prob)

# Determine bet direction
merged['bet_side'] = 'NONE'
merged.loc[merged['edge'] >= 3, 'bet_side'] = 'OVER'
merged.loc[merged['edge'] <= -3, 'bet_side'] = 'UNDER'

# Calculate bet results
def calculate_bet_result(row):
    """Calculate profit/loss for bet."""
    if row['bet_side'] == 'NONE':
        return 0

    actual = row['actual_pra']
    line = row['betting_line']

    if row['bet_side'] == 'OVER':
        won = actual > line
        odds = row['over_odds']
    else:  # UNDER
        won = actual < line
        odds = row['under_odds']

    # Push (tie) - no win/loss
    if actual == line:
        return 0

    # Calculate profit/loss (assuming $100 bet)
    if won:
        if odds > 0:
            return odds  # Win $odds on $100 bet
        else:
            return 100 * (100 / abs(odds))  # Win $100 / abs(odds) on $100 bet
    else:
        return -100  # Lost $100

merged['bet_result'] = merged.apply(calculate_bet_result, axis=1)

# Calculate win/push/loss ONLY for actual bets (not NONE)
merged['bet_won'] = (merged['bet_result'] > 0) & (merged['bet_side'] != 'NONE')
merged['bet_pushed'] = (merged['bet_result'] == 0) & (merged['bet_side'] != 'NONE')
merged['bet_lost'] = (merged['bet_result'] < 0) & (merged['bet_side'] != 'NONE')

# Overall statistics
total_bets = (merged['bet_side'] != 'NONE').sum()
won_bets = merged['bet_won'].sum()
pushed_bets = merged['bet_pushed'].sum()
lost_bets = merged['bet_lost'].sum()

total_profit = merged['bet_result'].sum()
total_wagered = total_bets * 100  # $100 per bet

print(f"\n📊 BETTING RESULTS (Edge ≥ ±3 points)")
print("-"*80)
print(f"Total bets placed: {total_bets:,}")
print(f"  Won:    {won_bets:,} ({won_bets/total_bets*100:.1f}%)")
print(f"  Lost:   {lost_bets:,} ({lost_bets/total_bets*100:.1f}%)")
print(f"  Pushed: {pushed_bets:,} ({pushed_bets/total_bets*100:.1f}%)")

print(f"\n💰 PROFIT & LOSS")
print("-"*80)
print(f"Total wagered: ${total_wagered:,.0f}")
print(f"Total profit:  ${total_profit:,.2f}")
print(f"ROI:           {total_profit/total_wagered*100:+.2f}%")

# Breakeven analysis
breakeven_rate = 52.38  # For -110 odds
print(f"\n📈 ANALYSIS")
print("-"*80)
print(f"Win rate needed (breakeven at -110): {breakeven_rate:.2f}%")
print(f"Actual win rate: {won_bets/total_bets*100:.2f}%")
print(f"Edge over breakeven: {won_bets/total_bets*100 - breakeven_rate:+.2f} percentage points")

if won_bets/total_bets*100 > breakeven_rate:
    print(f"✅ PROFITABLE MODEL!")
else:
    print(f"❌ Below breakeven")

# Results by edge size
print(f"\n" + "="*80)
print("PERFORMANCE BY EDGE SIZE")
print("="*80)

edge_buckets = [
    (3, 5, "Small edge (3-5 pts)"),
    (5, 7, "Medium edge (5-7 pts)"),
    (7, 10, "Large edge (7-10 pts)"),
    (10, 100, "Huge edge (10+ pts)")
]

for min_edge, max_edge, label in edge_buckets:
    bucket = merged[merged['abs_edge'].between(min_edge, max_edge)]
    if len(bucket) == 0:
        continue

    bucket_bets = (bucket['bet_side'] != 'NONE').sum()
    bucket_won = bucket['bet_won'].sum()
    bucket_profit = bucket['bet_result'].sum()
    bucket_wagered = bucket_bets * 100

    print(f"\n{label}:")
    print(f"  Bets: {bucket_bets:,}")
    print(f"  Win rate: {bucket_won/bucket_bets*100:.1f}%")
    print(f"  ROI: {bucket_profit/bucket_wagered*100:+.2f}%")

# Prediction accuracy
print(f"\n" + "="*80)
print("PREDICTION ACCURACY")
print("="*80)

mae = mean_absolute_error(merged['actual_pra'], merged['predicted_pra'])
print(f"MAE: {mae:.2f} points")
print(f"Within ±3 pts: {(abs(merged['actual_pra'] - merged['predicted_pra']) <= 3).mean()*100:.1f}%")
print(f"Within ±5 pts: {(abs(merged['actual_pra'] - merged['predicted_pra']) <= 5).mean()*100:.1f}%")

# Line accuracy (vs betting line, not actuals)
line_diff = abs(merged['predicted_pra'] - merged['betting_line'])
print(f"\nLine Accuracy (vs betting line):")
print(f"  Within ±3 pts of line: {(line_diff <= 3).mean()*100:.1f}%")
print(f"  Within ±5 pts of line: {(line_diff <= 5).mean()*100:.1f}%")

# Closing Line Value (CLV)
print(f"\n" + "="*80)
print("CLOSING LINE VALUE (CLV)")
print("="*80)

# CLV = How much better is our prediction than the betting line?
clv_positive = (merged['abs_edge'] >= 3).sum()
clv_rate = clv_positive / len(merged) * 100

print(f"Predictions with 3+ point edge: {clv_positive:,} ({clv_rate:.1f}%)")
print(f"Average edge (absolute): {merged['abs_edge'].mean():.2f} points")
print(f"Average edge (signed): {merged['edge'].mean():.2f} points")

if merged['edge'].mean() > 0:
    print(f"✅ Model slightly over-predicts vs market (bullish)")
else:
    print(f"ℹ️  Model slightly under-predicts vs market")

# Save detailed results
output_file = "data/results/backtest_with_real_odds.csv"
merged_output = merged[[
    'PLAYER_NAME', 'game_date', 'actual_pra', 'predicted_pra',
    'betting_line', 'edge', 'bet_side', 'bet_result', 'bet_won',
    'over_odds', 'under_odds', 'TEAM_ABBREVIATION', 'OPPONENT'
]]
merged_output.to_csv(output_file, index=False)
print(f"\n✅ Saved detailed results to {output_file}")

# Summary stats
summary = {
    'total_matched_predictions': int(len(merged)),
    'total_bets': int(total_bets),
    'win_rate': float(won_bets/total_bets*100),
    'roi_percent': float(total_profit/total_wagered*100),
    'total_profit': float(total_profit),
    'mae': float(mae),
    'average_edge': float(merged['edge'].mean()),
    'clv_rate': float(clv_rate)
}

import json
with open("data/results/backtest_real_odds_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
✅ REAL BACKTEST COMPLETE

Matched predictions: {len(merged):,} games
Total bets (≥3 pt edge): {total_bets:,}

Win rate: {won_bets/total_bets*100:.2f}%
ROI: {total_profit/total_wagered*100:+.2f}%
Total P&L: ${total_profit:,.2f}

Prediction MAE: {mae:.2f} points
CLV (3+ pt edge): {clv_rate:.1f}%

Result: {'✅ PROFITABLE!' if total_profit > 0 else '❌ Not profitable'}
""")

print("="*80)
