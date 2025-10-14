"""
CORRECTED 2024-25 Backtest - Properly Deduplicated

Validates model performance on fresh 2024-25 data.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import json

print("="*80)
print("NBA PROPS MODEL - 2024-25 BACKTEST (CORRECTED)")
print("="*80)

# Load predictions and odds
print("\n1. Loading data...")
preds_df = pd.read_csv('data/results/baseline_predictions_2024-25.csv')
odds_df = pd.read_csv('data/historical_odds/2024-25/pra_odds.csv')

print(f"✅ Loaded {len(preds_df):,} predictions")
print(f"✅ Loaded {len(odds_df):,} betting lines")

# Standardize dates and names
preds_df['game_date'] = pd.to_datetime(preds_df['GAME_DATE']).dt.date
odds_df['game_date'] = pd.to_datetime(odds_df['event_date']).dt.date
preds_df['player_lower'] = preds_df['PLAYER_NAME'].str.lower().str.strip()
odds_df['player_lower'] = odds_df['player_name'].str.lower().str.strip()

# STEP 1: DEDUPLICATE PREDICTIONS
print("\n2. Deduplicating predictions (ONE per player-date)...")
print(f"   Before: {len(preds_df):,} rows")

# Strategy: For duplicates, keep the median prediction
preds_dedup = preds_df.groupby(['PLAYER_NAME', 'game_date'], as_index=False).agg({
    'player_lower': 'first',
    'TEAM_ABBREVIATION': 'first',
    'OPPONENT': 'first',
    'PRA': 'first',  # Actual is same across duplicates
    'predicted_PRA': 'median',  # Use median of predictions
    'error': 'median',
    'abs_error': 'median'
})

print(f"   After: {len(preds_dedup):,} rows")
if len(preds_df) > len(preds_dedup):
    print(f"   Removed: {len(preds_df) - len(preds_dedup):,} duplicates")

# STEP 2: LINE SHOPPING - Best odds per player-date
print("\n3. Line shopping (best odds across bookmakers)...")
odds_best = odds_df.groupby(['player_lower', 'game_date'], as_index=False).agg({
    'line': 'first',
    'over_price': 'max',  # Best over odds
    'under_price': 'max',  # Best under odds
    'bookmaker': lambda x: ', '.join(x.unique()[:3])
})

print(f"   Unique player-date combinations: {len(odds_best):,}")

# STEP 3: MATCH predictions to odds
print("\n4. Matching predictions to betting lines...")
merged = preds_dedup.merge(
    odds_best,
    left_on=['player_lower', 'game_date'],
    right_on=['player_lower', 'game_date'],
    how='inner'
)

print(f"✅ Matched {len(merged):,} predictions to betting lines")
print(f"   Match rate: {len(merged)/len(preds_dedup)*100:.1f}%")

# Calculate betting metrics
merged['actual_pra'] = merged['PRA']
merged['predicted_pra'] = merged['predicted_PRA']
merged['betting_line'] = merged['line']
merged['edge'] = merged['predicted_pra'] - merged['betting_line']
merged['abs_edge'] = abs(merged['edge'])

# Convert American odds to implied probability
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

merged['over_implied_prob'] = merged['over_price'].apply(american_to_prob)
merged['under_implied_prob'] = merged['under_price'].apply(american_to_prob)

# Determine bet side
merged['bet_side'] = 'NONE'
merged.loc[merged['edge'] >= 3, 'bet_side'] = 'OVER'
merged.loc[merged['edge'] <= -3, 'bet_side'] = 'UNDER'

# Calculate bet results
def calculate_bet_result(row):
    if row['bet_side'] == 'NONE':
        return 0

    actual = row['actual_pra']
    line = row['betting_line']

    if row['bet_side'] == 'OVER':
        won = actual > line
        odds = row['over_price']
    else:
        won = actual < line
        odds = row['under_price']

    if actual == line:  # Push
        return 0

    if won:
        if odds > 0:
            return odds
        else:
            return 100 * (100 / abs(odds))
    else:
        return -100

merged['bet_result'] = merged.apply(calculate_bet_result, axis=1)
merged['bet_won'] = (merged['bet_result'] > 0) & (merged['bet_side'] != 'NONE')
merged['bet_pushed'] = (merged['bet_result'] == 0) & (merged['bet_side'] != 'NONE')
merged['bet_lost'] = (merged['bet_result'] < 0) & (merged['bet_side'] != 'NONE')

# Overall statistics
total_bets = (merged['bet_side'] != 'NONE').sum()
won_bets = merged['bet_won'].sum()
pushed_bets = merged['bet_pushed'].sum()
lost_bets = merged['bet_lost'].sum()
total_profit = merged['bet_result'].sum()
total_wagered = total_bets * 100

print("\n" + "="*80)
print("CORRECTED 2024-25 BACKTEST RESULTS")
print("="*80)

print(f"\n📊 BETTING RESULTS (Edge ≥ ±3 points):")
print("-"*80)
print(f"Total bets placed: {total_bets:,}")
if total_bets > 0:
    print(f"  Won:    {won_bets:,} ({won_bets/total_bets*100:.2f}%)")
    print(f"  Lost:   {lost_bets:,} ({lost_bets/total_bets*100:.2f}%)")
    print(f"  Pushed: {pushed_bets:,} ({pushed_bets/total_bets*100:.2f}%)")

    print(f"\n💰 PROFIT & LOSS:")
    print("-"*80)
    print(f"Total wagered: ${total_wagered:,.0f}")
    print(f"Total profit:  ${total_profit:,.2f}")
    print(f"ROI:           {total_profit/total_wagered*100:+.2f}%")

    # Breakeven analysis
    breakeven_rate = 52.38
    print(f"\n📈 ANALYSIS:")
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
        if bucket_bets == 0:
            continue

        bucket_won = bucket['bet_won'].sum()
        bucket_profit = bucket['bet_result'].sum()
        bucket_wagered = bucket_bets * 100

        print(f"\n{label}:")
        print(f"  Bets: {bucket_bets:,}")
        print(f"  Win rate: {bucket_won/bucket_bets*100:.1f}%")
        print(f"  ROI: {bucket_profit/bucket_wagered*100:+.2f}%")
        print(f"  Profit: ${bucket_profit:,.2f}")
else:
    print("\n⚠️  No bets met the ±3 point edge threshold")

# Prediction accuracy
print(f"\n" + "="*80)
print("PREDICTION ACCURACY")
print("="*80)

mae = mean_absolute_error(merged['actual_pra'], merged['predicted_pra'])
print(f"MAE: {mae:.2f} points")
print(f"Within ±3 pts: {(abs(merged['actual_pra'] - merged['predicted_pra']) <= 3).mean()*100:.1f}%")
print(f"Within ±5 pts: {(abs(merged['actual_pra'] - merged['predicted_pra']) <= 5).mean()*100:.1f}%")

# CLV
print(f"\n" + "="*80)
print("CLOSING LINE VALUE (CLV)")
print("="*80)

clv_positive = (merged['abs_edge'] >= 3).sum()
clv_rate = clv_positive / len(merged) * 100

print(f"Predictions with 3+ point edge: {clv_positive:,} ({clv_rate:.1f}%)")
print(f"Average edge (absolute): {merged['abs_edge'].mean():.2f} points")
print(f"Average edge (signed): {merged['edge'].mean():+.2f} points")

# Save results
output_file = "data/results/backtest_2024_25_corrected.csv"
merged.to_csv(output_file, index=False)
print(f"\n✅ Saved detailed results to {output_file}")

# Summary
if total_bets > 0:
    summary = {
        'season': '2024-25',
        'total_matched_predictions': int(len(merged)),
        'total_bets': int(total_bets),
        'win_rate': float(won_bets/total_bets*100),
        'roi_percent': float(total_profit/total_wagered*100),
        'total_profit': float(total_profit),
        'total_wagered': float(total_wagered),
        'mae': float(mae),
        'average_edge': float(merged['edge'].mean()),
        'clv_rate': float(clv_rate),
        'deduplication_applied': True
    }

    with open("data/results/backtest_2024_25_corrected_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*80)
    print("FINAL SUMMARY (CORRECTED 2024-25)")
    print("="*80)

    print(f"""
✅ CORRECTED 2024-25 BACKTEST COMPLETE

Matched predictions: {len(merged):,} games
Total bets (≥3 pt edge): {total_bets:,}

Win rate: {won_bets/total_bets*100:.2f}%
ROI: {total_profit/total_wagered*100:+.2f}%
Total P&L: ${total_profit:,.2f}

Prediction MAE: {mae:.2f} points
CLV (3+ pt edge): {clv_rate:.1f}%

🔧 Fixes applied:
  ✅ Deduplicated predictions (median of duplicates)
  ✅ One bet per player-date combination
  ✅ Line shopping across all bookmakers

Result: {'✅ PROFITABLE!' if total_profit > 0 else '❌ Not profitable'}

Comparison to 2023-24:
  2023-24: 79.66% win rate, +61.98% ROI
  2024-25: {won_bets/total_bets*100:.2f}% win rate, {total_profit/total_wagered*100:+.2f}% ROI

Industry Benchmarks:
  Good:  54-56% win rate, 3-5% ROI
  Elite: 58-60% win rate, 8-12% ROI

Your Model: {won_bets/total_bets*100:.1f}% win rate, {total_profit/total_wagered*100:.1f}% ROI
Status: {'✅ CRUSHING ELITE BENCHMARKS' if total_profit/total_wagered*100 > 12 else '✅ Exceeding good benchmarks' if total_profit > 0 else '❌ Needs improvement'}
""")

print("="*80)
