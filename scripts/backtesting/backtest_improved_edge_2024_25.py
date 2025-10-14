"""
Improved Backtest for 2024-25 Using Probability-Based Edge Calculation

Uses the new edge_calculator.py to properly account for:
- Prediction uncertainty
- Market odds and vig
- Expected value (EV) instead of simple edge
- Confidence-based bet quality tiers
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import json
import sys
sys.path.append('utils')
from edge_calculator import (
    calculate_true_edge,
    calculate_prediction_std,
    calculate_edge_quality,
    calculate_kelly_fraction
)

print("="*80)
print("NBA PROPS MODEL - 2024-25 IMPROVED EDGE BACKTEST")
print("="*80)

# Load walk-forward predictions and odds
print("\n1. Loading data...")
preds_df = pd.read_csv('data/results/walkforward_predictions_2024-25.csv')
odds_df = pd.read_csv('data/historical_odds/2024-25/pra_odds.csv')

print(f"✅ Loaded {len(preds_df):,} walk-forward predictions")
print(f"✅ Loaded {len(odds_df):,} betting lines")

# Standardize dates and names
preds_df['game_date'] = pd.to_datetime(preds_df['GAME_DATE']).dt.date
odds_df['game_date'] = pd.to_datetime(odds_df['event_date']).dt.date
preds_df['player_lower'] = preds_df['PLAYER_NAME'].str.lower().str.strip()
odds_df['player_lower'] = odds_df['player_name'].str.lower().str.strip()

# STEP 1: DEDUPLICATE PREDICTIONS
print("\n2. Deduplicating predictions (ONE per player-date)...")
print(f"   Before: {len(preds_df):,} rows")

preds_dedup = preds_df.groupby(['PLAYER_NAME', 'game_date'], as_index=False).agg({
    'player_lower': 'first',
    'PRA': 'first',
    'predicted_PRA': 'median',
    'error': 'median',
    'abs_error': 'median',
    'PLAYER_ID': 'first'
})

print(f"   After: {len(preds_dedup):,} rows")

# STEP 2: LINE SHOPPING
print("\n3. Line shopping (best odds across bookmakers)...")
odds_best = odds_df.groupby(['player_lower', 'game_date'], as_index=False).agg({
    'line': 'first',
    'over_price': 'max',
    'under_price': 'max',
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

# STEP 4: Calculate prediction uncertainty for each player
print("\n5. Calculating prediction uncertainty using player history...")

# Build player error history
player_errors = preds_df.groupby('PLAYER_ID').agg({
    'error': lambda x: list(x),
    'abs_error': 'mean'
}).reset_index()
player_errors.columns = ['PLAYER_ID', 'error_history', 'player_mae']

# Global MAE as fallback
global_mae = preds_df['abs_error'].mean()
print(f"   Global MAE: {global_mae:.2f} points")

# Calculate prediction std for each bet
def get_prediction_std(row):
    """Estimate prediction uncertainty for this bet"""
    player_id = row.get('PLAYER_ID')

    if player_id and player_id in player_errors['PLAYER_ID'].values:
        player_row = player_errors[player_errors['PLAYER_ID'] == player_id].iloc[0]
        error_list = player_row['error_history']

        if len(error_list) >= 5:
            # Player-specific std
            player_std = np.std(error_list) if len(error_list) > 1 else global_mae * 1.25

            # Adjust for extreme predictions
            player_avg = row['predicted_PRA']
            deviation_from_avg = abs(row['predicted_PRA'] - player_avg)
            uncertainty_factor = 1 + (deviation_from_avg / 20)

            return player_std * uncertainty_factor

    # Fallback to global uncertainty
    return global_mae * 1.25

merged['prediction_std'] = merged.apply(get_prediction_std, axis=1)
print(f"   Average prediction std: {merged['prediction_std'].mean():.2f} points")

# STEP 5: Calculate TRUE EDGE using probability-based EV
print("\n6. Calculating probability-based expected value...")

edges = []
bet_sides = []
details_list = []

for idx, row in merged.iterrows():
    edge, bet_side, details = calculate_true_edge(
        predicted_pra=row['predicted_PRA'],
        prediction_std=row['prediction_std'],
        betting_line=row['line'],
        over_odds=row['over_price'],
        under_odds=row['under_price'],
        min_edge_threshold=0.05  # 5% minimum EV
    )

    edges.append(edge)
    bet_sides.append(bet_side)
    details_list.append(details)

merged['ev'] = edges
merged['bet_side'] = bet_sides

# Extract details
merged['our_over_prob'] = [d['our_over_prob'] for d in details_list]
merged['our_under_prob'] = [d['our_under_prob'] for d in details_list]
merged['confidence_score'] = [d['confidence_score'] for d in details_list]

print(f"   Bets recommended: {(merged['bet_side'] != 'SKIP').sum():,}")
print(f"   OVER bets: {(merged['bet_side'] == 'OVER').sum():,}")
print(f"   UNDER bets: {(merged['bet_side'] == 'UNDER').sum():,}")
print(f"   SKIP: {(merged['bet_side'] == 'SKIP').sum():,}")

# STEP 6: Calculate bet quality tiers
merged['bet_quality'] = merged.apply(
    lambda row: calculate_edge_quality(row['ev'], row['prediction_std']),
    axis=1
)

# STEP 7: Calculate bet results
def calculate_bet_result(row):
    if row['bet_side'] == 'SKIP':
        return 0

    actual = row['PRA']
    line = row['line']

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
merged['bet_won'] = (merged['bet_result'] > 0) & (merged['bet_side'] != 'SKIP')
merged['bet_pushed'] = (merged['bet_result'] == 0) & (merged['bet_side'] != 'SKIP')
merged['bet_lost'] = (merged['bet_result'] < 0) & (merged['bet_side'] != 'SKIP')

# Overall statistics
total_bets = (merged['bet_side'] != 'SKIP').sum()
won_bets = merged['bet_won'].sum()
pushed_bets = merged['bet_pushed'].sum()
lost_bets = merged['bet_lost'].sum()
total_profit = merged[merged['bet_side'] != 'SKIP']['bet_result'].sum()
total_wagered = total_bets * 100

print("\n" + "="*80)
print("2024-25 IMPROVED EDGE BACKTEST RESULTS")
print("="*80)

print(f"\n📊 BETTING RESULTS (Probability-based EV ≥ 5%):")
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

    # Results by bet quality
    print(f"\n" + "="*80)
    print("PERFORMANCE BY BET QUALITY")
    print("="*80)

    quality_tiers = ['ELITE', 'EXCELLENT', 'GOOD', 'MARGINAL']

    for quality in quality_tiers:
        bucket = merged[merged['bet_quality'] == quality]
        if len(bucket) == 0:
            continue

        bucket_bets = (bucket['bet_side'] != 'SKIP').sum()
        if bucket_bets == 0:
            continue

        bucket_won = bucket['bet_won'].sum()
        bucket_profit = bucket[bucket['bet_side'] != 'SKIP']['bet_result'].sum()
        bucket_wagered = bucket_bets * 100

        print(f"\n{quality}:")
        print(f"  Bets: {bucket_bets:,}")
        print(f"  Win rate: {bucket_won/bucket_bets*100:.1f}%")
        print(f"  ROI: {bucket_profit/bucket_wagered*100:+.2f}%")
        print(f"  Profit: ${bucket_profit:,.2f}")
        print(f"  Avg EV: {bucket['ev'].mean()*100:.2f}%")
        print(f"  Avg Confidence: {bucket['confidence_score'].mean():.3f}")

# Prediction accuracy
print(f"\n" + "="*80)
print("PREDICTION ACCURACY")
print("="*80)

mae = mean_absolute_error(merged['PRA'], merged['predicted_PRA'])
print(f"MAE: {mae:.2f} points")
print(f"Within ±3 pts: {(abs(merged['PRA'] - merged['predicted_PRA']) <= 3).mean()*100:.1f}%")
print(f"Within ±5 pts: {(abs(merged['PRA'] - merged['predicted_PRA']) <= 5).mean()*100:.1f}%")

# Save results
output_file = "data/results/backtest_improved_edge_2024_25.csv"
merged.to_csv(output_file, index=False)
print(f"\n✅ Saved detailed results to {output_file}")

# Summary
if total_bets > 0:
    summary = {
        'season': '2024-25',
        'approach': 'improved_edge_calculation',
        'edge_method': 'probability_based_ev',
        'min_ev_threshold': 0.05,
        'total_matched_predictions': int(len(merged)),
        'total_bets': int(total_bets),
        'win_rate': float(won_bets/total_bets*100),
        'roi_percent': float(total_profit/total_wagered*100),
        'total_profit': float(total_profit),
        'total_wagered': float(total_wagered),
        'mae': float(mae),
        'average_ev': float(merged[merged['bet_side'] != 'SKIP']['ev'].mean()),
        'average_confidence': float(merged[merged['bet_side'] != 'SKIP']['confidence_score'].mean())
    }

    with open("data/results/backtest_improved_edge_2024_25_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "="*80)
    print("COMPARISON: OLD EDGE vs NEW PROBABILITY-BASED EDGE")
    print("="*80)

    print(f"""
OLD Edge Calculation (Simple):
  Method: edge = predicted_PRA - betting_line
  Bets: 2,495 (edge ≥ ±3 pts)
  Win Rate: 51.98%
  ROI: +0.91%
  Profit: $2,259

NEW Edge Calculation (Probability-Based):
  Method: EV = (our_prob × payout) - 1
  Accounts for: Uncertainty, odds, vig
  Threshold: EV ≥ 5%

  Bets: {total_bets:,}
  Win Rate: {won_bets/total_bets*100:.2f}%
  ROI: {total_profit/total_wagered*100:+.2f}%
  Profit: ${total_profit:,.2f}

{'✅ IMPROVEMENT!' if total_profit > 2259 else '⚠️ NEEDS CALIBRATION' if total_bets > 0 else '❌ NO BETS'}

Industry Benchmarks:
  Good:  54-56% win rate, 3-5% ROI
  Elite: 58-60% win rate, 8-12% ROI

Your Model: {won_bets/total_bets*100:.1f}% win rate, {total_profit/total_wagered*100:.1f}% ROI
Status: {'✅ CRUSHING ELITE' if total_profit/total_wagered*100 > 12 else '✅ EXCEEDING GOOD' if total_profit/total_wagered*100 > 5 else '⚠️ BELOW GOOD' if total_profit > 0 else '❌ UNPROFITABLE'}
""")

print("="*80)
