"""
Analyze Phase 1 Betting Simulation Results for Data Leakage
============================================================

This script identifies patterns suggesting data leakage or selection bias
in the 84% win rate from Phase 1 betting simulation.

Tasks:
1. Error distribution analysis - MAE on matched vs unmatched predictions
2. Temporal analysis - win rate degradation over time
3. Edge analysis - correlation between edge magnitude and win rate
4. Selection bias detection - matched vs unmatched characteristics
5. Red flag detection - suspicious patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("PHASE 1 BETTING SIMULATION - DATA LEAKAGE ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("\n[1] Loading data files...")

# Load betting simulation results
bets_file = '/Users/diyagamah/Documents/nba_props_model/data/results/phase1_betting_simulation_2024_25.csv'
ultra_file = '/Users/diyagamah/Documents/nba_props_model/data/results/phase1_ultra_selective_bets.csv'
preds_file = '/Users/diyagamah/Documents/nba_props_model/data/results/phase1_test_predictions_2024_25.csv'

bets = pd.read_csv(bets_file)
ultra_bets = pd.read_csv(ultra_file)
predictions = pd.read_csv(preds_file)

print(f"  - Betting simulation: {len(bets):,} bets")
print(f"  - Ultra-selective: {len(ultra_bets):,} bets")
print(f"  - All predictions: {len(predictions):,} predictions")

# Convert dates
for df in [bets, ultra_bets, predictions]:
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])

# =============================================================================
# 2. ERROR DISTRIBUTION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("[2] ERROR DISTRIBUTION ANALYSIS")
print("="*80)

# Calculate MAE on matched bets
bets['abs_error'] = bets['error'].abs()
mae_matched = bets['abs_error'].mean()

print(f"\nMAE on matched bets: {mae_matched:.2f} points")
print(f"Reported overall MAE: 4.19 points")
print(f"Difference: {mae_matched - 4.19:.2f} points")

# Create matched flag in predictions
predictions['matched_to_bet'] = predictions.apply(
    lambda row: len(bets[
        (bets['PLAYER_NAME'] == row['PLAYER_NAME']) &
        (bets['GAME_DATE'] == row['GAME_DATE'])
    ]) > 0,
    axis=1
)

print(f"\nPredictions matched to bets: {predictions['matched_to_bet'].sum():,} ({predictions['matched_to_bet'].mean()*100:.1f}%)")
print(f"Predictions NOT matched: {(~predictions['matched_to_bet']).sum():,} ({(~predictions['matched_to_bet']).mean()*100:.1f}%)")

# Calculate prediction error
predictions['predicted_pra'] = predictions['predicted_pra'].fillna(0)
predictions['actual_pra'] = predictions['actual_pra'].fillna(0)
predictions['error'] = predictions['predicted_pra'] - predictions['actual_pra']
predictions['abs_error'] = predictions['error'].abs()

# Compare MAE between matched and unmatched
mae_matched_preds = predictions[predictions['matched_to_bet']]['abs_error'].mean()
mae_unmatched_preds = predictions[~predictions['matched_to_bet']]['abs_error'].mean()

print(f"\nMAE on matched predictions: {mae_matched_preds:.2f} points")
print(f"MAE on unmatched predictions: {mae_unmatched_preds:.2f} points")
print(f"Difference: {mae_matched_preds - mae_unmatched_preds:.2f} points")

if mae_matched_preds < mae_unmatched_preds:
    print("⚠️  SELECTION BIAS DETECTED: Matched predictions have lower MAE")
else:
    print("✓ No selection bias in MAE (matched vs unmatched)")

# Distribution of errors
print("\n" + "-"*80)
print("Error Distribution (Matched Bets):")
print("-"*80)
print(bets['abs_error'].describe())

print("\n" + "-"*80)
print("Error Distribution (All Predictions):")
print("-"*80)
print(predictions['abs_error'].describe())

# =============================================================================
# 3. TEMPORAL ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("[3] TEMPORAL ANALYSIS")
print("="*80)

# Add month/week columns
bets['year_month'] = bets['GAME_DATE'].dt.to_period('M')
bets['week'] = bets['GAME_DATE'].dt.isocalendar().week
bets['month'] = bets['GAME_DATE'].dt.month

# Win rate by month
monthly_stats = bets.groupby('year_month').agg({
    'bet_won': ['sum', 'count', 'mean'],
    'abs_error': 'mean'
}).round(3)
monthly_stats.columns = ['wins', 'total_bets', 'win_rate', 'mae']

print("\nWin Rate by Month:")
print("-"*80)
print(monthly_stats)

# Check for degradation
first_half_wr = bets[bets['GAME_DATE'] < '2025-01-01']['bet_won'].mean()
second_half_wr = bets[bets['GAME_DATE'] >= '2025-01-01']['bet_won'].mean()

print(f"\nFirst half (Oct-Dec 2024) win rate: {first_half_wr*100:.1f}%")
print(f"Second half (Jan+ 2025) win rate: {second_half_wr*100:.1f}%")
print(f"Degradation: {(first_half_wr - second_half_wr)*100:.1f} percentage points")

if first_half_wr > second_half_wr + 0.05:
    print("⚠️  TEMPORAL LEAKAGE SUSPECTED: Win rate degrades over time")
else:
    print("✓ No significant temporal degradation")

# First 100 bets vs last 100 bets
if len(bets) >= 200:
    first_100_wr = bets.head(100)['bet_won'].mean()
    last_100_wr = bets.tail(100)['bet_won'].mean()
    print(f"\nFirst 100 bets win rate: {first_100_wr*100:.1f}%")
    print(f"Last 100 bets win rate: {last_100_wr*100:.1f}%")
    print(f"Difference: {(first_100_wr - last_100_wr)*100:.1f} percentage points")

# =============================================================================
# 4. EDGE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("[4] EDGE ANALYSIS")
print("="*80)

# Create edge magnitude bins
bets['edge_bin'] = pd.cut(
    bets['abs_edge'],
    bins=[0, 5, 6, 7, 8, 100],
    labels=['3-5pts', '5-6pts', '6-7pts', '7-8pts', '8+pts']
)

edge_analysis = bets.groupby('edge_bin', observed=True).agg({
    'bet_won': ['sum', 'count', 'mean'],
    'abs_error': 'mean'
}).round(3)
edge_analysis.columns = ['wins', 'total_bets', 'win_rate', 'mae']

print("\nWin Rate by Edge Magnitude:")
print("-"*80)
print(edge_analysis)

# Calculate correlation
edge_corr = bets[['abs_edge', 'bet_won']].corr().iloc[0, 1]
print(f"\nCorrelation between edge and win rate: {edge_corr:.3f}")

if edge_corr < 0.1:
    print("⚠️  RED FLAG: Large edges don't perform better (possible leakage)")
else:
    print("✓ Larger edges perform better (expected)")

# =============================================================================
# 5. SELECTION BIAS DETECTION
# =============================================================================

print("\n" + "="*80)
print("[5] SELECTION BIAS DETECTION")
print("="*80)

# Player distribution
matched_players = predictions[predictions['matched_to_bet']]['PLAYER_NAME'].value_counts()
unmatched_players = predictions[~predictions['matched_to_bet']]['PLAYER_NAME'].value_counts()

print(f"\nUnique players in matched predictions: {len(matched_players)}")
print(f"Unique players in unmatched predictions: {len(unmatched_players)}")

# Top 10 matched players
print("\nTop 10 Most Matched Players:")
print("-"*80)
print(matched_players.head(10))

# PRA range distribution
print("\n" + "-"*80)
print("Actual PRA Distribution:")
print("-"*80)
print("\nMatched predictions:")
print(predictions[predictions['matched_to_bet']]['actual_pra'].describe())
print("\nUnmatched predictions:")
print(predictions[~predictions['matched_to_bet']]['actual_pra'].describe())

# Check if matched predictions are for specific PRA ranges
matched_pra_mean = predictions[predictions['matched_to_bet']]['actual_pra'].mean()
unmatched_pra_mean = predictions[~predictions['matched_to_bet']]['actual_pra'].mean()

print(f"\nMatched PRA mean: {matched_pra_mean:.2f}")
print(f"Unmatched PRA mean: {unmatched_pra_mean:.2f}")
print(f"Difference: {matched_pra_mean - unmatched_pra_mean:.2f} points")

# =============================================================================
# 6. RED FLAG DETECTION
# =============================================================================

print("\n" + "="*80)
print("[6] RED FLAG DETECTION")
print("="*80)

# Perfect predictions (error < 1 point)
perfect_preds = bets[bets['abs_error'] < 1.0]
print(f"\nPerfect predictions (error < 1 pt): {len(perfect_preds)} ({len(perfect_preds)/len(bets)*100:.1f}%)")

if len(perfect_preds) > len(bets) * 0.05:
    print("⚠️  RED FLAG: Too many perfect predictions")
    print("\nSample of perfect predictions:")
    print(perfect_preds[['PLAYER_NAME', 'GAME_DATE', 'actual_pra', 'predicted_pra', 'error']].head(10))

# Exact matches (prediction = actual)
exact_matches = bets[bets['abs_error'] < 0.01]
print(f"\nExact matches (error < 0.01): {len(exact_matches)} ({len(exact_matches)/len(bets)*100:.1f}%)")

if len(exact_matches) > 5:
    print("⚠️  MAJOR RED FLAG: Exact predictions detected (possible leakage)")
    print("\nExact matches:")
    print(exact_matches[['PLAYER_NAME', 'GAME_DATE', 'actual_pra', 'predicted_pra', 'error']])

# Check bet outcome consistency
win_streak = 0
max_win_streak = 0
for won in bets.sort_values('GAME_DATE')['bet_won']:
    if won:
        win_streak += 1
        max_win_streak = max(max_win_streak, win_streak)
    else:
        win_streak = 0

print(f"\nLongest win streak: {max_win_streak} bets")

if max_win_streak > 15:
    print("⚠️  RED FLAG: Unrealistic win streak")
else:
    print("✓ Win streaks are reasonable")

# =============================================================================
# 7. SUMMARY
# =============================================================================

print("\n" + "="*80)
print("[7] SUMMARY OF FINDINGS")
print("="*80)

summary_flags = []

# MAE check
if mae_matched_preds < mae_unmatched_preds - 1.0:
    summary_flags.append("Selection bias: Matched predictions have significantly lower MAE")

# Temporal check
if first_half_wr > second_half_wr + 0.05:
    summary_flags.append("Temporal degradation: Win rate drops significantly over time")

# Edge check
if edge_corr < 0.1:
    summary_flags.append("Edge correlation: Large edges don't perform better")

# Perfect predictions
if len(perfect_preds) > len(bets) * 0.05:
    summary_flags.append(f"Too many perfect predictions: {len(perfect_preds)/len(bets)*100:.1f}%")

# Exact matches
if len(exact_matches) > 5:
    summary_flags.append(f"CRITICAL: {len(exact_matches)} exact matches detected")

print("\nRed Flags Detected:")
if summary_flags:
    for i, flag in enumerate(summary_flags, 1):
        print(f"  {i}. {flag}")
else:
    print("  None - results appear legitimate")

# Calculate expected win rate
print("\n" + "-"*80)
print("Expected vs Actual Performance:")
print("-"*80)
print(f"Actual win rate: {bets['bet_won'].mean()*100:.1f}%")
print(f"Expected win rate (with MAE={mae_matched:.1f}): ~52-58%")
print(f"Difference: {(bets['bet_won'].mean() - 0.55)*100:.1f} percentage points")

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

print("\n[8] Generating visualizations...")

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Win rate over time
monthly_stats_plot = monthly_stats.reset_index()
monthly_stats_plot['year_month_str'] = monthly_stats_plot['year_month'].astype(str)
axes[0, 0].plot(monthly_stats_plot['year_month_str'], monthly_stats_plot['win_rate']*100, marker='o', linewidth=2)
axes[0, 0].axhline(y=84, color='r', linestyle='--', label='Overall (84%)')
axes[0, 0].axhline(y=55, color='g', linestyle='--', label='Expected (55%)')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Win Rate (%)')
axes[0, 0].set_title('Win Rate Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Win rate by edge magnitude
edge_analysis_plot = edge_analysis.reset_index()
axes[0, 1].bar(edge_analysis_plot['edge_bin'].astype(str), edge_analysis_plot['win_rate']*100)
axes[0, 1].axhline(y=84, color='r', linestyle='--', label='Overall (84%)')
axes[0, 1].axhline(y=55, color='g', linestyle='--', label='Expected (55%)')
axes[0, 1].set_xlabel('Edge Magnitude')
axes[0, 1].set_ylabel('Win Rate (%)')
axes[0, 1].set_title('Win Rate by Edge Size')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Error distribution comparison
axes[1, 0].hist([
    predictions[predictions['matched_to_bet']]['abs_error'],
    predictions[~predictions['matched_to_bet']]['abs_error']
], bins=30, label=['Matched', 'Unmatched'], alpha=0.7)
axes[1, 0].set_xlabel('Absolute Error (points)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Error Distribution: Matched vs Unmatched')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Actual PRA distribution
axes[1, 1].hist([
    predictions[predictions['matched_to_bet']]['actual_pra'],
    predictions[~predictions['matched_to_bet']]['actual_pra']
], bins=30, label=['Matched', 'Unmatched'], alpha=0.7)
axes[1, 1].set_xlabel('Actual PRA')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Actual PRA Distribution: Matched vs Unmatched')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_file = '/Users/diyagamah/Documents/nba_props_model/data/results/phase1_leakage_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualizations: {output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
