"""
Comprehensive Exploratory Data Analysis for NBA Props Betting Model
Analyzes backtest results for 2024-25 season to identify optimal betting opportunities
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

# Load data
print("Loading backtest data...")
betting_df = pd.read_csv('data/results/backtest_2024_25_FIXED_V2_betting.csv')
full_df = pd.read_csv('data/results/backtest_2024_25_FIXED_V2.csv')

print(f"Betting file: {len(betting_df):,} bets")
print(f"Full predictions: {len(full_df):,} predictions")
print(f"\nColumns in betting file: {betting_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(betting_df.head())
print(f"\nData types:")
print(betting_df.dtypes)
print(f"\nBasic stats:")
print(betting_df.describe())

# Standardize column names - uppercase then remove duplicates
betting_df.columns = betting_df.columns.str.upper()
betting_df = betting_df.loc[:, ~betting_df.columns.duplicated()]

# Parse dates
if 'GAME_DATE' in betting_df.columns:
    betting_df['GAME_DATE'] = pd.to_datetime(betting_df['GAME_DATE'])
    betting_df['MONTH'] = betting_df['GAME_DATE'].dt.to_period('M')
    betting_df['WEEK'] = betting_df['GAME_DATE'].dt.to_period('W')

# Calculate error (use ERROR if exists, otherwise calculate)
if 'ERROR' not in betting_df.columns:
    betting_df['ERROR'] = betting_df['ACTUAL_PRA'] - betting_df['PREDICTED_PRA']
betting_df['ABS_ERROR'] = abs(betting_df['ERROR'])

print("\n" + "="*80)
print("1. OVERALL PERFORMANCE METRICS")
print("="*80)

total_bets = len(betting_df)
wins = betting_df['BET_CORRECT'].sum()
win_rate = wins / total_bets
mae = betting_df['ABS_ERROR'].mean()
rmse = np.sqrt((betting_df['ERROR']**2).mean())

# ROI calculation (assuming -110 odds)
# Win = +0.909 units, Loss = -1 unit
betting_df['PROFIT'] = betting_df['BET_CORRECT'].apply(lambda x: 0.909 if x else -1.0)
total_profit = betting_df['PROFIT'].sum()
roi = (total_profit / total_bets) * 100

# Binomial test for statistical significance
# Null hypothesis: win rate = 0.524 (breakeven at -110 odds)
p_value = stats.binomtest(wins, total_bets, 0.524, alternative='greater').pvalue

print(f"\nTotal Bets: {total_bets:,}")
print(f"Wins: {wins:,}")
print(f"Losses: {total_bets - wins:,}")
print(f"Win Rate: {win_rate:.2%}")
print(f"ROI: {roi:.2%}")
print(f"Total Profit: {total_profit:.2f} units")
print(f"\nPrediction Error:")
print(f"MAE: {mae:.2f} points")
print(f"RMSE: {rmse:.2f} points")
print(f"\nStatistical Significance:")
print(f"p-value (vs 52.4% breakeven): {p_value:.4f}")
print(f"Significantly better than breakeven: {'YES' if p_value < 0.05 else 'NO'}")

# Calculate confidence intervals
conf_level = 0.95
z_score = stats.norm.ppf((1 + conf_level) / 2)
se = np.sqrt(win_rate * (1 - win_rate) / total_bets)
ci_lower = win_rate - z_score * se
ci_upper = win_rate + z_score * se
print(f"95% Confidence Interval: [{ci_lower:.2%}, {ci_upper:.2%}]")

print("\n" + "="*80)
print("2. PERFORMANCE BY EDGE SIZE")
print("="*80)

# Create edge bins
betting_df['EDGE_BIN'] = pd.cut(
    betting_df['ABS_EDGE'],
    bins=[0, 3, 5, 7, 10, 15, 100],
    labels=['0-3 pts', '3-5 pts', '5-7 pts', '7-10 pts', '10-15 pts', '15+ pts']
)

edge_analysis = betting_df.groupby('EDGE_BIN', observed=True).agg({
    'BET_CORRECT': ['count', 'sum', 'mean'],
    'PROFIT': 'sum',
    'ABS_ERROR': 'mean',
    'ABS_EDGE': 'mean'
}).round(4)

edge_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit', 'MAE', 'Avg_Edge']
edge_analysis['ROI'] = (edge_analysis['Total_Profit'] / edge_analysis['Count']) * 100

print("\nPerformance by Edge Size:")
print(edge_analysis.to_string())

# Statistical significance by edge bin
print("\nStatistical Significance by Edge Bin:")
for edge_bin in betting_df['EDGE_BIN'].cat.categories:
    bin_data = betting_df[betting_df['EDGE_BIN'] == edge_bin]
    if len(bin_data) > 30:  # Only test if sufficient sample size
        bin_wins = bin_data['BET_CORRECT'].sum()
        bin_total = len(bin_data)
        bin_p_value = stats.binomtest(bin_wins, bin_total, 0.524, alternative='greater').pvalue
        sig = "***" if bin_p_value < 0.001 else "**" if bin_p_value < 0.01 else "*" if bin_p_value < 0.05 else ""
        print(f"{edge_bin}: p={bin_p_value:.4f} {sig}")

print("\n" + "="*80)
print("3. PERFORMANCE BY BET TYPE (OVER vs UNDER)")
print("="*80)

bet_type_analysis = betting_df.groupby('BET_TYPE').agg({
    'BET_CORRECT': ['count', 'sum', 'mean'],
    'PROFIT': 'sum',
    'ABS_ERROR': 'mean',
    'ABS_EDGE': 'mean'
}).round(4)

bet_type_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit', 'MAE', 'Avg_Edge']
bet_type_analysis['ROI'] = (bet_type_analysis['Total_Profit'] / bet_type_analysis['Count']) * 100

print("\nPerformance by Bet Type:")
print(bet_type_analysis.to_string())

# Chi-square test for independence
contingency = pd.crosstab(betting_df['BET_TYPE'], betting_df['BET_CORRECT'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square test for bet type independence:")
print(f"p-value: {p_value_chi:.4f}")
print(f"Significant directional bias: {'YES' if p_value_chi < 0.05 else 'NO'}")

print("\n" + "="*80)
print("4. PERFORMANCE BY PLAYER TYPE")
print("="*80)

# Calculate player-level stats
player_stats = betting_df.groupby('PLAYER_NAME').agg({
    'ACTUAL_PRA': 'mean',
    'BET_CORRECT': ['count', 'sum', 'mean'],
    'PROFIT': 'sum'
}).round(2)

player_stats.columns = ['Avg_PRA', 'Bets', 'Wins', 'Win_Rate', 'Profit']
player_stats = player_stats[player_stats['Bets'] >= 10]  # Min 10 bets
player_stats = player_stats.sort_values('Profit', ascending=False)

print(f"\nTotal players with 10+ bets: {len(player_stats)}")
print(f"\nTop 20 Most Profitable Players:")
print(player_stats.head(20).to_string())

print(f"\nBottom 20 Most Unprofitable Players:")
print(player_stats.tail(20).to_string())

# Player tier analysis (by average PRA)
betting_df['PLAYER_TIER'] = pd.cut(
    betting_df['ACTUAL_PRA'],
    bins=[0, 15, 25, 35, 50, 100],
    labels=['Role (<15)', 'Rotation (15-25)', 'Starter (25-35)', 'Star (35-50)', 'Superstar (50+)']
)

tier_analysis = betting_df.groupby('PLAYER_TIER', observed=True).agg({
    'BET_CORRECT': ['count', 'sum', 'mean'],
    'PROFIT': 'sum',
    'ABS_ERROR': 'mean'
}).round(4)

tier_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit', 'MAE']
tier_analysis['ROI'] = (tier_analysis['Total_Profit'] / tier_analysis['Count']) * 100

print("\nPerformance by Player Tier (based on average PRA):")
print(tier_analysis.to_string())

print("\n" + "="*80)
print("5. TEMPORAL ANALYSIS")
print("="*80)

if 'MONTH' in betting_df.columns:
    monthly_analysis = betting_df.groupby('MONTH').agg({
        'BET_CORRECT': ['count', 'sum', 'mean'],
        'PROFIT': 'sum',
        'ABS_ERROR': 'mean'
    }).round(4)

    monthly_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit', 'MAE']
    monthly_analysis['ROI'] = (monthly_analysis['Total_Profit'] / monthly_analysis['Count']) * 100

    print("\nPerformance by Month:")
    print(monthly_analysis.to_string())

    # Calculate correlation with time
    betting_df['DAYS_FROM_START'] = (betting_df['GAME_DATE'] - betting_df['GAME_DATE'].min()).dt.days
    time_correlation = betting_df['DAYS_FROM_START'].corr(betting_df['BET_CORRECT'].astype(int))
    print(f"\nCorrelation between time and win rate: {time_correlation:.4f}")
    print(f"Model decay over time: {'YES' if time_correlation < -0.05 else 'NO'}")

print("\n" + "="*80)
print("6. PERFORMANCE BY GAME CONTEXT")
print("="*80)

# Minutes projection analysis
if 'MIN_L5_MEAN' in betting_df.columns or 'MINUTES' in betting_df.columns:
    min_col = 'MIN_L5_MEAN' if 'MIN_L5_MEAN' in betting_df.columns else 'MINUTES'
    betting_df['MIN_BIN'] = pd.cut(
        betting_df[min_col],
        bins=[0, 20, 25, 30, 35, 50],
        labels=['<20 min', '20-25 min', '25-30 min', '30-35 min', '35+ min']
    )

    min_analysis = betting_df.groupby('MIN_BIN', observed=True).agg({
        'BET_CORRECT': ['count', 'sum', 'mean'],
        'PROFIT': 'sum',
        'ABS_ERROR': 'mean'
    }).round(4)

    min_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit', 'MAE']
    min_analysis['ROI'] = (min_analysis['Total_Profit'] / min_analysis['Count']) * 100

    print("\nPerformance by Minutes Projection:")
    print(min_analysis.to_string())

# Rest days analysis
if 'REST_DAYS' in betting_df.columns:
    betting_df['REST_BIN'] = pd.cut(
        betting_df['REST_DAYS'],
        bins=[-1, 0, 1, 2, 10],
        labels=['Back-to-back (0)', '1 day rest', '2 days rest', '3+ days rest']
    )

    rest_analysis = betting_df.groupby('REST_BIN', observed=True).agg({
        'BET_CORRECT': ['count', 'sum', 'mean'],
        'PROFIT': 'sum'
    }).round(4)

    rest_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit']
    rest_analysis['ROI'] = (rest_analysis['Total_Profit'] / rest_analysis['Count']) * 100

    print("\nPerformance by Rest Days:")
    print(rest_analysis.to_string())

# Home vs Away analysis
if 'HOME_AWAY' in betting_df.columns or 'IS_HOME' in betting_df.columns:
    home_col = 'HOME_AWAY' if 'HOME_AWAY' in betting_df.columns else 'IS_HOME'

    home_analysis = betting_df.groupby(home_col).agg({
        'BET_CORRECT': ['count', 'sum', 'mean'],
        'PROFIT': 'sum'
    }).round(4)

    home_analysis.columns = ['Count', 'Wins', 'Win_Rate', 'Total_Profit']
    home_analysis['ROI'] = (home_analysis['Total_Profit'] / home_analysis['Count']) * 100

    print("\nPerformance by Home/Away:")
    print(home_analysis.to_string())

print("\n" + "="*80)
print("7. ADVANCED SEGMENTATION - OPTIMAL BETTING FILTERS")
print("="*80)

# Find best edge range
print("\nOptimal Edge Range Analysis:")
for min_edge in [3, 5, 7, 10]:
    for max_edge in [15, 20, 30, 100]:
        if max_edge > min_edge:
            filtered = betting_df[(betting_df['ABS_EDGE'] >= min_edge) & (betting_df['ABS_EDGE'] < max_edge)]
            if len(filtered) >= 50:
                wr = filtered['BET_CORRECT'].mean()
                roi = (filtered['PROFIT'].sum() / len(filtered)) * 100
                if wr > 0.524:  # Better than breakeven
                    print(f"Edge {min_edge}-{max_edge} pts: {len(filtered):4d} bets, {wr:.2%} win rate, {roi:+.2%} ROI")

# Combined filters
print("\nBest Combined Filters (win rate > 52.4%):")

# Edge + Bet Type
for edge_bin in betting_df['EDGE_BIN'].cat.categories:
    for bet_type in ['OVER', 'UNDER']:
        filtered = betting_df[(betting_df['EDGE_BIN'] == edge_bin) & (betting_df['BET_TYPE'] == bet_type)]
        if len(filtered) >= 30:
            wr = filtered['BET_CORRECT'].mean()
            roi = (filtered['PROFIT'].sum() / len(filtered)) * 100
            if wr > 0.524:
                print(f"{edge_bin} + {bet_type}: {len(filtered):4d} bets, {wr:.2%} win rate, {roi:+.2%} ROI")

print("\n" + "="*80)
print("8. KEY FINDINGS SUMMARY")
print("="*80)

findings = []

# Finding 1: Overall performance
if win_rate > 0.524 and p_value < 0.05:
    findings.append(f"✓ Model has statistically significant edge (win rate: {win_rate:.2%}, p={p_value:.4f})")
else:
    findings.append(f"✗ Model does NOT have significant edge (win rate: {win_rate:.2%}, p={p_value:.4f})")

# Finding 2: Best edge range
best_edge_bin = edge_analysis.loc[edge_analysis['Win_Rate'].idxmax()]
findings.append(f"✓ Best edge range: {edge_analysis['Win_Rate'].idxmax()} ({best_edge_bin['Win_Rate']:.2%} win rate)")

# Finding 3: Directional bias
if p_value_chi < 0.05:
    best_bet_type = bet_type_analysis.loc[bet_type_analysis['Win_Rate'].idxmax()]
    findings.append(f"✓ Significant directional bias: {bet_type_analysis['Win_Rate'].idxmax()} performs better")
else:
    findings.append(f"○ No significant directional bias between OVER and UNDER")

# Finding 4: MAE
if mae < 5:
    findings.append(f"✓ Excellent prediction accuracy (MAE: {mae:.2f} points)")
elif mae < 7:
    findings.append(f"○ Good prediction accuracy (MAE: {mae:.2f} points)")
else:
    findings.append(f"✗ High prediction error (MAE: {mae:.2f} points - target: <5)")

print("\nKey Findings:")
for i, finding in enumerate(findings, 1):
    print(f"{i}. {finding}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to terminal output")
print(f"Analyzed {total_bets:,} bets from 2024-25 season")
