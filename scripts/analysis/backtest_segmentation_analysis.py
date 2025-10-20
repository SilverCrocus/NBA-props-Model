#!/usr/bin/env python3
"""
Comprehensive Backtest Segmentation Analysis

Analyzes V2 model backtest results to identify where edge exists:
1. Overall performance (no filters)
2. Performance by edge size
3. Performance by bet type (OVER vs UNDER)
4. Performance by player type (stars vs non-stars)
5. Performance over time
6. Optimal strategy identification
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================

BACKTEST_PATH = "data/results/backtest_2024_25_FIXED_V2.csv"  # Use full predictions file
BETTING_PATH = "data/results/backtest_2024_25_FIXED_V2_betting.csv"  # Filtered betting file
OUTPUT_PATH = "data/results/backtest_analysis_comprehensive.xlsx"

# Star players (top tier, efficient markets)
STAR_PLAYERS = [
    'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
    'Luka Doncic', 'Joel Embiid', 'Nikola Jokic', 'Jayson Tatum', 'Anthony Davis',
    'Damian Lillard', 'Devin Booker', 'Jimmy Butler', 'Kawhi Leonard',
    'Paul George', 'Kyrie Irving', 'James Harden', 'Trae Young', 'Donovan Mitchell',
    'Jaylen Brown', 'LaMelo Ball', 'De\'Aaron Fox', 'Anthony Edwards', 'Pascal Siakam',
    'Tyrese Haliburton', 'DeMar DeRozan', 'Shai Gilgeous-Alexander', 'Bam Adebayo',
    'Domantas Sabonis', 'Julius Randle', 'Zion Williamson', 'Brandon Ingram',
    'Paolo Banchero', 'Franz Wagner', 'Cade Cunningham', 'Jalen Green',
    'Alperen Sengun', 'Lauri Markkanen', 'Walker Kessler', 'Evan Mobley',
    'Darius Garland', 'Derrick White', 'OG Anunoby', 'RJ Barrett', 'Immanuel Quickley',
    'Mikal Bridges', 'Cameron Johnson', 'Nic Claxton', 'Spencer Dinwiddie',
    'Cam Thomas', 'Karl-Anthony Towns', 'Jalen Brunson', 'Scottie Barnes',
    'Dejounte Murray', 'Fred VanVleet', 'Jrue Holiday', 'CJ McCollum',
    'Tobias Harris', 'Khris Middleton', 'Tyler Herro', 'Jarrett Allen',
    'Kristaps Porzingis', 'Clint Capela', 'Jaren Jackson Jr'
]

print("=" * 80)
print("COMPREHENSIVE BACKTEST SEGMENTATION ANALYSIS")
print("=" * 80)
print()

# ======================================================================
# LOAD DATA
# ======================================================================

print("1. Loading backtest results...")
df = pd.read_csv(BACKTEST_PATH)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

print(f"   ✅ Loaded {len(df):,} predictions")
print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
print()

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def calculate_betting_metrics(predictions_df, line_col='line', odds=-110):
    """
    Calculate win rate, ROI, and other betting metrics

    Assumes:
    - predicted_PRA and actual_PRA columns exist
    - If line_col provided, matches predictions to betting lines
    """
    if line_col not in predictions_df.columns:
        # No betting lines, just evaluate predictions
        return {
            'total_predictions': len(predictions_df),
            'mae': (predictions_df['predicted_PRA'] - predictions_df['actual_PRA']).abs().mean()
        }

    # Determine bet type
    predictions_df = predictions_df.copy()
    predictions_df['edge'] = predictions_df['predicted_PRA'] - predictions_df[line_col]
    predictions_df['bet_type'] = predictions_df['edge'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')

    # Determine bet outcome
    predictions_df['bet_correct'] = (
        ((predictions_df['bet_type'] == 'OVER') & (predictions_df['actual_PRA'] > predictions_df[line_col])) |
        ((predictions_df['bet_type'] == 'UNDER') & (predictions_df['actual_PRA'] < predictions_df[line_col]))
    )

    # Calculate metrics
    total_bets = len(predictions_df)
    wins = predictions_df['bet_correct'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    # ROI calculation (assuming -110 odds)
    profit_per_win = 100 * (100 / 110)  # $90.91 per $100 bet
    loss_per_loss = -100
    total_profit = (wins * profit_per_win) + (losses * loss_per_loss)
    total_wagered = total_bets * 100
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Statistical significance (binomial test against 50%)
    p_value = stats.binomtest(wins, total_bets, p=0.50, alternative='two-sided').pvalue if total_bets > 0 else 1.0

    # Significance markers
    if p_value < 0.001:
        sig_marker = '***'
    elif p_value < 0.01:
        sig_marker = '**'
    elif p_value < 0.05:
        sig_marker = '*'
    else:
        sig_marker = ''

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'roi': roi,
        'p_value': p_value,
        'significance': sig_marker,
        'mae': (predictions_df['predicted_PRA'] - predictions_df['actual_PRA']).abs().mean()
    }

def format_results_table(results_list, title):
    """Format results as a pandas DataFrame for Excel export"""
    df_results = pd.DataFrame(results_list)

    # Format percentages
    if 'win_rate' in df_results.columns:
        df_results['win_rate_pct'] = (df_results['win_rate'] * 100).round(2).astype(str) + '%' + df_results['significance']

    if 'roi' in df_results.columns:
        df_results['roi_pct'] = df_results['roi'].round(2).astype(str) + '%'

    return df_results

# ======================================================================
# ANALYSIS 1: OVERALL PERFORMANCE (NO FILTERS)
# ======================================================================

print("2. Analyzing overall performance (no filters)...")

# Need to match predictions to betting lines
# Check if we have betting lines in the data
if 'line' not in df.columns:
    print("   ⚠️  No betting lines found in backtest data")
    print("   ⚠️  Will analyze predictions only (no win rate calculation)")

    overall_metrics = {
        'description': 'All Predictions',
        'total_predictions': len(df),
        'mae': (df['predicted_PRA'] - df['actual_PRA']).abs().mean()
    }
else:
    overall_metrics = calculate_betting_metrics(df)
    overall_metrics['description'] = 'All Predictions (No Filters)'

print(f"   Total predictions: {overall_metrics.get('total_predictions', len(df)):,}")
if 'win_rate' in overall_metrics:
    print(f"   Win rate: {overall_metrics['win_rate']*100:.2f}%")
    print(f"   ROI: {overall_metrics['roi']:.2f}%")
print(f"   MAE: {overall_metrics.get('mae', 0):.2f} pts")
print()

# ======================================================================
# ANALYSIS 2: PERFORMANCE BY EDGE SIZE
# ======================================================================

print("3. Analyzing performance by edge size...")

if 'line' in df.columns:
    df['edge'] = df['predicted_PRA'] - df['line']
    df['abs_edge'] = df['edge'].abs()

    # Define edge buckets
    edge_buckets = [
        ('0-3 pts', 0, 3),
        ('3-5 pts', 3, 5),
        ('5-7 pts', 5, 7),
        ('7-10 pts', 7, 10),
        ('10+ pts', 10, 100)
    ]

    edge_results = []

    for label, min_edge, max_edge in edge_buckets:
        bucket_df = df[(df['abs_edge'] >= min_edge) & (df['abs_edge'] < max_edge)]

        if len(bucket_df) > 0:
            metrics = calculate_betting_metrics(bucket_df)
            metrics['description'] = label
            metrics['min_edge'] = min_edge
            metrics['max_edge'] = max_edge
            edge_results.append(metrics)

            print(f"   {label}: {metrics['win_rate']*100:.2f}% win rate, {metrics['total_bets']:,} bets, ROI {metrics['roi']:.2f}%")

    print()
else:
    print("   ⚠️  No betting lines - skipping edge analysis")
    edge_results = []
    print()

# ======================================================================
# ANALYSIS 3: PERFORMANCE BY BET TYPE (OVER vs UNDER)
# ======================================================================

print("4. Analyzing performance by bet type...")

if 'line' in df.columns:
    # Classify bets
    df_over = df[df['predicted_PRA'] > df['line']]
    df_under = df[df['predicted_PRA'] < df['line']]

    over_metrics = calculate_betting_metrics(df_over)
    over_metrics['description'] = 'OVER Bets'

    under_metrics = calculate_betting_metrics(df_under)
    under_metrics['description'] = 'UNDER Bets'

    bet_type_results = [over_metrics, under_metrics]

    print(f"   OVER: {over_metrics['win_rate']*100:.2f}% win rate, {over_metrics['total_bets']:,} bets")
    print(f"   UNDER: {under_metrics['win_rate']*100:.2f}% win rate, {under_metrics['total_bets']:,} bets")
    print(f"   Difference: {(over_metrics['win_rate'] - under_metrics['win_rate'])*100:.2f} pp")
    print()
else:
    print("   ⚠️  No betting lines - skipping bet type analysis")
    bet_type_results = []
    print()

# ======================================================================
# ANALYSIS 4: PERFORMANCE BY PLAYER TYPE (STARS VS NON-STARS)
# ======================================================================

print("5. Analyzing performance by player type...")

# Classify players as stars
df['is_star'] = df['PLAYER_NAME'].isin(STAR_PLAYERS)

df_stars = df[df['is_star'] == True]
df_non_stars = df[df['is_star'] == False]

if 'line' in df.columns:
    star_metrics = calculate_betting_metrics(df_stars)
    star_metrics['description'] = 'Star Players'

    non_star_metrics = calculate_betting_metrics(df_non_stars)
    non_star_metrics['description'] = 'Non-Star Players'

    player_type_results = [star_metrics, non_star_metrics]

    print(f"   Stars: {star_metrics['win_rate']*100:.2f}% win rate, {star_metrics['total_bets']:,} bets")
    print(f"   Non-stars: {non_star_metrics['win_rate']*100:.2f}% win rate, {non_star_metrics['total_bets']:,} bets")
    print(f"   Difference: {(non_star_metrics['win_rate'] - star_metrics['win_rate'])*100:.2f} pp")
    print()
else:
    star_mae = (df_stars['predicted_PRA'] - df_stars['actual_PRA']).abs().mean()
    non_star_mae = (df_non_stars['predicted_PRA'] - df_non_stars['actual_PRA']).abs().mean()

    player_type_results = [
        {'description': 'Star Players', 'total_predictions': len(df_stars), 'mae': star_mae},
        {'description': 'Non-Star Players', 'total_predictions': len(df_non_stars), 'mae': non_star_mae}
    ]

    print(f"   Stars: {star_mae:.2f} MAE, {len(df_stars):,} predictions")
    print(f"   Non-stars: {non_star_mae:.2f} MAE, {len(df_non_stars):,} predictions")
    print()

# ======================================================================
# ANALYSIS 5: PERFORMANCE OVER TIME (MONTHLY)
# ======================================================================

print("6. Analyzing performance over time...")

df['month'] = df['GAME_DATE'].dt.to_period('M')
unique_months = sorted(df['month'].unique())

monthly_results = []

for month in unique_months:
    month_df = df[df['month'] == month]

    if 'line' in df.columns:
        metrics = calculate_betting_metrics(month_df)
        metrics['description'] = str(month)
        metrics['month'] = str(month)
        monthly_results.append(metrics)

        print(f"   {month}: {metrics['win_rate']*100:.2f}% win rate, {metrics['total_bets']:,} bets")
    else:
        mae = (month_df['predicted_PRA'] - month_df['actual_PRA']).abs().mean()
        monthly_results.append({
            'description': str(month),
            'month': str(month),
            'total_predictions': len(month_df),
            'mae': mae
        })
        print(f"   {month}: {mae:.2f} MAE, {len(month_df):,} predictions")

print()

# ======================================================================
# ANALYSIS 6: OPTIMAL STRATEGY IDENTIFICATION
# ======================================================================

print("7. Testing optimal strategy combinations...")

if 'line' in df.columns:
    strategies = [
        # Pure edge filters
        ('All bets (baseline)', lambda x: x),
        ('Edge >= 3 pts', lambda x: x[x['abs_edge'] >= 3]),
        ('Edge >= 5 pts', lambda x: x[x['abs_edge'] >= 5]),
        ('Edge >= 7 pts', lambda x: x[x['abs_edge'] >= 7]),
        ('Edge >= 10 pts', lambda x: x[x['abs_edge'] >= 10]),

        # Edge ranges
        ('Edge 3-5 pts', lambda x: x[(x['abs_edge'] >= 3) & (x['abs_edge'] < 5)]),
        ('Edge 5-7 pts', lambda x: x[(x['abs_edge'] >= 5) & (x['abs_edge'] < 7)]),
        ('Edge 7-10 pts', lambda x: x[(x['abs_edge'] >= 7) & (x['abs_edge'] < 10)]),
        ('Edge 5-7 OR 10+ pts', lambda x: x[((x['abs_edge'] >= 5) & (x['abs_edge'] < 7)) | (x['abs_edge'] >= 10)]),

        # Bet type filters
        ('OVER only', lambda x: x[x['predicted_PRA'] > x['line']]),
        ('UNDER only', lambda x: x[x['predicted_PRA'] < x['line']]),

        # Player type filters
        ('Non-stars only', lambda x: x[~x['is_star']]),
        ('Stars only', lambda x: x[x['is_star']]),

        # Combinations
        ('Non-stars + Edge >= 5', lambda x: x[~x['is_star'] & (x['abs_edge'] >= 5)]),
        ('Non-stars + Edge 5-7 OR 10+', lambda x: x[~x['is_star'] & (((x['abs_edge'] >= 5) & (x['abs_edge'] < 7)) | (x['abs_edge'] >= 10))]),
        ('UNDER + Edge >= 5', lambda x: x[(x['predicted_PRA'] < x['line']) & (x['abs_edge'] >= 5)]),
    ]

    strategy_results = []

    for strategy_name, filter_func in strategies:
        try:
            filtered_df = filter_func(df)

            if len(filtered_df) >= 30:  # Minimum sample size
                metrics = calculate_betting_metrics(filtered_df)
                metrics['description'] = strategy_name
                strategy_results.append(metrics)

                # Print only if win rate > 52% (profitable)
                if metrics['win_rate'] > 0.52:
                    print(f"   ✅ {strategy_name}: {metrics['win_rate']*100:.2f}% win rate, {metrics['total_bets']:,} bets, ROI {metrics['roi']:.2f}%")
        except Exception as e:
            print(f"   ⚠️  Error testing {strategy_name}: {e}")

    print()
else:
    print("   ⚠️  No betting lines - skipping strategy testing")
    strategy_results = []
    print()

# ======================================================================
# SAVE RESULTS TO EXCEL
# ======================================================================

print("8. Saving results to Excel...")

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    # Sheet 1: Overall Summary
    summary_df = pd.DataFrame([overall_metrics])
    summary_df.to_excel(writer, sheet_name='Overall', index=False)

    # Sheet 2: Edge Size Analysis
    if edge_results:
        edge_df = format_results_table(edge_results, 'Edge Size')
        edge_df.to_excel(writer, sheet_name='By Edge Size', index=False)

    # Sheet 3: Bet Type Analysis
    if bet_type_results:
        bet_type_df = format_results_table(bet_type_results, 'Bet Type')
        bet_type_df.to_excel(writer, sheet_name='By Bet Type', index=False)

    # Sheet 4: Player Type Analysis
    player_type_df = format_results_table(player_type_results, 'Player Type')
    player_type_df.to_excel(writer, sheet_name='By Player Type', index=False)

    # Sheet 5: Monthly Performance
    monthly_df = format_results_table(monthly_results, 'Monthly')
    monthly_df.to_excel(writer, sheet_name='By Month', index=False)

    # Sheet 6: Strategy Comparison
    if strategy_results:
        # Sort by win rate
        strategy_results_sorted = sorted(strategy_results, key=lambda x: x['win_rate'], reverse=True)
        strategy_df = format_results_table(strategy_results_sorted, 'Strategies')
        strategy_df.to_excel(writer, sheet_name='Strategy Comparison', index=False)

print(f"   ✅ Saved to {OUTPUT_PATH}")
print()

# ======================================================================
# FINAL RECOMMENDATIONS
# ======================================================================

print("=" * 80)
print("KEY FINDINGS & RECOMMENDATIONS")
print("=" * 80)
print()

if 'line' in df.columns:
    print("📊 OVERALL PERFORMANCE:")
    print(f"   Win Rate: {overall_metrics['win_rate']*100:.2f}% (target: 52%+)")
    print(f"   ROI: {overall_metrics['roi']:.2f}% (target: 5-10%)")
    print(f"   Total Bets: {overall_metrics['total_bets']:,}")
    print()

    # Find best strategy
    if strategy_results:
        best_strategy = max(strategy_results, key=lambda x: x['win_rate'] if x['total_bets'] >= 100 else 0)
        print(f"🏆 BEST STRATEGY (sample >=100):")
        print(f"   {best_strategy['description']}")
        print(f"   Win Rate: {best_strategy['win_rate']*100:.2f}%")
        print(f"   ROI: {best_strategy['roi']:.2f}%")
        print(f"   Total Bets: {best_strategy['total_bets']:,}")
        print()

    # Check star vs non-star
    if player_type_results and len(player_type_results) >= 2:
        non_star_wr = [r for r in player_type_results if 'Non-Star' in r['description']][0]['win_rate']
        star_wr = [r for r in player_type_results if r['description'] == 'Star Players'][0]['win_rate']

        if non_star_wr > star_wr + 0.02:  # 2 pp advantage
            print(f"✅ NON-STAR ADVANTAGE CONFIRMED:")
            print(f"   Non-stars: {non_star_wr*100:.2f}% vs Stars: {star_wr*100:.2f}%")
            print(f"   Recommendation: Continue excluding star players")
            print()

    # Check OVER vs UNDER
    if bet_type_results and len(bet_type_results) >= 2:
        over_wr = bet_type_results[0]['win_rate']
        under_wr = bet_type_results[1]['win_rate']

        if abs(over_wr - under_wr) > 0.02:  # 2 pp difference
            better_direction = 'OVER' if over_wr > under_wr else 'UNDER'
            print(f"⚡ DIRECTIONAL BIAS DETECTED:")
            print(f"   OVER: {over_wr*100:.2f}% vs UNDER: {under_wr*100:.2f}%")
            print(f"   Recommendation: Focus on {better_direction} bets")
            print()

print("📁 DETAILED RESULTS:")
print(f"   Full analysis saved to: {OUTPUT_PATH}")
print()

print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print()
print("1. Review Excel file for detailed breakdowns")
print("2. Identify strategy with highest win rate (>52%) and large sample (>500 bets)")
print("3. If win rate < 52%, proceed to Phase 2 (Calibration)")
print("4. If win rate > 52% but < 54%, proceed to Phase 3 (Retraining)")
print("5. If win rate > 54%, model is already good - optimize bet sizing")
print()
print("=" * 80)
