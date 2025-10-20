"""
Comprehensive analysis of backtest results to identify optimal betting strategies.

Analyzes:
1. Overall performance (no filters)
2. Performance by edge size
3. Performance by bet type (OVER vs UNDER)
4. Performance by player type (star vs non-star)
5. Performance over time
6. Optimal strategy identification
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading backtest data...")
predictions = pd.read_csv('data/results/backtest_2024_25_FIXED_V2.csv')
betting_results = pd.read_csv('data/results/backtest_2024_25_FIXED_V2_betting.csv')

print(f"\nTotal predictions: {len(predictions):,}")
print(f"Filtered bets: {len(betting_results):,}")

# Ensure date column is datetime
predictions['GAME_DATE'] = pd.to_datetime(predictions['GAME_DATE'])

# ==============================================================================
# 1. OVERALL PERFORMANCE (NO FILTERS)
# ==============================================================================
print("\n" + "="*80)
print("1. OVERALL PERFORMANCE (NO FILTERS)")
print("="*80)

# Determine bet type and result for ALL predictions
predictions['edge'] = predictions['prediction'] - predictions['line']
predictions['bet_type'] = predictions['edge'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')

# Determine win/loss for each prediction
predictions['bet_won'] = np.where(
    predictions['bet_type'] == 'OVER',
    predictions['actual'] > predictions['line'],
    predictions['actual'] < predictions['line']
)

# Push (actual == line)
predictions['is_push'] = predictions['actual'] == predictions['line']

# Overall stats
total_bets = len(predictions)
total_wins = predictions['bet_won'].sum()
total_losses = (~predictions['bet_won'] & ~predictions['is_push']).sum()
total_pushes = predictions['is_push'].sum()
win_rate_overall = total_wins / (total_bets - total_pushes) if (total_bets - total_pushes) > 0 else 0

print(f"\nAll Predictions (No Filters):")
print(f"  Total bets: {total_bets:,}")
print(f"  Wins: {total_wins:,}")
print(f"  Losses: {total_losses:,}")
print(f"  Pushes: {total_pushes:,}")
print(f"  Win Rate: {win_rate_overall:.2%}")
print(f"  ROI (assuming -110 odds): {((win_rate_overall * 1.91) - 1) * 100:.2f}%")

# ==============================================================================
# 2. PERFORMANCE BY EDGE SIZE
# ==============================================================================
print("\n" + "="*80)
print("2. PERFORMANCE BY EDGE SIZE")
print("="*80)

# Define edge buckets
predictions['edge_abs'] = predictions['edge'].abs()
predictions['edge_bucket'] = pd.cut(
    predictions['edge_abs'],
    bins=[0, 3, 5, 7, 10, float('inf')],
    labels=['0-3 pts', '3-5 pts', '5-7 pts', '7-10 pts', '10+ pts'],
    include_lowest=True
)

print("\nWin Rate by Edge Size:")
print("-" * 80)
edge_analysis = []

for bucket in ['0-3 pts', '3-5 pts', '5-7 pts', '7-10 pts', '10+ pts']:
    bucket_data = predictions[predictions['edge_bucket'] == bucket]

    if len(bucket_data) == 0:
        continue

    n_bets = len(bucket_data)
    n_pushes = bucket_data['is_push'].sum()
    n_wins = bucket_data['bet_won'].sum()
    n_losses = (~bucket_data['bet_won'] & ~bucket_data['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    # Statistical significance vs 50%
    if (n_bets - n_pushes) > 0:
        z_score = (win_rate - 0.5) / np.sqrt(0.5 * 0.5 / (n_bets - n_pushes))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    else:
        p_value = 1.0
        sig = ""

    edge_analysis.append({
        'Edge Bucket': bucket,
        'Total Bets': n_bets,
        'Wins': n_wins,
        'Losses': n_losses,
        'Pushes': n_pushes,
        'Win Rate': f"{win_rate:.2%}",
        'ROI': f"{roi:.2f}%",
        'P-value': f"{p_value:.4f}",
        'Sig': sig
    })

    print(f"\n{bucket}:")
    print(f"  Total bets: {n_bets:,}")
    print(f"  Wins: {n_wins:,} | Losses: {n_losses:,} | Pushes: {n_pushes:,}")
    print(f"  Win Rate: {win_rate:.2%} {sig}")
    print(f"  ROI: {roi:.2f}%")
    print(f"  P-value: {p_value:.4f}")

edge_df = pd.DataFrame(edge_analysis)

# ==============================================================================
# 3. PERFORMANCE BY BET TYPE (OVER vs UNDER)
# ==============================================================================
print("\n" + "="*80)
print("3. PERFORMANCE BY BET TYPE (OVER vs UNDER)")
print("="*80)

bet_type_analysis = []

for bet_type in ['OVER', 'UNDER']:
    bet_data = predictions[predictions['bet_type'] == bet_type]

    n_bets = len(bet_data)
    n_pushes = bet_data['is_push'].sum()
    n_wins = bet_data['bet_won'].sum()
    n_losses = (~bet_data['bet_won'] & ~bet_data['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    # Statistical significance
    if (n_bets - n_pushes) > 0:
        z_score = (win_rate - 0.5) / np.sqrt(0.5 * 0.5 / (n_bets - n_pushes))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    else:
        p_value = 1.0
        sig = ""

    bet_type_analysis.append({
        'Bet Type': bet_type,
        'Total Bets': n_bets,
        'Wins': n_wins,
        'Losses': n_losses,
        'Pushes': n_pushes,
        'Win Rate': f"{win_rate:.2%}",
        'ROI': f"{roi:.2f}%",
        'P-value': f"{p_value:.4f}",
        'Sig': sig
    })

    print(f"\n{bet_type} Bets:")
    print(f"  Total bets: {n_bets:,}")
    print(f"  Wins: {n_wins:,} | Losses: {n_losses:,} | Pushes: {n_pushes:,}")
    print(f"  Win Rate: {win_rate:.2%} {sig}")
    print(f"  ROI: {roi:.2f}%")

bet_type_df = pd.DataFrame(bet_type_analysis)

# ==============================================================================
# 4. PERFORMANCE BY PLAYER TYPE (Star vs Non-Star)
# ==============================================================================
print("\n" + "="*80)
print("4. PERFORMANCE BY PLAYER TYPE")
print("="*80)

# Define star players based on minutes played (proxy for stars)
# Calculate average minutes per player
player_avg_mins = predictions.groupby('PLAYER_NAME')['MIN'].mean()

# Top 25% of players by minutes = "stars"
star_threshold = player_avg_mins.quantile(0.75)
star_players = set(player_avg_mins[player_avg_mins >= star_threshold].index)

print(f"\nStar player definition: Top 25% by average minutes (>= {star_threshold:.1f} mins)")
print(f"Number of star players: {len(star_players)}")

predictions['is_star'] = predictions['PLAYER_NAME'].isin(star_players)

player_type_analysis = []

for is_star, label in [(True, 'Star Players'), (False, 'Non-Star Players')]:
    player_data = predictions[predictions['is_star'] == is_star]

    n_bets = len(player_data)
    n_pushes = player_data['is_push'].sum()
    n_wins = player_data['bet_won'].sum()
    n_losses = (~player_data['bet_won'] & ~player_data['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    # Statistical significance
    if (n_bets - n_pushes) > 0:
        z_score = (win_rate - 0.5) / np.sqrt(0.5 * 0.5 / (n_bets - n_pushes))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    else:
        p_value = 1.0
        sig = ""

    player_type_analysis.append({
        'Player Type': label,
        'Total Bets': n_bets,
        'Wins': n_wins,
        'Losses': n_losses,
        'Pushes': n_pushes,
        'Win Rate': f"{win_rate:.2%}",
        'ROI': f"{roi:.2f}%",
        'P-value': f"{p_value:.4f}",
        'Sig': sig
    })

    print(f"\n{label}:")
    print(f"  Total bets: {n_bets:,}")
    print(f"  Wins: {n_wins:,} | Losses: {n_losses:,} | Pushes: {n_pushes:,}")
    print(f"  Win Rate: {win_rate:.2%} {sig}")
    print(f"  ROI: {roi:.2f}%")

player_type_df = pd.DataFrame(player_type_analysis)

# ==============================================================================
# 5. PERFORMANCE OVER TIME
# ==============================================================================
print("\n" + "="*80)
print("5. PERFORMANCE OVER TIME")
print("="*80)

predictions['month'] = predictions['GAME_DATE'].dt.to_period('M')

time_analysis = []

for month in sorted(predictions['month'].unique()):
    month_data = predictions[predictions['month'] == month]

    n_bets = len(month_data)
    n_pushes = month_data['is_push'].sum()
    n_wins = month_data['bet_won'].sum()
    n_losses = (~month_data['bet_won'] & ~month_data['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    time_analysis.append({
        'Month': str(month),
        'Total Bets': n_bets,
        'Wins': n_wins,
        'Losses': n_losses,
        'Pushes': n_pushes,
        'Win Rate': f"{win_rate:.2%}",
        'ROI': f"{roi:.2f}%"
    })

    print(f"\n{month}:")
    print(f"  Total bets: {n_bets:,}")
    print(f"  Wins: {n_wins:,} | Losses: {n_losses:,} | Pushes: {n_pushes:,}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  ROI: {roi:.2f}%")

time_df = pd.DataFrame(time_analysis)

# ==============================================================================
# 6. COMBINATION ANALYSIS - FINDING OPTIMAL STRATEGY
# ==============================================================================
print("\n" + "="*80)
print("6. OPTIMAL STRATEGY IDENTIFICATION")
print("="*80)

strategies = []

# Test various combinations
test_configs = [
    # Edge-based filters
    {'name': 'Edge >= 3pts', 'filter': lambda df: df['edge_abs'] >= 3},
    {'name': 'Edge >= 5pts', 'filter': lambda df: df['edge_abs'] >= 5},
    {'name': 'Edge >= 7pts', 'filter': lambda df: df['edge_abs'] >= 7},
    {'name': '3 <= Edge < 7', 'filter': lambda df: (df['edge_abs'] >= 3) & (df['edge_abs'] < 7)},
    {'name': '5 <= Edge < 10', 'filter': lambda df: (df['edge_abs'] >= 5) & (df['edge_abs'] < 10)},

    # Bet type filters
    {'name': 'OVER bets only', 'filter': lambda df: df['bet_type'] == 'OVER'},
    {'name': 'UNDER bets only', 'filter': lambda df: df['bet_type'] == 'UNDER'},

    # Player type filters
    {'name': 'Star players only', 'filter': lambda df: df['is_star'] == True},
    {'name': 'Non-star players', 'filter': lambda df: df['is_star'] == False},

    # Combinations
    {'name': 'Stars + Edge >= 5', 'filter': lambda df: (df['is_star'] == True) & (df['edge_abs'] >= 5)},
    {'name': 'Stars + 3 <= Edge < 7', 'filter': lambda df: (df['is_star'] == True) & (df['edge_abs'] >= 3) & (df['edge_abs'] < 7)},
    {'name': 'Non-stars + Edge >= 5', 'filter': lambda df: (df['is_star'] == False) & (df['edge_abs'] >= 5)},
    {'name': 'OVER + Edge >= 5', 'filter': lambda df: (df['bet_type'] == 'OVER') & (df['edge_abs'] >= 5)},
    {'name': 'UNDER + Edge >= 5', 'filter': lambda df: (df['bet_type'] == 'UNDER') & (df['edge_abs'] >= 5)},
]

print("\nTesting various strategies...")
print("-" * 80)

for config in test_configs:
    filtered = predictions[config['filter'](predictions)]

    n_bets = len(filtered)

    if n_bets < 100:  # Skip strategies with too few bets
        continue

    n_pushes = filtered['is_push'].sum()
    n_wins = filtered['bet_won'].sum()
    n_losses = (~filtered['bet_won'] & ~filtered['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    # Statistical significance
    if (n_bets - n_pushes) > 0:
        z_score = (win_rate - 0.5) / np.sqrt(0.5 * 0.5 / (n_bets - n_pushes))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    else:
        p_value = 1.0
        sig = ""

    strategies.append({
        'Strategy': config['name'],
        'Total Bets': n_bets,
        'Win Rate': win_rate,
        'ROI': roi,
        'P-value': p_value,
        'Sig': sig,
        'Wins': n_wins,
        'Losses': n_losses
    })

# Sort by win rate
strategies_df = pd.DataFrame(strategies).sort_values('Win Rate', ascending=False)

print("\nTop 10 Strategies by Win Rate:")
print("-" * 80)
for idx, row in strategies_df.head(10).iterrows():
    print(f"\n{row['Strategy']}:")
    print(f"  Bets: {row['Total Bets']:,} | Win Rate: {row['Win Rate']:.2%} {row['Sig']}")
    print(f"  ROI: {row['ROI']:.2f}% | P-value: {row['P-value']:.4f}")
    print(f"  W/L: {row['Wins']}/{row['Losses']}")

# ==============================================================================
# 7. DETAILED EDGE SIZE ANALYSIS (Finer Granularity)
# ==============================================================================
print("\n" + "="*80)
print("7. DETAILED EDGE SIZE ANALYSIS (1-point buckets)")
print("="*80)

# Create 1-point buckets up to 15 points
edge_buckets_detailed = list(range(0, 16))
predictions['edge_bucket_detailed'] = pd.cut(
    predictions['edge_abs'],
    bins=edge_buckets_detailed + [float('inf')],
    labels=[f"{i}-{i+1} pts" for i in edge_buckets_detailed[:-1]] + ['15+ pts'],
    include_lowest=True
)

detailed_edge_analysis = []

for bucket in predictions['edge_bucket_detailed'].cat.categories:
    bucket_data = predictions[predictions['edge_bucket_detailed'] == bucket]

    n_bets = len(bucket_data)

    if n_bets < 50:  # Skip buckets with too few samples
        continue

    n_pushes = bucket_data['is_push'].sum()
    n_wins = bucket_data['bet_won'].sum()
    n_losses = (~bucket_data['bet_won'] & ~bucket_data['is_push']).sum()

    win_rate = n_wins / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    roi = ((win_rate * 1.91) - 1) * 100

    detailed_edge_analysis.append({
        'Edge Bucket': bucket,
        'Total Bets': n_bets,
        'Win Rate': f"{win_rate:.2%}",
        'ROI': f"{roi:.2f}%"
    })

detailed_edge_df = pd.DataFrame(detailed_edge_analysis)
print("\n", detailed_edge_df.to_string(index=False))

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save all analysis tables
with pd.ExcelWriter('data/results/backtest_analysis_comprehensive.xlsx') as writer:
    edge_df.to_excel(writer, sheet_name='Edge Size Analysis', index=False)
    bet_type_df.to_excel(writer, sheet_name='Bet Type Analysis', index=False)
    player_type_df.to_excel(writer, sheet_name='Player Type Analysis', index=False)
    time_df.to_excel(writer, sheet_name='Time Analysis', index=False)
    strategies_df.to_excel(writer, sheet_name='Strategy Comparison', index=False)
    detailed_edge_df.to_excel(writer, sheet_name='Detailed Edge Analysis', index=False)

print("\nAnalysis saved to: data/results/backtest_analysis_comprehensive.xlsx")

# ==============================================================================
# FINAL RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

# Find strategies with win rate >= 52% and at least 500 bets
viable_strategies = strategies_df[
    (strategies_df['Win Rate'] >= 0.52) &
    (strategies_df['Total Bets'] >= 500)
]

if len(viable_strategies) > 0:
    print("\nVIABLE STRATEGIES (Win Rate >= 52%, Sample Size >= 500):")
    print("-" * 80)
    for idx, row in viable_strategies.iterrows():
        print(f"\n{row['Strategy']}:")
        print(f"  Win Rate: {row['Win Rate']:.2%} {row['Sig']}")
        print(f"  Total Bets: {row['Total Bets']:,}")
        print(f"  ROI: {row['ROI']:.2f}%")
        print(f"  Statistical Significance: p={row['P-value']:.4f}")
else:
    print("\nNo strategies found with >= 52% win rate AND >= 500 bets.")
    print("\nBest strategies by win rate (regardless of sample size):")
    print("-" * 80)
    for idx, row in strategies_df.head(5).iterrows():
        print(f"\n{row['Strategy']}:")
        print(f"  Win Rate: {row['Win Rate']:.2%} {row['Sig']}")
        print(f"  Total Bets: {row['Total Bets']:,}")
        print(f"  ROI: {row['ROI']:.2f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
