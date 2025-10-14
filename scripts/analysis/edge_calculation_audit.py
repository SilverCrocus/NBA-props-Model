"""
Deep Dive: Edge Calculation Audit
Investigates why edge calibration is so poor
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EDGE CALCULATION AUDIT")
print("="*80)

# Load backtest data
backtest_df = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2024_25.csv')

print(f"\nDataset: {len(backtest_df)} bets")
print("\nColumns available:")
print(backtest_df.columns.tolist())

# Examine edge calculation
print("\n" + "="*80)
print("1. EDGE CALCULATION INSPECTION")
print("="*80)

sample = backtest_df.head(20)
print("\nSample of bet data:")
print(sample[['PLAYER_NAME', 'predicted_pra', 'betting_line', 'edge',
              'bet_side', 'actual_pra', 'bet_won']].to_string())

# Verify edge calculation
print("\n" + "="*80)
print("2. EDGE CALCULATION VERIFICATION")
print("="*80)

# Check if edge = predicted - line
backtest_df['edge_check'] = backtest_df['predicted_pra'] - backtest_df['betting_line']
backtest_df['edge_diff'] = backtest_df['edge'] - backtest_df['edge_check']

print(f"\nEdge calculation matches (predicted - line):")
print(f"  Max difference: {backtest_df['edge_diff'].abs().max():.6f}")
print(f"  Mean difference: {backtest_df['edge_diff'].abs().mean():.6f}")
print(f"  Calculation verified: {(backtest_df['edge_diff'].abs() < 0.01).all()}")

# Analyze edge vs actual outcome
print("\n" + "="*80)
print("3. EDGE VS ACTUAL OUTCOME ANALYSIS")
print("="*80)

# For OVER bets: win if actual > line
# For UNDER bets: win if actual < line
backtest_df['should_take_over'] = backtest_df['predicted_pra'] > backtest_df['betting_line']
backtest_df['should_take_under'] = backtest_df['predicted_pra'] < backtest_df['betting_line']
backtest_df['actual_over_wins'] = backtest_df['actual_pra'] > backtest_df['betting_line']
backtest_df['actual_under_wins'] = backtest_df['actual_pra'] < backtest_df['betting_line']

# Check bet_side assignment
print("\nBet side distribution:")
print(backtest_df['bet_side'].value_counts())

print("\nBet side logic verification:")
over_bets = backtest_df[backtest_df['bet_side'] == 'over']
under_bets = backtest_df[backtest_df['bet_side'] == 'under']

print(f"\nOVER bets:")
print(f"  Count: {len(over_bets)}")
print(f"  Predicted > Line: {(over_bets['predicted_pra'] > over_bets['betting_line']).sum()}")
print(f"  Actual > Line (wins): {(over_bets['actual_pra'] > over_bets['betting_line']).sum()}")
print(f"  Win rate: {(over_bets['actual_pra'] > over_bets['betting_line']).mean() * 100:.2f}%")
print(f"  Average edge: {over_bets['edge'].mean():.2f}")
print(f"  Average pred-line gap: {(over_bets['predicted_pra'] - over_bets['betting_line']).mean():.2f}")

print(f"\nUNDER bets:")
print(f"  Count: {len(under_bets)}")
print(f"  Predicted < Line: {(under_bets['predicted_pra'] < under_bets['betting_line']).sum()}")
print(f"  Actual < Line (wins): {(under_bets['actual_pra'] < under_bets['betting_line']).sum()}")
print(f"  Win rate: {(under_bets['actual_pra'] < under_bets['betting_line']).mean() * 100:.2f}%")
print(f"  Average edge: {under_bets['edge'].mean():.2f}")
print(f"  Average pred-line gap: {(under_bets['predicted_pra'] - under_bets['betting_line']).mean():.2f}")

# Analyze by edge magnitude
print("\n" + "="*80)
print("4. PERFORMANCE BY EDGE MAGNITUDE")
print("="*80)

edge_bins = [-100, -10, -5, -2, 0, 2, 5, 10, 100]
edge_labels = ['< -10', '-10 to -5', '-5 to -2', '-2 to 0',
               '0 to 2', '2 to 5', '5 to 10', '> 10']

backtest_df['edge_bin'] = pd.cut(backtest_df['edge'], bins=edge_bins, labels=edge_labels)

print("\nDetailed edge bin analysis:")
for label in edge_labels:
    bin_data = backtest_df[backtest_df['edge_bin'] == label]
    if len(bin_data) > 0:
        win_rate = bin_data['bet_won'].mean() * 100
        avg_edge = bin_data['edge'].mean()
        expected_wr = avg_edge + 50  # Assuming edge = win_prob - 50
        calibration_error = win_rate - expected_wr

        print(f"\n{label}:")
        print(f"  Count: {len(bin_data)}")
        print(f"  Avg Edge: {avg_edge:.2f}%")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Expected WR: {expected_wr:.2f}%")
        print(f"  Calibration Error: {calibration_error:.2f}%")
        print(f"  Over bets: {(bin_data['bet_side'] == 'over').sum()}")
        print(f"  Under bets: {(bin_data['bet_side'] == 'under').sum()}")

# Check if edge calculation accounts for betting odds
print("\n" + "="*80)
print("5. BETTING ODDS ANALYSIS")
print("="*80)

print("\nImplied probability from odds:")
print(f"  Avg Over Implied Prob: {backtest_df['over_implied_prob'].mean():.4f} ({backtest_df['over_implied_prob'].mean()*100:.2f}%)")
print(f"  Avg Under Implied Prob: {backtest_df['under_implied_prob'].mean():.4f} ({backtest_df['under_implied_prob'].mean()*100:.2f}%)")
print(f"  Avg Total (vig): {(backtest_df['over_implied_prob'] + backtest_df['under_implied_prob']).mean():.4f}")

# Check if edge should be: predicted_prob - implied_prob
# For OVER bets: need P(actual > line)
# For UNDER bets: need P(actual < line)

print("\nInvestigating alternative edge calculation...")
print("\nHypothesis: Current edge = predicted_pra - line")
print("           Better edge = P(win) - breakeven_probability")

# Simulate what the edge SHOULD be if properly calculated
# Assume normal distribution around prediction with std from training
mae = 7.97  # From overall analysis
std_approx = mae * 1.25  # Rough approximation

# For each bet, calculate P(actual > line) for over, P(actual < line) for under
from scipy.stats import norm

backtest_df['prob_over_line'] = norm.cdf(
    (backtest_df['predicted_pra'] - backtest_df['betting_line']) / std_approx
)
backtest_df['prob_under_line'] = 1 - backtest_df['prob_over_line']

# Theoretical edge accounting for odds
over_mask = backtest_df['bet_side'] == 'over'
under_mask = backtest_df['bet_side'] == 'under'

backtest_df['theoretical_edge'] = 0.0
backtest_df.loc[over_mask, 'theoretical_edge'] = (
    backtest_df.loc[over_mask, 'prob_over_line'] -
    backtest_df.loc[over_mask, 'over_implied_prob']
) * 100

backtest_df.loc[under_mask, 'theoretical_edge'] = (
    backtest_df.loc[under_mask, 'prob_under_line'] -
    backtest_df.loc[under_mask, 'under_implied_prob']
) * 100

print(f"\nCurrent edge calculation:")
print(f"  Mean: {backtest_df['edge'].mean():.2f}%")
print(f"  Std: {backtest_df['edge'].std():.2f}%")
print(f"  Min: {backtest_df['edge'].min():.2f}%")
print(f"  Max: {backtest_df['edge'].max():.2f}%")

print(f"\nTheoretical edge (probability-based):")
print(f"  Mean: {backtest_df['theoretical_edge'].mean():.2f}%")
print(f"  Std: {backtest_df['theoretical_edge'].std():.2f}%")
print(f"  Min: {backtest_df['theoretical_edge'].min():.2f}%")
print(f"  Max: {backtest_df['theoretical_edge'].max():.2f}%")

# Compare calibration
print("\n" + "="*80)
print("6. CALIBRATION COMPARISON")
print("="*80)

def analyze_calibration(df, edge_col, label):
    bins = pd.cut(df[edge_col], bins=[-100, -5, 0, 5, 100],
                  labels=['Negative', 'Near Zero', 'Positive'])
    results = df.groupby(bins).agg({
        'bet_won': ['count', 'sum', 'mean'],
        edge_col: 'mean'
    })
    results.columns = ['Count', 'Wins', 'Win_Rate', 'Avg_Edge']
    results['Win_Rate'] *= 100

    print(f"\n{label}:")
    print(results.to_string())

    return results

current_results = analyze_calibration(backtest_df, 'edge', 'Current Edge')
theoretical_results = analyze_calibration(backtest_df, 'theoretical_edge', 'Theoretical Edge')

# Find bets where we SHOULD have bet opposite
print("\n" + "="*80)
print("7. MAJOR DISCREPANCIES")
print("="*80)

# Current method says positive edge, theoretical says negative
backtest_df['edge_disagreement'] = (
    ((backtest_df['edge'] > 0) & (backtest_df['theoretical_edge'] < -5)) |
    ((backtest_df['edge'] < 0) & (backtest_df['theoretical_edge'] > 5))
)

disagreements = backtest_df[backtest_df['edge_disagreement']]
print(f"\nBets with major disagreement (>5% edge difference): {len(disagreements)}")

if len(disagreements) > 0:
    print("\nSample of disagreements:")
    print(disagreements[['PLAYER_NAME', 'predicted_pra', 'betting_line', 'actual_pra',
                         'bet_side', 'edge', 'theoretical_edge', 'bet_won']].head(20).to_string())

    print(f"\nDisagreement analysis:")
    print(f"  Win rate on disagreements: {disagreements['bet_won'].mean() * 100:.2f}%")
    print(f"  Average current edge: {disagreements['edge'].mean():.2f}%")
    print(f"  Average theoretical edge: {disagreements['theoretical_edge'].mean():.2f}%")

# Final diagnosis
print("\n" + "="*80)
print("8. DIAGNOSIS & RECOMMENDATIONS")
print("="*80)

print("\nKey Findings:")
print("1. Current edge = predicted_PRA - betting_line (simple difference)")
print("   → Does NOT account for prediction uncertainty")
print("   → Does NOT account for betting odds/vig properly")
print()
print("2. Theoretical edge = P(win) - breakeven_probability")
print("   → Accounts for prediction uncertainty (normal distribution)")
print("   → Accounts for actual betting odds")
print()
print("3. Major discrepancies suggest:")
print("   - Close calls (pred ≈ line) get assigned as positive edge")
print("   - But these have ~50% win probability due to uncertainty")
print("   - After accounting for vig (52.4% breakeven), these are NEGATIVE edge")
print()
print("RECOMMENDATION:")
print("  Rebuild edge calculation using probability-based approach")
print("  Edge = P(actual beats line) - implied_probability_from_odds")
print("  Where P(actual beats line) accounts for prediction variance")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
