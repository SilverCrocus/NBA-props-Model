"""
Deep Analysis - Investigating the model predictions
"""

import pandas as pd
import numpy as np

# Load data
betting_df = pd.read_csv('data/results/backtest_2024_25_FIXED_V2_betting.csv')

print("="*80)
print("DEEP DIVE ANALYSIS")
print("="*80)

print("\n1. PREDICTION DISTRIBUTION")
print("-"*80)
print(f"Average Predicted PRA: {betting_df['predicted_PRA'].mean():.2f}")
print(f"Average Actual PRA: {betting_df['actual_PRA'].mean():.2f}")
print(f"Average Line: {betting_df['line'].mean():.2f}")
print(f"\nPredicted PRA Range: {betting_df['predicted_PRA'].min():.2f} to {betting_df['predicted_PRA'].max():.2f}")
print(f"Actual PRA Range: {betting_df['actual_PRA'].min():.2f} to {betting_df['actual_PRA'].max():.2f}")

print("\n2. BET CLASSIFICATION")
print("-"*80)
print(f"Total bets: {len(betting_df)}")
print(f"OVER bets: {(betting_df['bet_type'] == 'OVER').sum()}")
print(f"UNDER bets: {(betting_df['bet_type'] == 'UNDER').sum()}")

print("\n3. CRITICAL ISSUE INVESTIGATION")
print("-"*80)
print("The model is predicting LOW values (avg: 10.2) vs HIGH actuals (avg: 16.2)")
print("This means:")
print("  - If betting UNDER, we'd expect HIGH win rate (actuals coming in OVER predictions)")
print("  - But actuals are MUCH HIGHER than predictions")
print("\nLet's check the betting logic...")

# Sample analysis
sample = betting_df.head(20)
print("\nFirst 20 bets breakdown:")
print(sample[['PLAYER_NAME', 'predicted_PRA', 'line', 'actual_PRA', 'edge', 'bet_type', 'bet_correct']].to_string())

print("\n4. EDGE CALCULATION VERIFICATION")
print("-"*80)
# Edge should be: predicted - line
# If predicted < line by 5+, we bet UNDER
# If predicted > line by 5+, we bet OVER

betting_df['calculated_edge'] = betting_df['predicted_PRA'] - betting_df['line']
betting_df['edge_match'] = np.isclose(betting_df['calculated_edge'], betting_df['edge'], atol=0.01)

print(f"Edge calculations match: {betting_df['edge_match'].sum()} / {len(betting_df)}")
print(f"\nSample edge verification:")
print(betting_df[['predicted_PRA', 'line', 'edge', 'calculated_edge', 'bet_type']].head(10))

print("\n5. BET OUTCOME LOGIC VERIFICATION")
print("-"*80)
# If bet UNDER and actual < line, should WIN
# If bet UNDER and actual >= line, should LOSE
# If bet OVER and actual > line, should WIN
# If bet OVER and actual < line, should LOSE

betting_df['calculated_outcome'] = False
betting_df.loc[(betting_df['bet_type'] == 'UNDER') & (betting_df['actual_PRA'] < betting_df['line']), 'calculated_outcome'] = True
betting_df.loc[(betting_df['bet_type'] == 'OVER') & (betting_df['actual_PRA'] > betting_df['line']), 'calculated_outcome'] = True

betting_df['outcome_match'] = betting_df['calculated_outcome'] == betting_df['bet_correct']
print(f"Outcome calculations match: {betting_df['outcome_match'].sum()} / {len(betting_df)}")

mismatches = betting_df[~betting_df['outcome_match']]
if len(mismatches) > 0:
    print(f"\nMismatched outcomes found: {len(mismatches)}")
    print(mismatches[['predicted_PRA', 'line', 'actual_PRA', 'bet_type', 'bet_correct', 'calculated_outcome']].head())

print("\n6. WHY IS THE MODEL FAILING?")
print("-"*80)

# The model predicts LOW (avg 10.2), lines are avg 16.2, actuals are avg 16.2
# So model says UNDER, but actuals come in right at the line
# This is a SEVERE UNDERPREDICTION PROBLEM

print("DIAGNOSIS:")
print(f"  Predicted PRA avg: {betting_df['predicted_PRA'].mean():.2f}")
print(f"  Line avg: {betting_df['line'].mean():.2f}")
print(f"  Actual PRA avg: {betting_df['actual_PRA'].mean():.2f}")
print(f"  Average Error: {betting_df['error'].mean():.2f}")
print(f"\nThe model is underpredicting by an average of {betting_df['error'].mean():.2f} points!")
print(f"This is EXACTLY why edge calculations are misleading.")

print("\n7. EDGE SIZE VS ERROR ANALYSIS")
print("-"*80)
edge_bins = pd.cut(betting_df['abs_edge'], bins=[5, 5.5, 6, 6.5, 7, 10], labels=['5.0-5.5', '5.5-6.0', '6.0-6.5', '6.5-7.0', '7.0+'])
analysis = betting_df.groupby(edge_bins, observed=True).agg({
    'bet_correct': ['count', 'sum', 'mean'],
    'error': 'mean',
    'abs_edge': 'mean'
})
print(analysis)

print("\n8. SYSTEMATIC BIAS CHECK")
print("-"*80)
# Check if the model is consistently underpredicting
under_predictions = (betting_df['error'] > 0).sum()
over_predictions = (betting_df['error'] < 0).sum()
print(f"Underpredictions (actual > predicted): {under_predictions} ({under_predictions/len(betting_df)*100:.1f}%)")
print(f"Overpredictions (actual < predicted): {over_predictions} ({over_predictions/len(betting_df)*100:.1f}%)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The model has a SYSTEMATIC UNDERPREDICTION BIAS.")
print("It predicts values ~6 points lower than actuals on average.")
print("This is why betting UNDER (when model predicts low) loses money.")
print("The 'edge' is an illusion - the model is miscalibrated, not finding value.")
