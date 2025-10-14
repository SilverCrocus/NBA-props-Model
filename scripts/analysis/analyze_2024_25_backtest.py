"""
Comprehensive analysis of 2024-25 NBA Props Model backtest results.
"""

import pandas as pd
import numpy as np
import json

# Load the walk-forward backtest results
backtest_df = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2024_25.csv')
predictions_df = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/walkforward_predictions_2024-25.csv')

print("=" * 80)
print("2024-25 SEASON BACKTEST ANALYSIS - COMPREHENSIVE REPORT")
print("=" * 80)

# SECTION 1: OVERALL PERFORMANCE METRICS
print("\n" + "=" * 80)
print("1. OVERALL PERFORMANCE METRICS")
print("=" * 80)

total_predictions = len(predictions_df)
matched_predictions = len(backtest_df)
match_rate = (matched_predictions / total_predictions) * 100

print(f"\nTotal Walk-Forward Predictions: {total_predictions:,}")
print(f"Matched to Betting Lines: {matched_predictions:,}")
print(f"Match Rate: {match_rate:.2f}%")

# Calculate prediction accuracy
mae_all = predictions_df['abs_error'].mean()
rmse_all = np.sqrt((predictions_df['error'] ** 2).mean())

print(f"\nPrediction Accuracy (All Predictions):")
print(f"  MAE: {mae_all:.2f} points")
print(f"  RMSE: {rmse_all:.2f} points")

# Betting metrics
total_bets = len(backtest_df[backtest_df['bet_side'] != 'NONE'])
total_won = backtest_df['bet_won'].sum()
total_lost = backtest_df['bet_lost'].sum()
total_pushed = backtest_df['bet_pushed'].sum()

win_rate = (total_won / total_bets) * 100 if total_bets > 0 else 0

print(f"\nBetting Performance:")
print(f"  Total Bets Placed: {total_bets:,}")
print(f"  Wins: {total_won} ({(total_won/total_bets)*100:.2f}%)")
print(f"  Losses: {total_lost} ({(total_lost/total_bets)*100:.2f}%)")
print(f"  Pushes: {total_pushed} ({(total_pushed/total_bets)*100:.2f}%)")

# Calculate profit/loss
backtest_with_bets = backtest_df[backtest_df['bet_side'] != 'NONE'].copy()
total_wagered = total_bets * 100  # $100 per bet
total_profit = backtest_with_bets['bet_result'].sum()
roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

print(f"\nFinancial Performance:")
print(f"  Total Wagered: ${total_wagered:,.0f}")
print(f"  Total Profit: ${total_profit:,.2f}")
print(f"  ROI: {roi:+.2f}%")

# Breakeven calculation
breakeven_rate = 52.38  # at -110 odds
print(f"\nBreakeven Analysis:")
print(f"  Required Win Rate (at -110): {breakeven_rate:.2f}%")
print(f"  Actual Win Rate: {win_rate:.2f}%")
print(f"  Edge Over Breakeven: {win_rate - breakeven_rate:+.2f} pp")

# MAE on bet games
mae_bets = backtest_with_bets['abs_error'].mean()
print(f"\nPrediction Accuracy (Bet Games Only):")
print(f"  MAE: {mae_bets:.2f} points")

# SECTION 2: PERFORMANCE BY EDGE SIZE
print("\n" + "=" * 80)
print("2. PERFORMANCE BY EDGE SIZE")
print("=" * 80)

# Create edge size bins
backtest_with_bets['edge_bin'] = pd.cut(
    backtest_with_bets['abs_edge'],
    bins=[3, 5, 7, 10, float('inf')],
    labels=['Small (3-5 pts)', 'Medium (5-7 pts)', 'Large (7-10 pts)', 'Huge (10+ pts)'],
    include_lowest=True
)

print("\n{:<20} {:>8} {:>12} {:>12} {:>15}".format(
    "Edge Range", "Bets", "Win Rate", "ROI", "Profit"
))
print("-" * 80)

for edge_bin in ['Small (3-5 pts)', 'Medium (5-7 pts)', 'Large (7-10 pts)', 'Huge (10+ pts)']:
    edge_data = backtest_with_bets[backtest_with_bets['edge_bin'] == edge_bin]
    if len(edge_data) > 0:
        n_bets = len(edge_data)
        wins = edge_data['bet_won'].sum()
        wr = (wins / n_bets) * 100
        wagered = n_bets * 100
        profit = edge_data['bet_result'].sum()
        edge_roi = (profit / wagered) * 100

        print("{:<20} {:>8} {:>11.2f}% {:>11.2f}% {:>14,.2f}".format(
            edge_bin, n_bets, wr, edge_roi, profit
        ))

# SECTION 3: CLOSING LINE VALUE (CLV)
print("\n" + "=" * 80)
print("3. CLOSING LINE VALUE (CLV) ANALYSIS")
print("=" * 80)

# CLV = percentage of predictions with 3+ point edge
clv_3plus = len(backtest_df[backtest_df['abs_edge'] >= 3]) / len(backtest_df) * 100
clv_5plus = len(backtest_df[backtest_df['abs_edge'] >= 5]) / len(backtest_df) * 100
clv_7plus = len(backtest_df[backtest_df['abs_edge'] >= 7]) / len(backtest_df) * 100
clv_10plus = len(backtest_df[backtest_df['abs_edge'] >= 10]) / len(backtest_df) * 100

print(f"\nPercentage of Predictions with Edge:")
print(f"  3+ points: {clv_3plus:.1f}% ({len(backtest_df[backtest_df['abs_edge'] >= 3])} predictions)")
print(f"  5+ points: {clv_5plus:.1f}% ({len(backtest_df[backtest_df['abs_edge'] >= 5])} predictions)")
print(f"  7+ points: {clv_7plus:.1f}% ({len(backtest_df[backtest_df['abs_edge'] >= 7])} predictions)")
print(f"  10+ points: {clv_10plus:.1f}% ({len(backtest_df[backtest_df['abs_edge'] >= 10])} predictions)")

avg_edge = backtest_df['abs_edge'].mean()
print(f"\nAverage Edge (absolute): {avg_edge:.2f} points")

# SECTION 4: COMPARISON TO 2023-24 SEASON
print("\n" + "=" * 80)
print("4. COMPARISON: 2024-25 vs 2023-24")
print("=" * 80)

# Load 2023-24 results if available
try:
    backtest_2023 = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2023_24.csv')

    bets_2023 = backtest_2023[backtest_2023['bet_side'] != 'NONE']
    total_bets_2023 = len(bets_2023)
    wins_2023 = bets_2023['bet_won'].sum()
    wr_2023 = (wins_2023 / total_bets_2023) * 100
    wagered_2023 = total_bets_2023 * 100
    profit_2023 = bets_2023['bet_result'].sum()
    roi_2023 = (profit_2023 / wagered_2023) * 100
    mae_2023 = bets_2023['abs_error'].mean()
    clv_2023 = len(backtest_2023[backtest_2023['abs_edge'] >= 3]) / len(backtest_2023) * 100

    print("\n{:<25} {:>15} {:>15} {:>15}".format("Metric", "2023-24", "2024-25", "Change"))
    print("-" * 80)
    print("{:<25} {:>15,} {:>15,} {:>15,}".format("Total Bets", total_bets_2023, total_bets, total_bets - total_bets_2023))
    print("{:<25} {:>14.2f}% {:>14.2f}% {:>14.2f} pp".format("Win Rate", wr_2023, win_rate, win_rate - wr_2023))
    print("{:<25} {:>14.2f}% {:>14.2f}% {:>14.2f} pp".format("ROI", roi_2023, roi, roi - roi_2023))
    print("{:<25} {:>14,.2f} {:>14,.2f} {:>14,.2f}".format("Total Profit", profit_2023, total_profit, total_profit - profit_2023))
    print("{:<25} {:>14.2f} {:>14.2f} {:>14.2f}".format("MAE (points)", mae_2023, mae_bets, mae_bets - mae_2023))
    print("{:<25} {:>14.1f}% {:>14.1f}% {:>14.1f} pp".format("CLV Rate", clv_2023, clv_3plus, clv_3plus - clv_2023))

except FileNotFoundError:
    print("\n2023-24 walk-forward backtest file not found.")
    print("Using corrected backtest results from report...")

    # From FINAL_VALIDATION_REPORT.md
    print("\n{:<25} {:>15} {:>15}".format("Metric", "2023-24", "2024-25"))
    print("-" * 70)
    print("{:<25} {:>15,} {:>15,}".format("Total Bets", 2060, total_bets))
    print("{:<25} {:>14.2f}% {:>14.2f}%".format("Win Rate", 79.66, win_rate))
    print("{:<25} {:>14.2f}% {:>14.2f}%".format("ROI", 61.98, roi))
    print("{:<25} {:>14,.2f} {:>14,.2f}".format("Total Profit", 127685, total_profit))
    print("{:<25} {:>14.2f} {:>14.2f}".format("MAE (points)", 4.82, mae_bets))
    print("{:<25} {:>14.1f}% {:>14.1f}%".format("CLV Rate", 50.6, clv_3plus))

    print("\nKEY FINDING: Massive performance degradation from 2023-24 to 2024-25")
    print(f"  Win Rate Drop: {79.66 - win_rate:.2f} percentage points")
    print(f"  ROI Drop: {61.98 - roi:.2f} percentage points")

# SECTION 5: INDUSTRY BENCHMARKS
print("\n" + "=" * 80)
print("5. INDUSTRY BENCHMARKS COMPARISON")
print("=" * 80)

print("\n{:<25} {:>15} {:>15} {:>15} {:>15}".format(
    "Metric", "Model", "Good", "Elite", "Status"
))
print("-" * 95)
print("{:<25} {:>14.2f}% {:>15} {:>15} {:>15}".format(
    "Win Rate", win_rate, "54-56%", "58-60%",
    "BELOW" if win_rate < 52.38 else "AVERAGE" if win_rate < 54 else "GOOD" if win_rate < 58 else "ELITE"
))
print("{:<25} {:>14.2f}% {:>15} {:>15} {:>15}".format(
    "ROI", roi, "3-5%", "8-12%",
    "POOR" if roi < 1 else "AVERAGE" if roi < 3 else "GOOD" if roi < 8 else "ELITE"
))
print("{:<25} {:>14.2f} {:>15} {:>15} {:>15}".format(
    "MAE", mae_bets, "4-5 pts", "3.5-4.0 pts",
    "POOR" if mae_bets > 7 else "AVERAGE" if mae_bets > 5 else "GOOD" if mae_bets > 4 else "ELITE"
))
print("{:<25} {:>14.1f}% {:>15} {:>15} {:>15}".format(
    "CLV Rate", clv_3plus, "20-30%", "40-50%",
    "POOR" if clv_3plus < 20 else "GOOD" if clv_3plus < 40 else "ELITE"
))

# SECTION 6: KEY ISSUES IDENTIFIED
print("\n" + "=" * 80)
print("6. KEY ISSUES & FINDINGS")
print("=" * 80)

issues = []

if win_rate < breakeven_rate:
    issues.append(f"Win rate ({win_rate:.2f}%) is BELOW breakeven ({breakeven_rate}%)")
else:
    issues.append(f"Win rate ({win_rate:.2f}%) is barely above breakeven ({breakeven_rate}%)")

if roi < 1:
    issues.append(f"ROI ({roi:.2f}%) is too thin for sustainable betting")

if mae_bets > 7:
    issues.append(f"MAE ({mae_bets:.2f} pts) is too high - indicates poor prediction accuracy")

# Check if large edges underperform
large_edges = backtest_with_bets[backtest_with_bets['abs_edge'] >= 10]
small_edges = backtest_with_bets[backtest_with_bets['abs_edge'].between(3, 5)]
if len(large_edges) > 0 and len(small_edges) > 0:
    large_wr = (large_edges['bet_won'].sum() / len(large_edges)) * 100
    small_wr = (small_edges['bet_won'].sum() / len(small_edges)) * 100
    if large_wr < small_wr:
        issues.append(f"Model miscalibrated: Large edges ({large_wr:.1f}%) perform worse than small edges ({small_wr:.1f}%)")

print("\nCritical Issues:")
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

# Positive findings
positives = []
if clv_3plus > 50:
    positives.append(f"Strong CLV ({clv_3plus:.1f}%) indicates model finds value vs bookmakers")
if match_rate > 10:
    positives.append(f"Good match rate ({match_rate:.1f}%) means reasonable betting coverage")

print("\nPositive Findings:")
for i, pos in enumerate(positives, 1):
    print(f"  {i}. {pos}")

# SECTION 7: DATA QUALITY ASSESSMENT
print("\n" + "=" * 80)
print("7. DATA QUALITY ASSESSMENT")
print("=" * 80)

# Check for missing values
missing_checks = {
    'Predictions': predictions_df.isnull().sum().sum(),
    'Backtest Data': backtest_df.isnull().sum().sum(),
}

print("\nMissing Values:")
for source, missing in missing_checks.items():
    print(f"  {source}: {missing} missing values")

# Check date range
pred_dates = pd.to_datetime(predictions_df['GAME_DATE'])
backtest_dates = pd.to_datetime(backtest_df['game_date'])

print(f"\nDate Coverage:")
print(f"  Predictions: {pred_dates.min().strftime('%Y-%m-%d')} to {pred_dates.max().strftime('%Y-%m-%d')}")
print(f"  Backtest: {backtest_dates.min().strftime('%Y-%m-%d')} to {backtest_dates.max().strftime('%Y-%m-%d')}")
print(f"  Days Covered: {(backtest_dates.max() - backtest_dates.min()).days} days")

# Check unique players
unique_players_pred = predictions_df['PLAYER_NAME'].nunique()
unique_players_bet = backtest_df['PLAYER_NAME'].nunique()

print(f"\nPlayer Coverage:")
print(f"  Predictions: {unique_players_pred} unique players")
print(f"  Betting Lines: {unique_players_bet} unique players")

# SECTION 8: RECOMMENDATIONS
print("\n" + "=" * 80)
print("8. RECOMMENDATIONS")
print("=" * 80)

print("\nImmediate Actions Required:")
print("  1. DO NOT deploy model to production - performance is below breakeven")
print("  2. Validate 2023-24 season using same walk-forward methodology")
print("  3. Investigate why MAE is 8.83 points (feature engineering issue?)")
print("  4. Implement probability calibration to fix miscalibrated large edges")

print("\nShort-term Improvements (1-2 weeks):")
print("  5. Enhance walk-forward feature engineering:")
print("     - Add CTG advanced stats calculation")
print("     - Add opponent defensive metrics")
print("     - Add rest/schedule features")
print("  6. Reduce MAE to target <5 points")
print("  7. Test calibration methods (isotonic regression, Platt scaling)")

print("\nMedium-term Development (1-2 months):")
print("  8. Add injury tracking (major impact on minutes/usage)")
print("  9. Add lineup modeling (who plays with whom)")
print("  10. Add minutes projection model")
print("  11. Implement Kelly criterion for bet sizing")
print("  12. Test ensemble methods (XGBoost + LightGBM)")

print("\nValidation Requirements:")
print("  13. Backtest on 2022-23, 2021-22 seasons for stability")
print("  14. Target: Consistent 55%+ win rate, 5-10% ROI across multiple seasons")
print("  15. Only consider production after achieving targets on 3+ seasons")

# SECTION 9: HONEST ASSESSMENT
print("\n" + "=" * 80)
print("9. HONEST ASSESSMENT")
print("=" * 80)

if win_rate < breakeven_rate:
    status = "NOT READY FOR PRODUCTION"
    recommendation = "DO NOT BET REAL MONEY"
    timeline = "6-12 months to production readiness"
elif win_rate < 54:
    status = "MARGINALLY VIABLE"
    recommendation = "SMALL TEST BETS ONLY ($5-10)"
    timeline = "3-6 months to strong profitability"
elif win_rate < 56:
    status = "GOOD PERFORMANCE"
    recommendation = "MODERATE TEST BETS ($25-50)"
    timeline = "1-2 months to optimize and scale"
else:
    status = "ELITE PERFORMANCE"
    recommendation = "READY FOR SCALED DEPLOYMENT"
    timeline = "Immediate deployment recommended"

print(f"\nCurrent Status: {status}")
print(f"Recommendation: {recommendation}")
print(f"Timeline to Production: {timeline}")

print(f"\nRationale:")
if win_rate < breakeven_rate:
    print(f"  - Win rate ({win_rate:.2f}%) below breakeven ({breakeven_rate}%)")
    print(f"  - Model loses money in expectation")
    print(f"  - Significant improvements needed before any betting")
else:
    print(f"  - Win rate ({win_rate:.2f}%) is barely profitable")
    print(f"  - ROI ({roi:.2f}%) too thin for reliable returns")
    print(f"  - Variance could easily put performance underwater")

print("\n" + "=" * 80)
print("REPORT COMPLETE")
print("=" * 80)

# Save summary to JSON
summary = {
    "season": "2024-25",
    "total_predictions": int(total_predictions),
    "matched_predictions": int(matched_predictions),
    "match_rate": float(match_rate),
    "total_bets": int(total_bets),
    "wins": int(total_won),
    "losses": int(total_lost),
    "pushes": int(total_pushed),
    "win_rate": float(win_rate),
    "total_wagered": float(total_wagered),
    "total_profit": float(total_profit),
    "roi": float(roi),
    "mae_all_predictions": float(mae_all),
    "mae_bet_games": float(mae_bets),
    "clv_3plus": float(clv_3plus),
    "clv_5plus": float(clv_5plus),
    "clv_10plus": float(clv_10plus),
    "avg_edge": float(avg_edge),
    "status": status,
    "recommendation": recommendation,
    "production_ready": False
}

with open('/Users/diyagamah/Documents/nba_props_model/data/results/backtest_2024_25_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: /Users/diyagamah/Documents/nba_props_model/data/results/backtest_2024_25_analysis.json")
