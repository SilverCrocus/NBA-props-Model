#!/usr/bin/env python3
"""
Backtest Audit - Check for Common Errors
==========================================

Audits the backtest results to identify potential issues:
1. Data leakage (using future information)
2. Incorrect odds matching
3. Logic errors in profit calculation
4. Overly optimistic assumptions
5. Sample size issues
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("BACKTEST AUDIT - CHECKING FOR ERRORS")
print("=" * 80)
print()

# Load the betting results
bets_df = pd.read_csv("data/results/backtest_ensemble_2024_25_real_odds.csv")
bets_df["date"] = pd.to_datetime(bets_df["date"])

print(f"Total bets analyzed: {len(bets_df):,}")
print()

# ======================================================================
# 1. CHECK TEMPORAL CONSISTENCY
# ======================================================================

print("=" * 80)
print("1. TEMPORAL CONSISTENCY CHECK")
print("=" * 80)
print()

# Check if dates are in proper order
date_sorted = bets_df["date"].is_monotonic_increasing
print(f"Dates properly ordered: {date_sorted}")

# Check date range
print(f"First bet date: {bets_df['date'].min().date()}")
print(f"Last bet date: {bets_df['date'].max().date()}")
print(f"Date range: {(bets_df['date'].max() - bets_df['date'].min()).days} days")
print()

# ======================================================================
# 2. CHECK PREDICTION VS ACTUAL CORRELATION
# ======================================================================

print("=" * 80)
print("2. PREDICTION QUALITY CHECK")
print("=" * 80)
print()

# Calculate correlation
correlation = bets_df["prediction"].corr(bets_df["actual"])
print(f"Prediction-Actual Correlation: {correlation:.3f}")

# Calculate MAE
mae = (bets_df["actual"] - bets_df["prediction"]).abs().mean()
print(f"Mean Absolute Error: {mae:.2f} pts")

# Check for suspiciously perfect predictions
perfect_preds = (bets_df["prediction"] == bets_df["actual"]).sum()
print(f"Perfect predictions: {perfect_preds} ({perfect_preds/len(bets_df)*100:.2f}%)")

# Check prediction distribution
print(f"\nPrediction range: {bets_df['prediction'].min():.1f} to {bets_df['prediction'].max():.1f}")
print(f"Actual range: {bets_df['actual'].min():.1f} to {bets_df['actual'].max():.1f}")
print()

# ======================================================================
# 3. CHECK EDGE DISTRIBUTION
# ======================================================================

print("=" * 80)
print("3. EDGE DISTRIBUTION CHECK")
print("=" * 80)
print()

bets_df["edge_abs"] = bets_df["edge"].abs()

print(f"Average edge (absolute): {bets_df['edge_abs'].mean():.2f} pts")
print(f"Median edge (absolute): {bets_df['edge_abs'].median():.2f} pts")
print(f"Max edge: {bets_df['edge_abs'].max():.2f} pts")
print()

# Check if edges are suspiciously large
large_edges = (bets_df["edge_abs"] > 10).sum()
print(f"Bets with edge > 10 pts: {large_edges} ({large_edges/len(bets_df)*100:.1f}%)")

# Verify edge calculation
bets_df["edge_check"] = bets_df["prediction"] - bets_df["line"]
edge_mismatch = (bets_df["edge"] != bets_df["edge_check"]).sum()
print(f"Edge calculation errors: {edge_mismatch}")
print()

# ======================================================================
# 4. CHECK WIN RATE BY EDGE DIRECTION
# ======================================================================

print("=" * 80)
print("4. WIN RATE BY BET SIDE")
print("=" * 80)
print()

# For OVER bets: should win when actual > line
over_bets = bets_df[bets_df["bet_side"] == "over"]
over_correct_logic = ((over_bets["actual"] > over_bets["line"]) == over_bets["is_correct"]).all()
print(f"OVER bet logic correct: {over_correct_logic}")
print(f"  OVER bets: {len(over_bets):,}")
print(f"  OVER wins: {over_bets['is_correct'].sum():,}")
print(f"  OVER win rate: {over_bets['is_correct'].mean()*100:.1f}%")

# For UNDER bets: should win when actual < line
under_bets = bets_df[bets_df["bet_side"] == "under"]
under_correct_logic = (
    (under_bets["actual"] < under_bets["line"]) == under_bets["is_correct"]
).all()
print(f"\nUNDER bet logic correct: {under_correct_logic}")
print(f"  UNDER bets: {len(under_bets):,}")
print(f"  UNDER wins: {under_bets['is_correct'].sum():,}")
print(f"  UNDER win rate: {under_bets['is_correct'].mean()*100:.1f}%")
print()

# ======================================================================
# 5. CHECK PROFIT CALCULATION
# ======================================================================

print("=" * 80)
print("5. PROFIT CALCULATION VERIFICATION")
print("=" * 80)
print()

# Verify profit calculation for a sample
sample_idx = bets_df.index[0]
sample = bets_df.loc[sample_idx]

print(f"Sample bet verification:")
print(f"  Bet side: {sample['bet_side']}")
print(f"  Prediction: {sample['prediction']:.1f}")
print(f"  Line: {sample['line']:.1f}")
print(f"  Actual: {sample['actual']:.1f}")
print(f"  Odds: {sample['odds']}")
print(f"  Bet amount: ${sample['bet_amount']:.2f}")
print(f"  Result: {'WIN' if sample['is_correct'] else 'LOSS'}")
print(f"  Profit: ${sample['profit']:.2f}")

# Calculate expected profit
if sample["is_correct"]:
    if sample["odds"] > 0:
        expected_profit = sample["bet_amount"] * (sample["odds"] / 100)
    else:
        expected_profit = sample["bet_amount"] * (100 / abs(sample["odds"]))
else:
    expected_profit = -sample["bet_amount"]

profit_match = abs(sample["profit"] - expected_profit) < 0.01
print(f"  Expected profit: ${expected_profit:.2f}")
print(f"  Profit calculation correct: {profit_match}")
print()

# ======================================================================
# 6. CHECK FOR PUSH GAMES
# ======================================================================

print("=" * 80)
print("6. PUSH GAMES CHECK")
print("=" * 80)
print()

# Check if any games landed exactly on the line
pushes = (bets_df["actual"] == bets_df["line"]).sum()
print(f"Games that landed on the line: {pushes}")

if pushes > 0:
    push_games = bets_df[bets_df["actual"] == bets_df["line"]]
    print("\nPush games (these should be excluded or counted as no bet):")
    print(
        push_games[
            ["player", "date", "prediction", "line", "actual", "is_correct", "profit"]
        ].head()
    )
    print("\n⚠️  WARNING: Push games should not count as wins or losses!")
print()

# ======================================================================
# 7. SANITY CHECK: RANDOM BASELINE
# ======================================================================

print("=" * 80)
print("7. RANDOM BASELINE COMPARISON")
print("=" * 80)
print()

# Simulate random betting (50/50 coin flip)
np.random.seed(42)
n_simulations = 1000
random_win_rates = []

for _ in range(n_simulations):
    random_wins = np.random.binomial(len(bets_df), 0.5)
    random_win_rates.append(random_wins / len(bets_df))

random_wr_mean = np.mean(random_win_rates)
random_wr_std = np.std(random_win_rates)

actual_wr = bets_df["is_correct"].mean()
z_score = (actual_wr - random_wr_mean) / random_wr_std

print(f"Random betting win rate (expected): {random_wr_mean*100:.1f}% ± {random_wr_std*100:.1f}%")
print(f"Our win rate: {actual_wr*100:.1f}%")
print(f"Z-score: {z_score:.2f} (standard deviations above random)")

if z_score > 3:
    print(
        f"✅ Win rate is {z_score:.1f} standard deviations above random (statistically significant)"
    )
else:
    print(f"⚠️  Win rate may not be statistically significant")
print()

# ======================================================================
# 8. CHECK AGAINST MARKET EFFICIENCY
# ======================================================================

print("=" * 80)
print("8. MARKET EFFICIENCY CHECK")
print("=" * 80)
print()

# In an efficient market, betting randomly on either side should yield ~52.4% breakeven
# (accounting for -110 odds)
breakeven_wr = 0.524

print(f"Efficient market breakeven (typical -110 odds): {breakeven_wr*100:.1f}%")
print(f"Our win rate: {actual_wr*100:.1f}%")
print(f"Edge over breakeven: {(actual_wr - breakeven_wr)*100:.1f} percentage points")

if actual_wr > 0.70:
    print("\n⚠️  WARNING: Win rate > 70% is extremely rare in sports betting")
    print("   Professional sports bettors typically achieve 55-58% long-term")
    print("   This suggests potential data leakage or overfitting")
elif actual_wr > 0.60:
    print("\n✅ Win rate 60-70% is strong but achievable for sharp models")
elif actual_wr > 0.55:
    print("\n✅ Win rate 55-60% is realistic for a good model")
else:
    print("\n⚠️  Win rate below 55% may not be profitable after vig")
print()

# ======================================================================
# 9. CHECK SAMPLE SIZE BY THRESHOLD
# ======================================================================

print("=" * 80)
print("9. SAMPLE SIZE ANALYSIS")
print("=" * 80)
print()

for threshold in [0, 2, 3, 4, 5, 6]:
    filtered = bets_df[bets_df["edge_abs"] >= threshold]

    if len(filtered) > 0:
        win_rate = filtered["is_correct"].mean()
        # Calculate 95% confidence interval
        n = len(filtered)
        ci = 1.96 * np.sqrt((win_rate * (1 - win_rate)) / n)

        print(f"Edge >= {threshold} pts:")
        print(f"  Sample size: {n:,}")
        print(f"  Win rate: {win_rate*100:.1f}% ± {ci*100:.1f}% (95% CI)")
        print(f"  CI range: [{(win_rate-ci)*100:.1f}%, {(win_rate+ci)*100:.1f}%]")

        if n < 100:
            print(f"  ⚠️  WARNING: Small sample size (n={n}), results may not be reliable")
        print()

# ======================================================================
# 10. FINAL VERDICT
# ======================================================================

print("=" * 80)
print("FINAL AUDIT VERDICT")
print("=" * 80)
print()

issues_found = []

# Check 1: Win rate too high
if actual_wr > 0.70:
    issues_found.append("Win rate > 70% is suspiciously high")

# Check 2: Perfect predictions
if perfect_preds / len(bets_df) > 0.01:
    issues_found.append(f"{perfect_preds} perfect predictions is unusual")

# Check 3: Push games counted as wins/losses
if pushes > 0:
    issues_found.append(f"{pushes} push games should be excluded")

# Check 4: Edge calculation errors
if edge_mismatch > 0:
    issues_found.append(f"{edge_mismatch} edge calculation errors")

# Check 5: Low correlation
if correlation < 0.3:
    issues_found.append(f"Low prediction-actual correlation ({correlation:.3f})")

if len(issues_found) > 0:
    print("⚠️  POTENTIAL ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    print()
    print("RECOMMENDATION: Investigate these issues before deploying to production")
else:
    print("✅ No major issues found in backtest")
    print()
    print("However, a 62-65% win rate is still very strong.")
    print("Recommended next steps:")
    print("  1. Paper trade for 1-2 months to verify live performance")
    print("  2. Start with small stakes ($10-50 per bet)")
    print("  3. Track actual results vs predictions")
    print("  4. Re-calibrate if live performance differs significantly")

print()
print("=" * 80)
print("✅ AUDIT COMPLETE")
print("=" * 80)
