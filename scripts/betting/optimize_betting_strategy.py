#!/usr/bin/env python3
"""
Optimal Betting Strategy Analysis
==================================

Analyzes the no-filter backtest results to find optimal betting strategies:
- Best edge thresholds
- Bet sizing optimization
- Risk-adjusted returns
- Drawdown analysis
- Over vs Under performance

Input: data/results/backtest_ensemble_2024_25_real_odds.csv
Output: Comprehensive strategy recommendations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 80)
print("OPTIMAL BETTING STRATEGY ANALYSIS")
print("=" * 80)
print()

# Load backtest results
bets_df = pd.read_csv("data/results/backtest_ensemble_2024_25_real_odds.csv")
print(f"✅ Loaded {len(bets_df):,} bets from backtest")
print()

# Convert date column
bets_df["date"] = pd.to_datetime(bets_df["date"])

# ======================================================================
# 1. ANALYZE PERFORMANCE BY EDGE SIZE
# ======================================================================

print("=" * 80)
print("1. PERFORMANCE BY EDGE SIZE")
print("=" * 80)
print()

# Create edge bins
edge_bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 20]
bets_df["edge_abs"] = bets_df["edge"].abs()
bets_df["edge_bin"] = pd.cut(bets_df["edge_abs"], bins=edge_bins)

edge_analysis = (
    bets_df.groupby("edge_bin")
    .agg({"is_correct": ["sum", "count", "mean"], "profit": "sum", "bet_amount": "sum"})
    .reset_index()
)

edge_analysis.columns = ["edge_bin", "wins", "bets", "win_rate", "profit", "wagered"]
edge_analysis["roi"] = (edge_analysis["profit"] / edge_analysis["wagered"]) * 100
edge_analysis["avg_bet"] = edge_analysis["wagered"] / edge_analysis["bets"]

print("Win Rate and ROI by Edge Size:")
print(edge_analysis.to_string(index=False))
print()

# Find optimal edge threshold
print("Optimal Edge Thresholds:")
for threshold in [0, 1, 2, 3, 4, 5, 6]:
    filtered = bets_df[bets_df["edge_abs"] >= threshold]
    if len(filtered) > 0:
        win_rate = filtered["is_correct"].mean()
        roi = (filtered["profit"].sum() / filtered["bet_amount"].sum()) * 100
        total_profit = filtered["profit"].sum()
        print(
            f"  Edge >= {threshold} pts: {len(filtered):4d} bets | {win_rate*100:5.1f}% WR | {roi:6.2f}% ROI | ${total_profit:8,.0f} profit"
        )
print()

# ======================================================================
# 2. ANALYZE PERFORMANCE BY PREDICTION CONFIDENCE
# ======================================================================

print("=" * 80)
print("2. PERFORMANCE BY PREDICTION CONFIDENCE")
print("=" * 80)
print()

# Prediction confidence = how far prediction is from line
conf_bins = [0, 2, 4, 6, 8, 10, 15, 20, 50]
bets_df["conf_bin"] = pd.cut(bets_df["edge_abs"], bins=conf_bins)

conf_analysis = (
    bets_df.groupby("conf_bin")
    .agg({"is_correct": ["sum", "count", "mean"], "profit": "sum", "bet_amount": "sum"})
    .reset_index()
)

conf_analysis.columns = ["conf_bin", "wins", "bets", "win_rate", "profit", "wagered"]
conf_analysis["roi"] = (conf_analysis["profit"] / conf_analysis["wagered"]) * 100

print("Win Rate and ROI by Prediction Confidence:")
print(conf_analysis.to_string(index=False))
print()

# ======================================================================
# 3. OVER VS UNDER PERFORMANCE
# ======================================================================

print("=" * 80)
print("3. OVER VS UNDER PERFORMANCE BY EDGE SIZE")
print("=" * 80)
print()

for bet_type in ["over", "under"]:
    print(f"\n{bet_type.upper()} BETS:")
    print("-" * 40)

    type_df = bets_df[bets_df["bet_side"] == bet_type]

    for threshold in [0, 2, 3, 4, 5]:
        filtered = type_df[type_df["edge_abs"] >= threshold]
        if len(filtered) > 0:
            win_rate = filtered["is_correct"].mean()
            roi = (filtered["profit"].sum() / filtered["bet_amount"].sum()) * 100
            total_profit = filtered["profit"].sum()
            print(
                f"  Edge >= {threshold} pts: {len(filtered):4d} bets | {win_rate*100:5.1f}% WR | {roi:6.2f}% ROI | ${total_profit:8,.0f} profit"
            )

print()

# ======================================================================
# 4. DRAWDOWN ANALYSIS
# ======================================================================

print("=" * 80)
print("4. DRAWDOWN ANALYSIS")
print("=" * 80)
print()

# Sort by date
bets_df = bets_df.sort_values("date")

# Calculate cumulative profit
bets_df["cumulative_profit"] = bets_df["profit"].cumsum()

# Calculate running max
bets_df["running_max"] = bets_df["cumulative_profit"].cummax()

# Calculate drawdown
bets_df["drawdown"] = bets_df["running_max"] - bets_df["cumulative_profit"]
bets_df["drawdown_pct"] = (bets_df["drawdown"] / (10000 + bets_df["running_max"])) * 100

max_drawdown = bets_df["drawdown"].max()
max_drawdown_pct = bets_df["drawdown_pct"].max()
max_drawdown_date = bets_df.loc[bets_df["drawdown"].idxmax(), "date"]

print(f"Maximum Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
print(f"Occurred on: {max_drawdown_date.date()}")
print()

# Calculate Sharpe ratio (approximate)
daily_returns = bets_df.groupby(bets_df["date"].dt.date)["profit"].sum()
sharpe_ratio = (
    (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else 0
)
print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
print()

# ======================================================================
# 5. STREAK ANALYSIS
# ======================================================================

print("=" * 80)
print("5. WINNING AND LOSING STREAKS")
print("=" * 80)
print()

# Calculate streaks
bets_df["streak_id"] = (bets_df["is_correct"] != bets_df["is_correct"].shift()).cumsum()
streaks = bets_df.groupby("streak_id").agg({"is_correct": ["first", "count"]}).reset_index()
streaks.columns = ["streak_id", "is_win", "length"]

win_streaks = streaks[streaks["is_win"] == True]["length"]
loss_streaks = streaks[streaks["is_win"] == False]["length"]

print(f"Longest winning streak: {win_streaks.max() if len(win_streaks) > 0 else 0} bets")
print(f"Longest losing streak: {loss_streaks.max() if len(loss_streaks) > 0 else 0} bets")
print(f"Average winning streak: {win_streaks.mean():.1f} bets")
print(f"Average losing streak: {loss_streaks.mean():.1f} bets")
print()

# ======================================================================
# 6. MONTHLY CONSISTENCY
# ======================================================================

print("=" * 80)
print("6. MONTHLY CONSISTENCY")
print("=" * 80)
print()

bets_df["month"] = bets_df["date"].dt.to_period("M")
monthly_stats = (
    bets_df.groupby("month")
    .agg({"profit": "sum", "is_correct": ["sum", "count", "mean"], "bet_amount": "sum"})
    .reset_index()
)

monthly_stats.columns = ["month", "profit", "wins", "bets", "win_rate", "wagered"]
monthly_stats["roi"] = (monthly_stats["profit"] / monthly_stats["wagered"]) * 100

profitable_months = (monthly_stats["profit"] > 0).sum()
total_months = len(monthly_stats)

print(
    f"Profitable months: {profitable_months}/{total_months} ({profitable_months/total_months*100:.1f}%)"
)
print(f"Average monthly profit: ${monthly_stats['profit'].mean():,.2f}")
print(
    f"Best month: {monthly_stats.loc[monthly_stats['profit'].idxmax(), 'month']} (${monthly_stats['profit'].max():,.2f})"
)
print(
    f"Worst month: {monthly_stats.loc[monthly_stats['profit'].idxmin(), 'month']} (${monthly_stats['profit'].min():,.2f})"
)
print()

# ======================================================================
# 7. RECOMMENDED STRATEGIES
# ======================================================================

print("=" * 80)
print("7. RECOMMENDED BETTING STRATEGIES")
print("=" * 80)
print()

strategies = [
    {
        "name": "Conservative (Low Risk)",
        "edge_threshold": 4,
        "max_bet_pct": 0.03,
        "kelly_fraction": 0.20,
    },
    {
        "name": "Balanced (Medium Risk)",
        "edge_threshold": 3,
        "max_bet_pct": 0.05,
        "kelly_fraction": 0.25,
    },
    {
        "name": "Aggressive (High Risk)",
        "edge_threshold": 2,
        "max_bet_pct": 0.07,
        "kelly_fraction": 0.30,
    },
    {
        "name": "Maximum Volume",
        "edge_threshold": 0,
        "max_bet_pct": 0.05,
        "kelly_fraction": 0.25,
    },
]

for strategy in strategies:
    filtered = bets_df[bets_df["edge_abs"] >= strategy["edge_threshold"]]

    if len(filtered) > 0:
        win_rate = filtered["is_correct"].mean()
        total_profit = filtered["profit"].sum()
        roi = (total_profit / filtered["bet_amount"].sum()) * 100

        print(f"\n{strategy['name']}:")
        print(f"  Settings:")
        print(f"    - Edge threshold: {strategy['edge_threshold']} pts")
        print(f"    - Max bet size: {strategy['max_bet_pct']*100:.0f}% of bankroll")
        print(f"    - Kelly fraction: {strategy['kelly_fraction']*100:.0f}%")
        print(f"  Results:")
        print(f"    - Bets: {len(filtered):,}")
        print(f"    - Win rate: {win_rate*100:.1f}%")
        print(f"    - ROI: {roi:.2f}%")
        print(f"    - Total profit: ${total_profit:,.2f}")

print()
print()

# ======================================================================
# 8. FINAL RECOMMENDATIONS
# ======================================================================

print("=" * 80)
print("FINAL RECOMMENDATIONS FOR PRODUCTION")
print("=" * 80)
print()

# Find sweet spot
optimal_threshold = None
best_sharpe = -999

for threshold in [0, 1, 2, 3, 4, 5]:
    filtered = bets_df[bets_df["edge_abs"] >= threshold]
    if len(filtered) > 20:  # Need minimum sample
        win_rate = filtered["is_correct"].mean()
        roi = (filtered["profit"].sum() / filtered["bet_amount"].sum()) * 100
        daily_returns = filtered.groupby(filtered["date"].dt.date)["profit"].sum()
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() > 0
            else 0
        )

        if sharpe > best_sharpe and win_rate > 0.55:  # Must beat 55% to account for vig
            best_sharpe = sharpe
            optimal_threshold = threshold

if optimal_threshold is not None:
    optimal_bets = bets_df[bets_df["edge_abs"] >= optimal_threshold]
    optimal_win_rate = optimal_bets["is_correct"].mean()
    optimal_roi = (optimal_bets["profit"].sum() / optimal_bets["bet_amount"].sum()) * 100

    print("✅ RECOMMENDED PRODUCTION STRATEGY:")
    print()
    print(f"1. Edge Threshold: {optimal_threshold} points")
    print(f"   - Only bet when |prediction - line| >= {optimal_threshold}")
    print()
    print(f"2. Bet Sizing: Quarter Kelly (25% Kelly)")
    print(f"   - Max bet size: 5% of bankroll")
    print()
    print(f"3. Expected Results:")
    print(f"   - Volume: ~{len(optimal_bets)/7:.0f} bets per month")
    print(f"   - Win Rate: {optimal_win_rate*100:.1f}%")
    print(f"   - ROI: {optimal_roi:.2f}%")
    print(f"   - Risk-adjusted return (Sharpe): {best_sharpe:.2f}")
    print()
    print(f"4. Risk Management:")
    print(f"   - Max drawdown observed: {max_drawdown_pct:.2f}%")
    print(f"   - Recommended bankroll: $5,000 minimum")
    print(f"   - Stop trading if drawdown > 30%")
    print()

print("=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
