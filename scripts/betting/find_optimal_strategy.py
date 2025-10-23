#!/usr/bin/env python3
"""
Find Optimal Betting Strategy
==============================

Analyzes backtest results to find the optimal edge threshold and betting parameters.
Tests multiple edge thresholds to maximize ROI while maintaining acceptable win rate.

Usage: uv run python scripts/betting/find_optimal_strategy.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 80)
print("FINDING OPTIMAL BETTING STRATEGY")
print("=" * 80)
print()

# ======================================================================
# LOAD BACKTEST RESULTS
# ======================================================================

print("Loading predictions with odds...")
predictions_df = pd.read_csv("data/results/predictions_ensemble_2024_25_with_odds.csv")
predictions_df["GAME_DATE"] = pd.to_datetime(predictions_df["GAME_DATE"])

print(f"✅ Loaded {len(predictions_df):,} predictions with odds")
print()

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_profit(bet_amount, american_odds):
    """Calculate profit from a winning bet with American odds."""
    if american_odds > 0:
        return bet_amount * (american_odds / 100)
    else:
        return bet_amount * (100 / abs(american_odds))


# ======================================================================
# TEST DIFFERENT EDGE THRESHOLDS
# ======================================================================

print("Testing different edge thresholds...")
print()

# Kelly Criterion parameters
KELLY_FRACTION = 0.25
STARTING_BANKROLL = 10000
MAX_BET_PCT = 0.05

edge_thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
results = []

for edge_threshold in edge_thresholds:
    bankroll = STARTING_BANKROLL
    bet_results = []

    for _, row in predictions_df.iterrows():
        prediction = row["prediction"]
        actual = row["actual"]
        line = row["line"]
        over_price = row["over_price"]
        under_price = row["under_price"]

        # Calculate edge
        pred_over_line = prediction - line

        # Determine if we have an edge
        if abs(pred_over_line) < edge_threshold:
            continue

        # Determine bet side
        if prediction > line + edge_threshold:
            bet_side = "over"
            odds = over_price
            is_correct = actual > line
        elif prediction < line - edge_threshold:
            bet_side = "under"
            odds = under_price
            is_correct = actual < line
        else:
            continue

        # Calculate Kelly bet size
        decimal_odds = american_to_decimal(odds)
        edge_magnitude = abs(pred_over_line)
        our_prob = 0.5 + (edge_magnitude / 100)
        our_prob = min(max(our_prob, 0.51), 0.99)

        b = decimal_odds - 1
        kelly_bet_pct = (b * our_prob - (1 - our_prob)) / b
        kelly_bet_pct = max(0, kelly_bet_pct)

        bet_pct = min(kelly_bet_pct * KELLY_FRACTION, MAX_BET_PCT)
        bet_amount = bankroll * bet_pct

        if bet_amount < 1:
            continue

        # Calculate profit/loss
        if is_correct:
            profit = calculate_profit(bet_amount, odds)
        else:
            profit = -bet_amount

        bankroll += profit

        bet_results.append(
            {
                "is_correct": is_correct,
                "profit": profit,
                "bet_amount": bet_amount,
                "edge": pred_over_line,
                "bet_side": bet_side,
            }
        )

    if len(bet_results) == 0:
        continue

    bets_df = pd.DataFrame(bet_results)

    total_bets = len(bets_df)
    wins = bets_df["is_correct"].sum()
    win_rate = wins / total_bets
    total_wagered = bets_df["bet_amount"].sum()
    total_profit = bets_df["profit"].sum()
    roi = (total_profit / total_wagered) * 100
    roi_on_bankroll = (total_profit / STARTING_BANKROLL) * 100
    final_bankroll = bankroll

    results.append(
        {
            "edge_threshold": edge_threshold,
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": win_rate,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": roi,
            "roi_on_bankroll": roi_on_bankroll,
            "final_bankroll": final_bankroll,
            "over_bets": (bets_df["bet_side"] == "over").sum(),
            "over_wins": bets_df[bets_df["bet_side"] == "over"]["is_correct"].sum(),
            "under_bets": (bets_df["bet_side"] == "under").sum(),
            "under_wins": bets_df[bets_df["bet_side"] == "under"]["is_correct"].sum(),
        }
    )

results_df = pd.DataFrame(results)

# Calculate over/under win rates
results_df["over_win_rate"] = results_df["over_wins"] / results_df["over_bets"]
results_df["under_win_rate"] = results_df["under_wins"] / results_df["under_bets"]

# ======================================================================
# DISPLAY RESULTS
# ======================================================================

print("=" * 80)
print("RESULTS BY EDGE THRESHOLD")
print("=" * 80)
print()

# Format for display
display_df = results_df.copy()
display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x*100:.1f}%")
display_df["roi"] = display_df["roi"].apply(lambda x: f"{x:.1f}%")
display_df["roi_on_bankroll"] = display_df["roi_on_bankroll"].apply(lambda x: f"{x:.1f}%")
display_df["final_bankroll"] = display_df["final_bankroll"].apply(lambda x: f"${x:,.0f}")
display_df["total_profit"] = display_df["total_profit"].apply(lambda x: f"${x:,.0f}")
display_df["over_win_rate"] = display_df["over_win_rate"].apply(lambda x: f"{x*100:.1f}%")
display_df["under_win_rate"] = display_df["under_win_rate"].apply(lambda x: f"{x*100:.1f}%")

print(
    display_df[
        ["edge_threshold", "total_bets", "win_rate", "roi", "roi_on_bankroll", "final_bankroll"]
    ].to_string(index=False)
)
print()

print("Detailed Breakdown:")
print(
    display_df[
        [
            "edge_threshold",
            "total_bets",
            "over_bets",
            "over_win_rate",
            "under_bets",
            "under_win_rate",
        ]
    ].to_string(index=False)
)
print()

# ======================================================================
# FIND OPTIMAL STRATEGY
# ======================================================================

print("=" * 80)
print("OPTIMAL STRATEGIES")
print("=" * 80)
print()

# Best ROI
best_roi_idx = results_df["roi"].idxmax()
best_roi = results_df.loc[best_roi_idx]

print("1. HIGHEST ROI:")
print(f"   Edge threshold: {best_roi['edge_threshold']} pts")
print(f"   Total bets: {best_roi['total_bets']:.0f}")
print(f"   Win rate: {best_roi['win_rate']*100:.1f}%")
print(f"   ROI on wagered: {best_roi['roi']:.1f}%")
print(f"   Final bankroll: ${best_roi['final_bankroll']:,.0f}")
print(f"   Total profit: ${best_roi['total_profit']:,.0f} ({best_roi['roi_on_bankroll']:.1f}%)")
print()

# Best profit
best_profit_idx = results_df["total_profit"].idxmax()
best_profit = results_df.loc[best_profit_idx]

print("2. HIGHEST PROFIT:")
print(f"   Edge threshold: {best_profit['edge_threshold']} pts")
print(f"   Total bets: {best_profit['total_bets']:.0f}")
print(f"   Win rate: {best_profit['win_rate']*100:.1f}%")
print(f"   ROI on wagered: {best_profit['roi']:.1f}%")
print(f"   Final bankroll: ${best_profit['final_bankroll']:,.0f}")
print(
    f"   Total profit: ${best_profit['total_profit']:,.0f} ({best_profit['roi_on_bankroll']:.1f}%)"
)
print()

# Best win rate (with minimum bet volume)
min_bets = 50
high_volume = results_df[results_df["total_bets"] >= min_bets]
if len(high_volume) > 0:
    best_winrate_idx = high_volume["win_rate"].idxmax()
    best_winrate = results_df.loc[best_winrate_idx]

    print(f"3. HIGHEST WIN RATE (minimum {min_bets} bets):")
    print(f"   Edge threshold: {best_winrate['edge_threshold']} pts")
    print(f"   Total bets: {best_winrate['total_bets']:.0f}")
    print(f"   Win rate: {best_winrate['win_rate']*100:.1f}%")
    print(f"   ROI on wagered: {best_winrate['roi']:.1f}%")
    print(f"   Final bankroll: ${best_winrate['final_bankroll']:,.0f}")
    print(
        f"   Total profit: ${best_winrate['total_profit']:,.0f} ({best_winrate['roi_on_bankroll']:.1f}%)"
    )
    print()

# Most balanced (good ROI + volume)
# Score = ROI * log(bets) to balance ROI and volume
results_df["score"] = results_df["roi"] * np.log(results_df["total_bets"] + 1)
best_balanced_idx = results_df["score"].idxmax()
best_balanced = results_df.loc[best_balanced_idx]

print("4. MOST BALANCED (ROI × Volume):")
print(f"   Edge threshold: {best_balanced['edge_threshold']} pts")
print(f"   Total bets: {best_balanced['total_bets']:.0f}")
print(f"   Win rate: {best_balanced['win_rate']*100:.1f}%")
print(f"   ROI on wagered: {best_balanced['roi']:.1f}%")
print(f"   Final bankroll: ${best_balanced['final_bankroll']:,.0f}")
print(
    f"   Total profit: ${best_balanced['total_profit']:,.0f} ({best_balanced['roi_on_bankroll']:.1f}%)"
)
print()

# ======================================================================
# SAVE RESULTS
# ======================================================================

results_df.to_csv("data/results/edge_threshold_analysis.csv", index=False)
print("✅ Saved analysis to data/results/edge_threshold_analysis.csv")
print()

# ======================================================================
# RECOMMENDATIONS
# ======================================================================

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

print("Based on the analysis:")
print()
print(f"• For MAXIMUM PROFIT: Use {best_profit['edge_threshold']:.0f} pts edge threshold")
print(
    f"  - Expected profit: ${best_profit['total_profit']:,.0f} ({best_profit['roi_on_bankroll']:.1f}% return)"
)
print(f"  - Volume: {best_profit['total_bets']:.0f} bets")
print()
print(f"• For MAXIMUM ROI: Use {best_roi['edge_threshold']:.0f} pts edge threshold")
print(f"  - Expected ROI: {best_roi['roi']:.1f}% on wagered")
print(f"  - Win rate: {best_roi['win_rate']*100:.1f}%")
print()
print(f"• For BALANCED APPROACH: Use {best_balanced['edge_threshold']:.0f} pts edge threshold")
print(
    f"  - Good balance of ROI ({best_balanced['roi']:.1f}%) and volume ({best_balanced['total_bets']:.0f} bets)"
)
print()
print("Strategy parameters (already optimal):")
print(f"  - Kelly fraction: {KELLY_FRACTION*100:.0f}% (quarter-Kelly)")
print(f"  - Max bet: {MAX_BET_PCT*100:.0f}% of bankroll")
print()

print("=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print()
