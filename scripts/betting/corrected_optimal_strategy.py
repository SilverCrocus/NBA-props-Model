#!/usr/bin/env python3
"""
CORRECTED Optimal Betting Strategy Analysis
============================================

Re-analyzes betting strategy with CORRECT understanding:
- Small edges (0-3 pts) have LOW win rates (~53%)
- Medium edges (3-6 pts) have GOOD win rates (~61%)
- Large edges (6+ pts) have GREAT win rates (~67-72%)

This finds the optimal balance between edge threshold and volume.

Usage: uv run python scripts/betting/corrected_optimal_strategy.py
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("CORRECTED OPTIMAL BETTING STRATEGY ANALYSIS")
print("=" * 80)
print()

# Load verification results
predictions_df = pd.read_csv("data/results/predictions_with_edge_analysis.csv")
predictions_df["GAME_DATE"] = pd.to_datetime(predictions_df["GAME_DATE"])

print(f"Loaded {len(predictions_df):,} predictions with edge analysis")
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
# TEST REALISTIC EDGE THRESHOLDS
# ======================================================================

print("Testing edge thresholds with Kelly Criterion betting...")
print()

KELLY_FRACTION = 0.25
STARTING_BANKROLL = 10000
MAX_BET_PCT = 0.05

# Test these thresholds
edge_thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

results = []

for edge_threshold in edge_thresholds:
    bankroll = STARTING_BANKROLL
    bet_results = []

    for _, row in predictions_df.iterrows():
        edge = row["edge"]

        # Skip if below threshold
        if abs(edge) < edge_threshold:
            continue

        # Determine bet side
        if edge > 0:
            bet_side = "over"
            odds = row["over_price"]
            is_correct = row["over_wins"]
        else:
            bet_side = "under"
            odds = row["under_price"]
            is_correct = row["under_wins"]

        # Calculate Kelly bet size
        decimal_odds = american_to_decimal(odds)
        edge_magnitude = abs(edge)
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
                "edge": edge,
                "edge_abs": abs(edge),
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
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    roi_on_bankroll = (total_profit / STARTING_BANKROLL) * 100
    final_bankroll = bankroll

    # Calculate average edge
    avg_edge = bets_df["edge_abs"].mean()

    results.append(
        {
            "edge_threshold": edge_threshold,
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": win_rate,
            "avg_edge": avg_edge,
            "total_wagered": total_wagered,
            "total_profit": total_profit,
            "roi": roi,
            "roi_on_bankroll": roi_on_bankroll,
            "final_bankroll": final_bankroll,
            "over_bets": (bets_df["bet_side"] == "over").sum(),
            "under_bets": (bets_df["bet_side"] == "under").sum(),
        }
    )

results_df = pd.DataFrame(results)

# ======================================================================
# DISPLAY RESULTS
# ======================================================================

print("=" * 80)
print("BETTING SIMULATION RESULTS BY EDGE THRESHOLD")
print("=" * 80)
print()

print(
    f"{'Threshold':<12} {'Bets':<8} {'Win Rate':<12} {'Avg Edge':<12} {'ROI':<12} {'Profit':<15} {'Final $':<12}"
)
print("-" * 90)
for _, row in results_df.iterrows():
    print(
        f"{row['edge_threshold']:>3.0f} pts     "
        f"{row['total_bets']:>5.0f}    "
        f"{row['win_rate']*100:>5.1f}%       "
        f"{row['avg_edge']:>5.1f} pts    "
        f"{row['roi']:>6.1f}%      "
        f"${row['total_profit']:>8,.0f}       "
        f"${row['final_bankroll']:>8,.0f}"
    )
print()

# ======================================================================
# FIND OPTIMAL STRATEGIES
# ======================================================================

print("=" * 80)
print("OPTIMAL STRATEGIES (CORRECTED)")
print("=" * 80)
print()

# Best profit
best_profit_idx = results_df["total_profit"].idxmax()
best_profit = results_df.loc[best_profit_idx]

print("1. MAXIMUM PROFIT:")
print(f"   Edge threshold: {best_profit['edge_threshold']:.0f} pts")
print(f"   Total bets: {best_profit['total_bets']:.0f}")
print(f"   Win rate: {best_profit['win_rate']*100:.1f}%")
print(f"   Average edge: {best_profit['avg_edge']:.1f} pts")
print(f"   ROI on wagered: {best_profit['roi']:.1f}%")
print(f"   Final bankroll: ${best_profit['final_bankroll']:,.0f}")
print(
    f"   Total profit: ${best_profit['total_profit']:,.0f} ({best_profit['roi_on_bankroll']:.1f}%)"
)
print()

# Best ROI
best_roi_idx = results_df["roi"].idxmax()
best_roi = results_df.loc[best_roi_idx]

print("2. MAXIMUM ROI:")
print(f"   Edge threshold: {best_roi['edge_threshold']:.0f} pts")
print(f"   Total bets: {best_roi['total_bets']:.0f}")
print(f"   Win rate: {best_roi['win_rate']*100:.1f}%")
print(f"   Average edge: {best_roi['avg_edge']:.1f} pts")
print(f"   ROI on wagered: {best_roi['roi']:.1f}%")
print(f"   Final bankroll: ${best_roi['final_bankroll']:,.0f}")
print(f"   Total profit: ${best_roi['total_profit']:,.0f} ({best_roi['roi_on_bankroll']:.1f}%)")
print()

# Best win rate (with minimum 50 bets)
min_bets = 50
high_volume = results_df[results_df["total_bets"] >= min_bets]
if len(high_volume) > 0:
    best_winrate_idx = high_volume["win_rate"].idxmax()
    best_winrate = results_df.loc[best_winrate_idx]

    print(f"3. HIGHEST WIN RATE (min {min_bets} bets):")
    print(f"   Edge threshold: {best_winrate['edge_threshold']:.0f} pts")
    print(f"   Total bets: {best_winrate['total_bets']:.0f}")
    print(f"   Win rate: {best_winrate['win_rate']*100:.1f}%")
    print(f"   Average edge: {best_winrate['avg_edge']:.1f} pts")
    print(f"   ROI on wagered: {best_winrate['roi']:.1f}%")
    print(f"   Final bankroll: ${best_winrate['final_bankroll']:,.0f}")
    print(
        f"   Total profit: ${best_winrate['total_profit']:,.0f} ({best_winrate['roi_on_bankroll']:.1f}%)"
    )
    print()

# Most balanced (Sharpe-like metric: ROI / sqrt(variance_proxy))
# Use bets as variance proxy (more bets = more variance)
results_df["sharpe_proxy"] = results_df["roi_on_bankroll"] / np.sqrt(results_df["total_bets"])
best_sharpe_idx = results_df["sharpe_proxy"].idxmax()
best_sharpe = results_df.loc[best_sharpe_idx]

print("4. BEST RISK-ADJUSTED RETURN:")
print(f"   Edge threshold: {best_sharpe['edge_threshold']:.0f} pts")
print(f"   Total bets: {best_sharpe['total_bets']:.0f}")
print(f"   Win rate: {best_sharpe['win_rate']*100:.1f}%")
print(f"   Average edge: {best_sharpe['avg_edge']:.1f} pts")
print(f"   ROI on wagered: {best_sharpe['roi']:.1f}%")
print(f"   Final bankroll: ${best_sharpe['final_bankroll']:,.0f}")
print(
    f"   Total profit: ${best_sharpe['total_profit']:,.0f} ({best_sharpe['roi_on_bankroll']:.1f}%)"
)
print()

# ======================================================================
# RECOMMENDATIONS
# ======================================================================

print("=" * 80)
print("CORRECTED RECOMMENDATIONS")
print("=" * 80)
print()

print("KEY INSIGHT: Small edges (0-3 pts) have only 53.6% win rate!")
print("This is barely above the 52.4% breakeven at -110 odds.")
print()

print("RECOMMENDED STRATEGIES:")
print()

print("🥇 AGGRESSIVE (Maximum Profit):")
print(f"   → Use {best_profit['edge_threshold']:.0f} pts threshold")
print(
    f"   → Expected: {best_profit['total_bets']:.0f} bets, {best_profit['win_rate']*100:.1f}% win rate, ${best_profit['total_profit']:,.0f} profit"
)
print()

print("🥈 CONSERVATIVE (High Win Rate):")
print(f"   → Use {best_winrate['edge_threshold']:.0f} pts threshold")
print(
    f"   → Expected: {best_winrate['total_bets']:.0f} bets, {best_winrate['win_rate']*100:.1f}% win rate, ${best_winrate['total_profit']:,.0f} profit"
)
print()

print("🥉 RISK-ADJUSTED (Best Sharpe):")
print(f"   → Use {best_sharpe['edge_threshold']:.0f} pts threshold")
print(
    f"   → Expected: {best_sharpe['total_bets']:.0f} bets, {best_sharpe['win_rate']*100:.1f}% win rate, ${best_sharpe['total_profit']:,.0f} profit"
)
print()

print("Strategy Parameters (keep same):")
print(f"  - Kelly fraction: {KELLY_FRACTION*100:.0f}%")
print(f"  - Max bet: {MAX_BET_PCT*100:.0f}% of bankroll")
print()

# Save results
results_df.to_csv("data/results/corrected_edge_strategy.csv", index=False)
print("✅ Saved corrected analysis to data/results/corrected_edge_strategy.csv")
print()

print("=" * 80)
print("✅ CORRECTED ANALYSIS COMPLETE")
print("=" * 80)
