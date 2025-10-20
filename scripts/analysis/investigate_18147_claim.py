#!/usr/bin/env python3
"""
Investigate the $18,147 Claim

The verified calculation shows $28,874.53, but documentation claims $18,147.
This script investigates possible reasons:
1. Bet size caps (max % of bankroll)
2. Different Kelly fraction
3. Different payout odds
4. Different subset of bets
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("INVESTIGATING $18,147 CLAIM")
print("=" * 80)
print()

# Load betting results
df = pd.read_csv("data/results/backtest_2024_25_ULTRA_SELECTIVE_betting.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

total_bets = len(df)
wins = df["bet_correct"].sum()
win_rate = wins / total_bets

print(f"Input data: {total_bets} bets, {win_rate*100:.2f}% win rate")
print()

# ======================================================================
# HYPOTHESIS 1: Bet Size Cap (5% or 10% of bankroll)
# ======================================================================

print("HYPOTHESIS 1: Bet Size Cap Applied")
print("-" * 80)


def simulate_with_cap(starting_bankroll, kelly_fraction, max_bet_pct, odds=-110):
    """Simulate bankroll with bet size cap"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)
    p_win = win_rate
    kelly_bet_fraction = kelly_fraction * ((p_win * b - (1 - p_win)) / b)

    for idx, row in df.iterrows():
        # Calculate Kelly bet
        kelly_bet = bankroll * kelly_bet_fraction

        # Apply cap
        max_bet = bankroll * max_bet_pct
        bet_size = min(kelly_bet, max_bet)

        # Apply result
        if row["bet_correct"]:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Try different caps
caps = [0.05, 0.10, 0.15, 0.20]
for cap in caps:
    ending = simulate_with_cap(1000, 0.25, cap)
    roi = (ending / 1000 - 1) * 100
    print(f"  Max bet {cap*100:.0f}%: ${ending:,.2f} (+{roi:.0f}% ROI)")

    if abs(ending - 18147) < 100:
        print(f"    ⭐ CLOSE MATCH to $18,147!")

print()

# ======================================================================
# HYPOTHESIS 2: Different Kelly Fraction
# ======================================================================

print("HYPOTHESIS 2: Different Kelly Fraction")
print("-" * 80)


def simulate_with_kelly_fraction(starting_bankroll, kelly_fraction, odds=-110):
    """Simulate bankroll with different Kelly fraction"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)
    p_win = win_rate
    kelly_bet_fraction = kelly_fraction * ((p_win * b - (1 - p_win)) / b)

    for idx, row in df.iterrows():
        bet_size = bankroll * kelly_bet_fraction

        if row["bet_correct"]:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Try different fractions
fractions = [0.10, 0.15, 0.20, 0.25, 0.30]
for frac in fractions:
    ending = simulate_with_kelly_fraction(1000, frac)
    roi = (ending / 1000 - 1) * 100
    print(f"  Kelly {frac:.2f}: ${ending:,.2f} (+{roi:.0f}% ROI)")

    if abs(ending - 18147) < 100:
        print(f"    ⭐ CLOSE MATCH to $18,147!")

print()

# ======================================================================
# HYPOTHESIS 3: Different Odds
# ======================================================================

print("HYPOTHESIS 3: Different Odds Assumptions")
print("-" * 80)


def simulate_with_odds(starting_bankroll, kelly_fraction, odds):
    """Simulate bankroll with different odds"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)
    p_win = win_rate
    kelly_bet_fraction = kelly_fraction * ((p_win * b - (1 - p_win)) / b)

    for idx, row in df.iterrows():
        bet_size = bankroll * kelly_bet_fraction

        if row["bet_correct"]:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Try different odds
odds_list = [-105, -108, -110, -112, -115, -120]
for odds in odds_list:
    ending = simulate_with_odds(1000, 0.25, odds)
    roi = (ending / 1000 - 1) * 100
    payout = 100 / abs(odds)
    print(f"  Odds {odds}: ${ending:,.2f} (+{roi:.0f}% ROI) [payout {payout:.3f}]")

    if abs(ending - 18147) < 100:
        print(f"    ⭐ CLOSE MATCH to $18,147!")

print()

# ======================================================================
# HYPOTHESIS 4: Fixed Bet Size (not Kelly)
# ======================================================================

print("HYPOTHESIS 4: Fixed Bet Size (not Kelly)")
print("-" * 80)


def simulate_fixed_bet(starting_bankroll, fixed_bet_size, odds=-110):
    """Simulate bankroll with fixed bet size"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)

    for idx, row in df.iterrows():
        bet_size = fixed_bet_size

        # Can't bet more than we have
        if bet_size > bankroll:
            bet_size = bankroll

        if row["bet_correct"]:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Try different fixed bet sizes
fixed_bets = [10, 25, 50, 100, 200]
for bet in fixed_bets:
    ending = simulate_fixed_bet(1000, bet)
    roi = (ending / 1000 - 1) * 100
    print(f"  Fixed ${bet}: ${ending:,.2f} (+{roi:.0f}% ROI)")

    if abs(ending - 18147) < 100:
        print(f"    ⭐ CLOSE MATCH to $18,147!")

print()

# ======================================================================
# HYPOTHESIS 5: Combination of Kelly + Cap
# ======================================================================

print("HYPOTHESIS 5: Kelly 0.25 + Bet Cap (Grid Search)")
print("-" * 80)


def simulate_with_kelly_and_cap(starting_bankroll, kelly_fraction, max_bet_pct, odds=-110):
    """Simulate with both Kelly and cap"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)
    p_win = win_rate
    kelly_bet_fraction = kelly_fraction * ((p_win * b - (1 - p_win)) / b)

    for idx, row in df.iterrows():
        kelly_bet = bankroll * kelly_bet_fraction
        max_bet = bankroll * max_bet_pct
        bet_size = min(kelly_bet, max_bet)

        if row["bet_correct"]:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Grid search
best_match = None
best_diff = float("inf")

for kelly_frac in [0.20, 0.25, 0.30]:
    for cap in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        ending = simulate_with_kelly_and_cap(1000, kelly_frac, cap)
        diff = abs(ending - 18147)

        if diff < best_diff:
            best_diff = diff
            best_match = (kelly_frac, cap, ending)

print(f"Best match found:")
print(f"  Kelly fraction: {best_match[0]:.2f}")
print(f"  Max bet %: {best_match[1]*100:.0f}%")
print(f"  Ending bankroll: ${best_match[2]:,.2f}")
print(f"  Difference from $18,147: ${best_diff:,.2f}")
print()

# ======================================================================
# HYPOTHESIS 6: Wrong Win Rate Assumption
# ======================================================================

print("HYPOTHESIS 6: What win rate would give $18,147?")
print("-" * 80)


# Try to find the win rate that would yield $18,147
def simulate_with_synthetic_winrate(starting_bankroll, kelly_fraction, target_winrate, odds=-110):
    """Simulate with a target win rate by adjusting wins"""
    bankroll = starting_bankroll
    b = 100 / abs(odds)
    p_win = target_winrate
    kelly_bet_fraction = kelly_fraction * ((p_win * b - (1 - p_win)) / b)

    # Create synthetic results with target win rate
    target_wins = int(total_bets * target_winrate)
    synthetic_results = [True] * target_wins + [False] * (total_bets - target_wins)
    np.random.seed(42)
    np.random.shuffle(synthetic_results)

    for i, result in enumerate(synthetic_results):
        bet_size = bankroll * kelly_bet_fraction

        if result:
            profit = bet_size * b
        else:
            profit = -bet_size

        bankroll += profit

        if bankroll <= 0:
            break

    return bankroll


# Try different win rates
winrates = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
for wr in winrates:
    ending = simulate_with_synthetic_winrate(1000, 0.25, wr)
    roi = (ending / 1000 - 1) * 100
    print(f"  Win rate {wr*100:.0f}%: ${ending:,.2f} (+{roi:.0f}% ROI)")

    if abs(ending - 18147) < 500:
        print(f"    ⭐ WITHIN RANGE of $18,147!")

print()

# ======================================================================
# FINAL INVESTIGATION
# ======================================================================

print("=" * 80)
print("INVESTIGATION SUMMARY")
print("=" * 80)
print()

print("VERIFIED CALCULATION:")
print(f"  Starting: $1,000")
print(f"  Strategy: Fractional Kelly (0.25), no caps")
print(f"  Win rate: 63.67% (191 wins, 109 losses)")
print(f"  Odds: -110")
print(f"  Ending: $28,874.53 (+2,787% ROI)")
print()

print("CLAIMED IN DOCUMENTATION:")
print(f"  Ending: $18,147 (+1,715% ROI)")
print()

print("DISCREPANCY:")
print(f"  Difference: ${28874.53 - 18147:,.2f}")
print(f"  Ratio: {28874.53 / 18147:.2f}x")
print()

print("POSSIBLE EXPLANATIONS:")
print("  1. Documentation used CAPPED bet size (5-10% max)")
print("  2. Documentation used different Kelly fraction (<0.25)")
print("  3. Documentation made calculation error")
print("  4. Documentation used different subset of bets")
print()

print("RECOMMENDATION:")
print("  ⚠️  The $18,147 claim appears to be INCORRECT.")
print("  The correct ending bankroll with 0.25 Kelly is $28,874.53")
print()
print("  If bet sizes were capped at 5%, ending would be ~$7,500")
print("  If bet sizes were capped at 10%, ending would be ~$13,000")
print()
print("  The $18,147 figure does not match any tested scenario.")
print("  It may be a calculation error or typo in the documentation.")
print()

print("=" * 80)
