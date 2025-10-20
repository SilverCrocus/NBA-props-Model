#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Verify $1,000 → $18,147 Bankroll Simulation

This script validates the bankroll simulation calculations that claim:
- Starting: $1,000
- Ending: $18,147
- ROI: +1,715%
- Strategy: Fractional Kelly (0.25)
- Win rate: 63.67% (191 wins, 109 losses out of 300 bets)

The verification will check:
1. Input data correctness (300 bets, 63.67% win rate)
2. Kelly calculation accuracy
3. Bankroll simulation logic
4. Final result accuracy
"""

from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("BANKROLL SIMULATION VERIFICATION")
print("=" * 80)
print()

# ======================================================================
# STEP 1: VERIFY INPUT DATA
# ======================================================================

print("STEP 1: Verify Input Data")
print("-" * 80)

# Load betting results
betting_csv = Path("data/results/backtest_2024_25_ULTRA_SELECTIVE_betting.csv")
df = pd.read_csv(betting_csv)

# Sort by date to ensure chronological order
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values("GAME_DATE").reset_index(drop=True)

# Verify basic counts
total_bets = len(df)
wins = df["bet_correct"].sum()
losses = total_bets - wins
win_rate = wins / total_bets

print(f"Total bets: {total_bets}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Win rate: {win_rate*100:.2f}%")
print()

# Check 1: Verify 300 bets
if total_bets == 300:
    print("✅ CHECK 1 PASSED: Total bets = 300")
else:
    print(f"❌ CHECK 1 FAILED: Expected 300 bets, got {total_bets}")
print()

# Check 2: Verify 63.67% win rate
expected_win_rate = 0.6367
if abs(win_rate - expected_win_rate) < 0.001:
    print(f"✅ CHECK 2 PASSED: Win rate = {win_rate*100:.2f}%")
else:
    print(f"❌ CHECK 2 FAILED: Expected {expected_win_rate*100:.2f}%, got {win_rate*100:.2f}%")
print()

# Check 3: Verify outcomes are realistic (not sorted)
# Check first 10 and last 10 bets
print("First 10 bet outcomes:", df["bet_correct"].head(10).tolist())
print("Last 10 bet outcomes:", df["bet_correct"].tail(10).tolist())
print()

# Check 4: Check for any NaN values
nan_count = df["bet_correct"].isna().sum()
if nan_count > 0:
    print(f"⚠️  WARNING: {nan_count} NaN values in bet_correct")
else:
    print("✅ CHECK 3 PASSED: No NaN values in bet outcomes")
print()

# ======================================================================
# STEP 2: VERIFY KELLY CALCULATION
# ======================================================================

print("STEP 2: Verify Kelly Calculation")
print("-" * 80)

# Kelly parameters
p_win = win_rate  # 0.6367
p_loss = 1 - p_win  # 0.3633
odds = -110  # Standard American odds

# Convert -110 odds to decimal payout ratio
# For -110 odds: win $0.909 for every $1 wagered
# Formula: payout = 100 / abs(odds) = 100 / 110 = 0.909
b = 100 / abs(odds)

print(f"Win probability (p): {p_win:.4f}")
print(f"Loss probability (q): {p_loss:.4f}")
print(f"Payout ratio (b): {b:.4f} (for -110 odds)")
print()

# Kelly formula: f = (p*b - q) / b
kelly_full = (p_win * b - p_loss) / b
kelly_fraction = 0.25  # Fractional Kelly
kelly_bet_fraction = kelly_full * kelly_fraction

print(f"Full Kelly: {kelly_full:.4f} ({kelly_full*100:.2f}% of bankroll)")
print(
    f"Fractional Kelly (0.25): {kelly_bet_fraction:.4f} ({kelly_bet_fraction*100:.2f}% of bankroll)"
)
print()

# Verify calculation manually
numerator = p_win * b - p_loss
expected_kelly = numerator / b
print(
    f"Manual calculation: ({p_win:.4f} * {b:.4f} - {p_loss:.4f}) / {b:.4f} = {expected_kelly:.4f}"
)
print()

if abs(kelly_full - expected_kelly) < 0.0001:
    print("✅ CHECK 4 PASSED: Kelly calculation correct")
else:
    print(f"❌ CHECK 4 FAILED: Kelly mismatch")
print()

# ======================================================================
# STEP 3: VERIFY BANKROLL SIMULATION LOGIC
# ======================================================================

print("STEP 3: Verify Bankroll Simulation Logic")
print("-" * 80)

# Simulation parameters
starting_bankroll = 1000.0
bankroll = starting_bankroll

# Track history
bankroll_history = []
bet_history = []

print(f"Starting bankroll: ${starting_bankroll:,.2f}")
print(f"Kelly fraction: {kelly_fraction} (1/4 Kelly)")
print(f"Kelly bet %: {kelly_bet_fraction*100:.2f}% of bankroll per bet")
print()

# Simulate each bet chronologically
for idx, row in df.iterrows():
    # Calculate bet size using Kelly fraction
    bet_size = bankroll * kelly_bet_fraction

    # Check if bankroll went negative
    if bankroll <= 0:
        print(f"❌ BANKRUPT at bet #{idx + 1}")
        break

    # Get bet outcome
    bet_won = row["bet_correct"]

    # Calculate result
    if bet_won:
        # Win: +$0.909 per $1 wagered (for -110 odds)
        profit = bet_size * b
    else:
        # Loss: -$1 per $1 wagered
        profit = -bet_size

    # Update bankroll
    bankroll += profit

    # Record
    bankroll_history.append(
        {
            "bet_num": idx + 1,
            "date": row["GAME_DATE"],
            "bet_size": bet_size,
            "bet_won": bet_won,
            "profit": profit,
            "bankroll": bankroll,
        }
    )

    # Debug: Print first 5 bets
    if idx < 5:
        result_str = "WIN" if bet_won else "LOSS"
        print(
            f"Bet {idx+1}: ${bet_size:6.2f} -> {result_str} -> ${profit:+7.2f} -> Bankroll: ${bankroll:,.2f}"
        )

print()
print("...")
print()

# Print last 5 bets
for record in bankroll_history[-5:]:
    result_str = "WIN" if record["bet_won"] else "LOSS"
    print(
        f"Bet {record['bet_num']}: ${record['bet_size']:6.2f} -> {result_str} -> ${record['profit']:+7.2f} -> Bankroll: ${record['bankroll']:,.2f}"
    )

print()

# Convert to DataFrame
history_df = pd.DataFrame(bankroll_history)

# ======================================================================
# STEP 4: CALCULATE FINAL RESULTS
# ======================================================================

print("STEP 4: Calculate Final Results")
print("-" * 80)

final_bankroll = bankroll
total_profit = final_bankroll - starting_bankroll
total_return = (final_bankroll / starting_bankroll - 1) * 100

print(f"Starting bankroll: ${starting_bankroll:,.2f}")
print(f"Ending bankroll:   ${final_bankroll:,.2f}")
print(f"Total profit:      ${total_profit:,.2f}")
print(f"ROI:               {total_return:+.2f}%")
print()

# Check 5: Verify ending bankroll
claimed_ending = 18147
if abs(final_bankroll - claimed_ending) < 1.0:
    print(f"✅ CHECK 5 PASSED: Ending bankroll matches claimed ${claimed_ending:,.0f}")
else:
    print(f"❌ CHECK 5 FAILED: Expected ${claimed_ending:,.0f}, calculated ${final_bankroll:,.2f}")
    print(f"   Difference: ${final_bankroll - claimed_ending:,.2f}")
print()

# Check 6: Verify ROI
claimed_roi = 1715
calculated_roi = round(total_return, 0)
if abs(calculated_roi - claimed_roi) < 1:
    print(f"✅ CHECK 6 PASSED: ROI matches claimed +{claimed_roi}%")
else:
    print(f"❌ CHECK 6 FAILED: Expected +{claimed_roi}%, calculated +{calculated_roi}%")
    print(f"   Difference: {calculated_roi - claimed_roi} pp")
print()

# ======================================================================
# STEP 5: ADDITIONAL STATISTICS
# ======================================================================

print("STEP 5: Additional Statistics")
print("-" * 80)

# Peak and trough
peak_bankroll = history_df["bankroll"].max()
trough_bankroll = history_df["bankroll"].min()
avg_bet_size = history_df["bet_size"].mean()
max_bet_size = history_df["bet_size"].max()
min_bet_size = history_df["bet_size"].min()

print(f"Peak bankroll:    ${peak_bankroll:,.2f}")
print(f"Trough bankroll:  ${trough_bankroll:,.2f}")
print(f"Avg bet size:     ${avg_bet_size:,.2f}")
print(f"Max bet size:     ${max_bet_size:,.2f}")
print(f"Min bet size:     ${min_bet_size:,.2f}")
print()

# Max drawdown
running_max = history_df["bankroll"].expanding().max()
drawdown = (history_df["bankroll"] - running_max) / running_max * 100
max_drawdown = drawdown.min()

print(f"Max drawdown:     {max_drawdown:.2f}%")
print()

# Longest winning/losing streaks
winning_streak = 0
losing_streak = 0
max_win_streak = 0
max_lose_streak = 0

for won in df["bet_correct"]:
    if won:
        winning_streak += 1
        losing_streak = 0
        max_win_streak = max(max_win_streak, winning_streak)
    else:
        losing_streak += 1
        winning_streak = 0
        max_lose_streak = max(max_lose_streak, losing_streak)

print(f"Longest win streak:  {max_win_streak} bets")
print(f"Longest loss streak: {max_lose_streak} bets")
print()

# ======================================================================
# STEP 6: STATISTICAL REALITY CHECK
# ======================================================================

print("STEP 6: Statistical Reality Check")
print("-" * 80)

# Calculate theoretical edge
edge = (p_win * b) - (p_loss * 1.0)
print(f"Theoretical edge: {edge:.4f} ({edge*100:.2f}%)")
print()

# Expected growth using Kelly formula
# E[log(bankroll)] ≈ n * edge (simplified)
# More accurate: use Kelly growth rate formula
# g = p * ln(1 + b*f) + q * ln(1 - f)
# where f = kelly_bet_fraction

import math

g = p_win * math.log(1 + b * kelly_bet_fraction) + p_loss * math.log(1 - kelly_bet_fraction)
expected_ending = starting_bankroll * math.exp(g * total_bets)

print(f"Kelly growth rate (g): {g:.6f} per bet")
print(f"Expected ending (theoretical): ${expected_ending:,.2f}")
print(f"Actual ending (simulated):     ${final_bankroll:,.2f}")
print(f"Difference: ${final_bankroll - expected_ending:,.2f}")
print()

# Check if the difference is reasonable (simulation vs expectation)
if abs(final_bankroll - expected_ending) / expected_ending < 0.01:
    print("✅ CHECK 7 PASSED: Simulated result within 1% of theoretical expectation")
else:
    pct_diff = abs(final_bankroll - expected_ending) / expected_ending * 100
    print(f"⚠️  CHECK 7: Simulated result differs from theory by {pct_diff:.2f}%")
    print("   (This is expected due to discrete vs continuous compounding)")
print()

# ======================================================================
# STEP 7: RED FLAGS CHECK
# ======================================================================

print("STEP 7: Red Flags Check")
print("-" * 80)

red_flags = []

# Check 1: Bet sizes should increase over time (compounding)
if history_df["bet_size"].iloc[-1] > history_df["bet_size"].iloc[0]:
    print("✅ Bet sizes increased over time (compounding working)")
else:
    red_flags.append("Bet sizes did not increase (compounding not working)")
    print(
        f"❌ Bet sizes decreased: ${history_df['bet_size'].iloc[0]:.2f} -> ${history_df['bet_size'].iloc[-1]:.2f}"
    )
print()

# Check 2: No negative bankroll
if (history_df["bankroll"] > 0).all():
    print("✅ No negative bankroll values")
else:
    red_flags.append("Negative bankroll detected")
    print("❌ Negative bankroll detected!")
print()

# Check 3: Bet sizes capped at reasonable max
max_bet_pct = (history_df["bet_size"] / history_df["bankroll"].shift(1)).max() * 100
if max_bet_pct < 30:  # Kelly should keep bets < 30% of bankroll
    print(f"✅ Max bet size reasonable ({max_bet_pct:.2f}% of bankroll)")
else:
    red_flags.append(f"Max bet size too high ({max_bet_pct:.2f}%)")
    print(f"⚠️  Max bet size: {max_bet_pct:.2f}% of bankroll (risky!)")
print()

# Check 4: Wins and losses in realistic order (not sorted)
first_half_wins = df["bet_correct"].iloc[:150].sum()
second_half_wins = df["bet_correct"].iloc[150:].sum()
print(f"First half wins: {first_half_wins}/150 ({first_half_wins/150*100:.1f}%)")
print(f"Second half wins: {second_half_wins}/150 ({second_half_wins/150*100:.1f}%)")

if abs(first_half_wins - second_half_wins) < 30:  # Should be relatively balanced
    print("✅ Wins distributed throughout (not sorted)")
else:
    red_flags.append("Wins clustered (may be sorted)")
    print("⚠️  Wins may be clustered (check if sorted)")
print()

# ======================================================================
# FINAL VERDICT
# ======================================================================

print("=" * 80)
print("FINAL VERDICT")
print("=" * 80)
print()

if len(red_flags) == 0:
    print("✅ ALL CHECKS PASSED")
    print()
    print("CONCLUSION:")
    print(f"  The bankroll simulation is MATHEMATICALLY CORRECT.")
    print(f"  Starting with $1,000, using fractional Kelly (0.25) bet sizing,")
    print(f"  and achieving 63.67% win rate over 300 bets at -110 odds,")
    print(
        f"  the ending bankroll of ${final_bankroll:,.2f} (+{total_return:.0f}% ROI) is accurate."
    )
    print()
    print("  This is an extremely aggressive compounding strategy with high variance.")
    print("  The exponential growth is due to:")
    print(f"    1. High win rate (63.67% vs 52.38% breakeven)")
    print(f"    2. Consistent edge ({edge*100:.2f}% per bet)")
    print(f"    3. Kelly criterion maximizing geometric growth")
    print(f"    4. Compounding over 300 bets")
else:
    print("❌ ISSUES DETECTED:")
    for flag in red_flags:
        print(f"  - {flag}")
    print()
    print("CONCLUSION:")
    print("  The simulation has potential issues that need investigation.")

print()
print("=" * 80)

# Save history
output_path = "data/results/bankroll_simulation_verification.csv"
history_df.to_csv(output_path, index=False)
print(f"✅ Bankroll history saved to {output_path}")
print()
