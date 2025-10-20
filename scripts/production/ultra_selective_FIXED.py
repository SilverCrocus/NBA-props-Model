#!/usr/bin/env python3
"""
Ultra-Selective Betting Strategy - Applied to LEAK-FREE Predictions

This script applies the 4-tier quality scoring system to the calibrated
predictions that have NO DATA LEAKAGE (trained on 2023-24, applied to 2024-25).
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

print("=" * 80)
print("ULTRA-SELECTIVE STRATEGY - LEAK-FREE VERSION")
print("=" * 80)
print()

# Load leak-free calibrated predictions
print("1. Loading leak-free calibrated predictions (2024-25)...")
df = pd.read_csv("data/results/backtest_2024_25_CALIBRATED_FIXED.csv")
df = df.dropna(subset=["predicted_PRA_calibrated", "betting_line"])

print(f"   ✅ Loaded {len(df):,} predictions")
print()

# Calculate edge using CALIBRATED predictions
df["edge"] = df["predicted_PRA_calibrated"] - df["betting_line"]
df["abs_edge"] = df["edge"].abs()

# Phase 1: Edge filter (5-7 pts)
print("2. Applying Phase 1 filter (edge 5-7 pts)...")
phase1_filtered = df[(df["abs_edge"] >= 5.0) & (df["abs_edge"] <= 7.0)].copy()
print(
    f"   ✅ {len(phase1_filtered):,} bets pass edge filter ({len(phase1_filtered)/len(df)*100:.1f}%)"
)
print()


# Quality Scoring Function
def calculate_quality_score(row):
    """4-Tier Quality Scoring System"""
    scores = {}

    # Tier 1: Edge Quality (30%)
    abs_edge = row.get("abs_edge", 0)
    if abs_edge >= 6.75:
        scores["edge"] = 1.0
    elif abs_edge >= 6.5:
        scores["edge"] = 0.9
    elif abs_edge >= 6.0:
        scores["edge"] = 0.7
    elif abs_edge >= 5.5:
        scores["edge"] = 0.5
    else:
        scores["edge"] = 0.0

    # Tier 2: Prediction Confidence (25%)
    pred_pra = row.get("predicted_PRA_calibrated", 0)
    if 18 <= pred_pra <= 28:
        scores["confidence"] = 1.0
    elif 15 <= pred_pra <= 32:
        scores["confidence"] = 0.8
    elif 10 <= pred_pra <= 35:
        scores["confidence"] = 0.5
    else:
        scores["confidence"] = 0.3

    # Tier 3: Game Context (25%)
    # Simplified (no minutes data in walk-forward backtest)
    scores["context"] = 0.7  # Default moderate score

    # Tier 4: Player Consistency (20%)
    if pred_pra < 18:
        scores["consistency"] = 0.9  # Role players more consistent
    elif pred_pra < 25:
        scores["consistency"] = 0.7
    elif pred_pra < 35:
        scores["consistency"] = 0.5
    else:
        scores["consistency"] = 0.3  # High-usage players variable

    # Weighted average
    weights = {"edge": 0.30, "confidence": 0.25, "context": 0.25, "consistency": 0.20}
    quality_score = sum(scores[key] * weights[key] for key in scores.keys())

    return quality_score, scores


# Apply quality scoring
print("3. Applying 4-tier quality scoring...")
quality_data = phase1_filtered.apply(calculate_quality_score, axis=1)
phase1_filtered["quality_score"] = [q[0] for q in quality_data]
print("   ✅ Quality scores calculated")
print()

# Ultra-selective filter (quality ≥ 0.75)
print("4. Applying ultra-selective filter (quality ≥ 0.75)...")
ultra_selective = phase1_filtered[phase1_filtered["quality_score"] >= 0.75].copy()
print(
    f"   ✅ {len(ultra_selective):,} bets selected ({len(ultra_selective)/len(df)*100:.1f}% of all predictions)"
)
print()

# Determine bet side
ultra_selective["bet_side"] = ultra_selective["edge"].apply(lambda x: "OVER" if x > 0 else "UNDER")

# Determine outcome
ultra_selective["bet_won"] = ultra_selective.apply(
    lambda row: (
        row["actual_pra"] > row["betting_line"]
        if row["bet_side"] == "OVER"
        else row["actual_pra"] < row["betting_line"]
    ),
    axis=1,
)

# Calculate performance
total_bets = len(ultra_selective)
wins = ultra_selective["bet_won"].sum()
losses = total_bets - wins
win_rate = wins / total_bets if total_bets > 0 else 0

print("=" * 80)
print("RESULTS - LEAK-FREE ULTRA-SELECTIVE STRATEGY")
print("=" * 80)
print()

print(f"Total Bets: {total_bets}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Win Rate: {win_rate*100:.2f}%")
print()

# Statistical significance test
if total_bets > 0:
    # Test against 56% null hypothesis (professional sharp level)
    result = binomtest(wins, total_bets, 0.56, alternative="two-sided")
    print(f"Statistical Test (H0: win rate = 56%):")
    print(f"  P-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print(f"  ✅ Significantly different from 56% (p < 0.05)")
    else:
        print(f"  ✅ Not significantly different from 56% (consistent with sharp bettor)")
    print()

# ROI calculation (assuming -110 odds)
roi = (wins * 0.909 - losses) / total_bets if total_bets > 0 else 0
print(f"ROI (flat betting): {roi*100:+.2f}%")
print()

# Bankroll simulation (Fixed 2%)
print("=" * 80)
print("BANKROLL SIMULATION - LEAK-FREE PREDICTIONS")
print("=" * 80)
print()

if total_bets > 0:
    # Sort by date
    ultra_selective = ultra_selective.sort_values("game_date")

    # Fixed 2% simulation
    bankroll_2pct = 1000.0
    for _, row in ultra_selective.iterrows():
        bet_size = bankroll_2pct * 0.02
        if row["bet_won"]:
            bankroll_2pct += bet_size * 0.909
        else:
            bankroll_2pct -= bet_size

    profit_2pct = bankroll_2pct - 1000
    roi_2pct = (profit_2pct / 1000) * 100

    print(f"Starting Bankroll: $1,000")
    print(f"Ending Bankroll (Fixed 2%): ${bankroll_2pct:,.2f}")
    print(f"Profit: ${profit_2pct:+,.2f}")
    print(f"ROI: {roi_2pct:+.1f}%")
    print()

# Save results
ultra_selective.to_csv("data/results/backtest_2024_25_ULTRA_SELECTIVE_FIXED.csv", index=False)
print("✅ Saved: data/results/backtest_2024_25_ULTRA_SELECTIVE_FIXED.csv")
print()

print("=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print()

if win_rate < 0.52:
    print("⚠️  Win rate below breakeven (52.4%) - strategy not profitable")
elif win_rate < 0.56:
    print("⚠️  Win rate below sharp bettor level (56%) - needs improvement")
elif win_rate <= 0.60:
    print("✅ Win rate at professional sharp level (56-60%) - EXCELLENT!")
else:
    print("⚠️  Win rate suspiciously high (>60%) - verify no remaining leakage")
