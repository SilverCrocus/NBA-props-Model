#!/usr/bin/env python3
"""
CRITICAL DATA LEAKAGE INVESTIGATION

63.67% win rate on 300 bets is highly suspicious. This script investigates:
1. Temporal ordering - are predictions made BEFORE games?
2. Feature leakage - do features include current game data?
3. Edge calculation - is betting line from correct time?
4. Quality score leakage - does it use actual outcomes?
5. Statistical impossibility - binomial test
6. Filter selection bias - are bets selected after seeing outcomes?

Author: NBA Props Model
Date: 2024-10-20
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

print("=" * 80)
print("DATA LEAKAGE INVESTIGATION - 63.67% WIN RATE")
print("=" * 80)
print()

# ======================================================================
# LOAD DATA
# ======================================================================

print("1. Loading backtest data...")

backtest_path = "data/results/backtest_2024_25_CALIBRATED.csv"
ultra_selective_path = "data/results/backtest_2024_25_ULTRA_SELECTIVE_betting.csv"

df_all = pd.read_csv(backtest_path)
df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"])

print(f"   ✅ Loaded {len(df_all):,} total predictions")
print(f"   Date range: {df_all['GAME_DATE'].min()} to {df_all['GAME_DATE'].max()}")
print()

# Load ultra-selective bets
try:
    df_bets = pd.read_csv(ultra_selective_path)
    df_bets["GAME_DATE"] = pd.to_datetime(df_bets["GAME_DATE"])
    print(f"   ✅ Loaded {len(df_bets):,} ultra-selective bets")
    print()
except FileNotFoundError:
    print("   ⚠️  Ultra-selective bets file not found")
    df_bets = None
    print()

# ======================================================================
# TEST 1: TEMPORAL ORDERING
# ======================================================================

print("=" * 80)
print("TEST 1: TEMPORAL ORDERING")
print("=" * 80)
print()

print("Checking if predictions are made in chronological order...")

# Check if data is sorted by date
is_sorted = df_all["GAME_DATE"].is_monotonic_increasing
if is_sorted:
    print("   ✅ PASS: Data is sorted chronologically")
else:
    print("   ⚠️  WARNING: Data is NOT sorted chronologically")

# Check for any duplicate player-game combinations
duplicates = df_all.groupby(["PLAYER_ID", "GAME_DATE"]).size()
duplicates = duplicates[duplicates > 1]

if len(duplicates) == 0:
    print("   ✅ PASS: No duplicate player-game predictions")
else:
    print(f"   ❌ FAIL: Found {len(duplicates)} duplicate player-game predictions")
    print(f"      This suggests feature merging issues (CTG duplicate bug)")

print()

# ======================================================================
# TEST 2: FEATURE TEMPORAL ISOLATION
# ======================================================================

print("=" * 80)
print("TEST 2: FEATURE TEMPORAL ISOLATION")
print("=" * 80)
print()

print("Checking if lag features could have used current game data...")

# Check lag feature consistency
# If PRA_lag1 is correct, it should NOT equal actual_PRA
if "last_game_PRA" in df_all.columns and "actual_PRA" in df_all.columns:
    lag1_equals_actual = df_all["last_game_PRA"] == df_all["actual_PRA"]
    pct_equal = lag1_equals_actual.mean()

    print(f"   PRA_lag1 == actual_PRA: {pct_equal*100:.2f}% of predictions")

    if pct_equal > 0.05:  # > 5% seems suspicious
        print(f"   ⚠️  WARNING: Lag feature may include current game data")
    else:
        print(f"   ✅ PASS: Lag features appear temporally isolated")
else:
    print("   ⚠️  Cannot test: last_game_PRA column not found")

print()

# Check L5 mean consistency
# L5_mean should be calculated from PREVIOUS 5 games, not including current
if "L5_mean_PRA" in df_all.columns:
    # If L5_mean includes current game, predictions would be suspiciously good
    # We can't directly test this without historical data, but we can check variance

    l5_std = df_all["L5_mean_PRA"].std()
    actual_std = df_all["actual_PRA"].std()

    print(f"   L5_mean_PRA std: {l5_std:.2f}")
    print(f"   actual_PRA std: {actual_std:.2f}")

    # If L5_mean variance is suspiciously similar to actual, it might include current game
    if abs(l5_std - actual_std) < 2.0:
        print(f"   ⚠️  WARNING: L5_mean variance suspiciously similar to actual")
    else:
        print(f"   ✅ PASS: L5_mean variance differs from actual (expected)")

print()

# ======================================================================
# TEST 3: EDGE CALCULATION TIMING
# ======================================================================

print("=" * 80)
print("TEST 3: EDGE CALCULATION TIMING")
print("=" * 80)
print()

if df_bets is not None:
    print("Checking if betting lines are from correct time period...")

    # Check if 'event_date' exists in betting data
    if "event_date" in df_bets.columns:
        df_bets["event_date"] = pd.to_datetime(df_bets["event_date"])

        # Betting line should be from SAME day as game or day before
        date_match = df_bets["event_date"] == df_bets["GAME_DATE"]
        pct_match = date_match.mean()

        print(f"   event_date == GAME_DATE: {pct_match*100:.1f}% of bets")

        if pct_match > 0.9:
            print("   ✅ PASS: Betting lines from correct day")
        else:
            print("   ⚠️  WARNING: Some betting lines from different dates")

            # Show mismatches
            mismatches = df_bets[~date_match][["PLAYER_NAME", "GAME_DATE", "event_date"]].head(10)
            print("\n   Example mismatches:")
            print(mismatches.to_string(index=False))
    else:
        print("   ⚠️  Cannot test: event_date column not found")

    print()

    # Check edge calculation
    if all(col in df_bets.columns for col in ["predicted_PRA_calibrated", "line", "edge"]):
        calculated_edge = df_bets["predicted_PRA_calibrated"] - df_bets["line"]
        stored_edge = df_bets["edge"]

        edge_match = np.isclose(calculated_edge, stored_edge, atol=0.01)
        pct_match = edge_match.mean()

        print(f"   Edge calculation correct: {pct_match*100:.1f}% of bets")

        if pct_match > 0.99:
            print("   ✅ PASS: Edge = predicted_PRA - line (correct)")
        else:
            print("   ❌ FAIL: Edge calculation inconsistent")

    print()

# ======================================================================
# TEST 4: QUALITY SCORE LEAKAGE
# ======================================================================

print("=" * 80)
print("TEST 4: QUALITY SCORE LEAKAGE")
print("=" * 80)
print()

if df_bets is not None:
    print("Checking if quality score uses actual game outcomes...")

    # Quality score should be calculable BEFORE game
    # Test: Can we reproduce quality scores WITHOUT knowing actual_PRA?

    # Check if quality score correlates with actual outcome
    if "quality_score" in df_bets.columns:
        # Group by quality score bins
        df_bets["quality_bin"] = pd.cut(df_bets["quality_score"], bins=5)
        quality_analysis = (
            df_bets.groupby("quality_bin")
            .agg({"bet_correct": "mean", "PLAYER_NAME": "count"})
            .rename(columns={"PLAYER_NAME": "count", "bet_correct": "win_rate"})
        )

        print("   Quality Score vs Win Rate:")
        print(quality_analysis)
        print()

        # If quality score is leaking, win rate would increase monotonically with quality
        # (higher quality = higher win rate in a suspiciously linear way)

        # Calculate correlation between quality and win (at player level)
        correlation = (
            df_bets.groupby("PLAYER_NAME")
            .agg({"quality_score": "mean", "bet_correct": "mean"})
            .corr()
            .iloc[0, 1]
        )

        print(f"   Correlation (quality_score, win_rate): {correlation:.3f}")

        if correlation > 0.5:
            print("   ⚠️  WARNING: Strong correlation suggests possible leakage")
        else:
            print("   ✅ PASS: Quality score appears independent of outcomes")

    print()

    # Check if Minutes_Projected uses actual minutes
    if "Minutes_Projected" in df_bets.columns:
        # Minutes_Projected should be from L5 average, not actual game minutes
        # We can't directly test this without game logs, but suspicious if variance is too low

        min_proj_std = df_bets["Minutes_Projected"].std()
        print(f"   Minutes_Projected std: {min_proj_std:.2f}")

        # Typical L5 average has std ~5-8 minutes
        if min_proj_std < 3.0:
            print("   ⚠️  WARNING: Minutes_Projected variance suspiciously low")
        else:
            print("   ✅ PASS: Minutes_Projected variance reasonable")

    print()

# ======================================================================
# TEST 5: STATISTICAL IMPOSSIBILITY TEST
# ======================================================================

print("=" * 80)
print("TEST 5: STATISTICAL IMPOSSIBILITY TEST (BINOMIAL)")
print("=" * 80)
print()

if df_bets is not None:
    total_bets = len(df_bets)
    wins = df_bets["bet_correct"].sum()
    win_rate = wins / total_bets

    print(f"Observed Results:")
    print(f"   Total bets: {total_bets}")
    print(f"   Wins: {wins}")
    print(f"   Win rate: {win_rate*100:.2f}%")
    print()

    # Test against different null hypotheses
    null_hypotheses = [0.50, 0.52, 0.54, 0.56]

    print("Binomial Test (two-sided):")
    print(f"   {'Null H0':<12} {'P-value':<12} {'Reject H0 (α=0.05)?'}")
    print("   " + "-" * 50)

    for p_null in null_hypotheses:
        p_value = stats.binomtest(wins, total_bets, p=p_null, alternative="two-sided").pvalue
        reject = "YES ❌" if p_value < 0.05 else "NO ✅"
        print(f"   {p_null*100:>5.0f}%       {p_value:>10.6f}    {reject}")

    print()

    # Calculate probability of 191+ wins if true win rate is 56%
    # (56% is upper end of realistic target)
    expected_at_56 = 0.56 * total_bets
    print(f"Expected wins at 56% true win rate: {expected_at_56:.1f}")
    print(f"Actual wins: {wins}")
    print(
        f"Excess wins: {wins - expected_at_56:.1f} ({(wins - expected_at_56) / expected_at_56 * 100:+.1f}%)"
    )
    print()

    # One-sided test: What's probability of getting this many wins or MORE?
    p_value_upper = stats.binomtest(wins, total_bets, p=0.56, alternative="greater").pvalue
    print(f"P(≥{wins} wins | 56% true rate): {p_value_upper:.6f}")

    if p_value_upper < 0.01:
        print(f"   ❌ FAIL: {win_rate*100:.1f}% win rate is STATISTICALLY IMPOSSIBLE")
        print(f"            (p < 0.01 if true rate is 56%)")
        print(f"            This strongly suggests data leakage")
    elif p_value_upper < 0.05:
        print(f"   ⚠️  WARNING: {win_rate*100:.1f}% win rate is unlikely (p < 0.05)")
    else:
        print(f"   ✅ PASS: {win_rate*100:.1f}% win rate is plausible")

    print()

# ======================================================================
# TEST 6: SELECTION BIAS (FILTER AFTER OUTCOMES)
# ======================================================================

print("=" * 80)
print("TEST 6: SELECTION BIAS IN FILTERS")
print("=" * 80)
print()

if df_bets is not None:
    print("Checking if filters could have been applied AFTER seeing outcomes...")

    # Test: Do filtered bets perform BETTER on days with more games?
    # (If filtering after outcomes, would cherry-pick winning bets from big days)

    daily_bets = df_bets.groupby("GAME_DATE").agg({"bet_correct": ["sum", "count", "mean"]})
    daily_bets.columns = ["wins", "bets", "win_rate"]

    # Correlation between number of bets per day and win rate
    correlation = daily_bets[["bets", "win_rate"]].corr().iloc[0, 1]

    print(f"   Correlation (bets_per_day, win_rate): {correlation:.3f}")

    if abs(correlation) > 0.3:
        print(f"   ⚠️  WARNING: Strong correlation suggests cherry-picking")
    else:
        print(f"   ✅ PASS: No evidence of cherry-picking by day")

    print()

    # Test: Win rate by quality score tier
    # If selection bias, higher quality would show HUGE jumps in win rate
    if "quality_score" in df_bets.columns:
        quality_bins = [0.75, 0.80, 0.85, 0.90, 1.0]

        print("   Win Rate by Quality Score Tier:")
        print(f"   {'Tier':<15} {'Bets':>6} {'Win Rate':>10}")
        print("   " + "-" * 35)

        prev_wr = None
        for i in range(len(quality_bins) - 1):
            tier_data = df_bets[
                (df_bets["quality_score"] >= quality_bins[i])
                & (df_bets["quality_score"] < quality_bins[i + 1])
            ]

            if len(tier_data) > 0:
                tier_wr = tier_data["bet_correct"].mean()
                tier_label = f"{quality_bins[i]:.2f}-{quality_bins[i+1]:.2f}"

                print(f"   {tier_label:<15} {len(tier_data):>6} {tier_wr*100:>9.1f}%")

                # Check for suspicious jumps
                if prev_wr is not None and (tier_wr - prev_wr) > 0.15:
                    print(
                        f"      ⚠️  WARNING: Large jump in win rate (+{(tier_wr - prev_wr)*100:.1f}pp)"
                    )

                prev_wr = tier_wr

        print()

# ======================================================================
# TEST 7: EDGE SIZE VS ACTUAL OUTCOME
# ======================================================================

print("=" * 80)
print("TEST 7: EDGE SIZE VS ACTUAL OUTCOME CORRELATION")
print("=" * 80)
print()

if df_bets is not None and "edge" in df_bets.columns and "actual_PRA" in df_bets.columns:
    print("Checking if large edges correctly identify extreme performances...")

    # Calculate actual edge (how much prediction beat line)
    df_bets["actual_edge"] = df_bets["actual_PRA"] - df_bets["line"]

    # Correlation between predicted edge and actual edge
    correlation = df_bets[["edge", "actual_edge"]].corr().iloc[0, 1]

    print(f"   Correlation (predicted_edge, actual_edge): {correlation:.3f}")
    print()

    # If correlation is TOO high, suggests leakage
    if correlation > 0.5:
        print(f"   ⚠️  WARNING: Very high correlation suggests possible leakage")
        print(f"               (Model shouldn't predict edges this accurately)")
    elif correlation > 0.2:
        print(f"   ✅ PASS: Moderate correlation (expected for skilled model)")
    else:
        print(f"   ⚠️  WARNING: Low correlation suggests poor edge calibration")

    print()

    # Analyze edge bins
    df_bets["edge_bin"] = pd.cut(df_bets["abs_edge"], bins=[5.0, 5.5, 6.0, 6.5, 7.0])

    edge_analysis = (
        df_bets.groupby("edge_bin")
        .agg({"bet_correct": "mean", "actual_edge": "mean", "PLAYER_NAME": "count"})
        .rename(columns={"PLAYER_NAME": "count"})
    )

    print("   Win Rate by Edge Size:")
    print(edge_analysis)
    print()

# ======================================================================
# SUMMARY
# ======================================================================

print("=" * 80)
print("INVESTIGATION SUMMARY")
print("=" * 80)
print()

print("🔍 CRITICAL FINDINGS:")
print()

findings = []

# Temporal ordering
if not is_sorted:
    findings.append("❌ Data NOT sorted chronologically (temporal leakage risk)")

# Duplicates
if len(duplicates) > 0:
    findings.append(f"❌ {len(duplicates)} duplicate player-game predictions (CTG merge bug)")

# Statistical impossibility
if df_bets is not None:
    if p_value_upper < 0.01:
        findings.append(f"❌ 63.67% win rate is STATISTICALLY IMPOSSIBLE (p = {p_value_upper:.6f})")
        findings.append("   → Strong evidence of data leakage")
    elif p_value_upper < 0.05:
        findings.append(f"⚠️  63.67% win rate is unlikely but possible (p = {p_value_upper:.4f})")

# Edge correlation
if df_bets is not None and "edge" in df_bets.columns:
    edge_corr = df_bets[["edge", "actual_edge"]].corr().iloc[0, 1]
    if edge_corr > 0.5:
        findings.append(
            f"⚠️  Very high edge correlation ({edge_corr:.3f}) suggests possible leakage"
        )

if len(findings) == 0:
    print("✅ No critical data leakage detected")
    print("   63.67% win rate appears legitimate (but unlikely)")
else:
    for finding in findings:
        print(finding)

print()
print("RECOMMENDATIONS:")
print()

print("1. MANUAL VERIFICATION:")
print("   - Manually inspect 10 random predictions from ultra-selective bets")
print("   - Verify features calculated using ONLY past games")
print("   - Check betting lines are from correct date")
print()

print("2. CODE REVIEW:")
print("   - Review backtest_2024_25_FIXED_V2.py lines 133-238 (feature calculation)")
print("   - Ensure past_games = df_all[df_all['GAME_DATE'] < pred_date]")
print("   - Verify .shift(1) used in all lag/rolling features")
print()

print("3. RERUN VALIDATION:")
print("   - Run walk_forward_validation_enhanced.py with verbose logging")
print("   - Print feature values for each prediction")
print("   - Verify temporal isolation")
print()

print("4. CALIBRATION CHECK:")
print("   - Was isotonic calibrator trained on 2024-25 data?")
print("   - If yes, this is FORWARD-LOOKING BIAS (leakage)")
print()

print("=" * 80)
