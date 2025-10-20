#!/usr/bin/env python3
"""
Phase 1 Betting Simulation - 2024-25 Season

Matches Phase 1 test predictions (MAE 4.19) to betting odds to calculate
true win rate and ROI, comparing to baseline (52.94% win rate, 1.06% ROI).

Expected: 54-55% win rate based on 52.5% MAE improvement
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print("=" * 80)
print("PHASE 1 BETTING SIMULATION - 2024-25 SEASON")
print("=" * 80)
print()

# ======================================================================
# CONFIGURATION
# ======================================================================

PHASE1_PREDICTIONS_PATH = "data/results/phase1_test_predictions_2024_25.csv"
BETTING_LINES_PATH = "data/historical_odds/2024-25/pra_odds.csv"
OUTPUT_PATH = "data/results/phase1_betting_simulation_2024_25.csv"
ULTRA_SELECTIVE_OUTPUT = "data/results/phase1_ultra_selective_bets.csv"

# Star players to exclude (efficient markets)
STAR_PLAYERS = [
    "LeBron James",
    "Stephen Curry",
    "Kevin Durant",
    "Giannis Antetokounmpo",
    "Luka Doncic",
    "Joel Embiid",
    "Nikola Jokic",
    "Jayson Tatum",
    "Anthony Davis",
    "Damian Lillard",
    "Devin Booker",
    "Jimmy Butler",
    "Kawhi Leonard",
    "Paul George",
    "Kyrie Irving",
    "James Harden",
    "Trae Young",
    "Donovan Mitchell",
    "Jaylen Brown",
    "LaMelo Ball",
    "De'Aaron Fox",
    "Anthony Edwards",
    "Pascal Siakam",
    "Tyrese Haliburton",
    "DeMar DeRozan",
    "Shai Gilgeous-Alexander",
    "Bam Adebayo",
    "Domantas Sabonis",
    "Julius Randle",
    "Zion Williamson",
    "Brandon Ingram",
    "Dejounte Murray",
    "Fred VanVleet",
    "Jrue Holiday",
    "CJ McCollum",
    "Tobias Harris",
    "Khris Middleton",
    "Tyler Herro",
    "Jarrett Allen",
    "Kristaps Porzingis",
    "Clint Capela",
    "Jaren Jackson Jr",
    "Scottie Barnes",
    "Paolo Banchero",
    "Franz Wagner",
    "Cade Cunningham",
    "Jalen Green",
    "Alperen Sengun",
    "Lauri Markkanen",
    "Walker Kessler",
    "Evan Mobley",
    "Darius Garland",
    "Derrick White",
    "OG Anunoby",
    "RJ Barrett",
    "Immanuel Quickley",
    "Mikal Bridges",
    "Cameron Johnson",
    "Nic Claxton",
    "Spencer Dinwiddie",
    "Cam Thomas",
]

# ======================================================================
# QUALITY SCORING SYSTEM (4-TIER)
# ======================================================================


def calculate_quality_score(row):
    """
    4-Tier Quality Scoring System

    Returns 0.0-1.0, threshold at 0.75 for ultra-selective betting

    Tiers:
    1. Edge Quality (30%): Larger edges more reliable
    2. Prediction Confidence (25%): Sweet spot avoids extremes
    3. Game Context (25%): Minutes projection
    4. Player Consistency (20%): Low variance more predictable
    """
    scores = {}

    # ===================================================================
    # TIER 1: Edge Quality (Weight: 30%)
    # ===================================================================
    abs_edge = abs(row.get("edge", 0))

    if abs_edge >= 6.75:
        scores["edge"] = 1.0  # Premium edge
    elif abs_edge >= 6.5:
        scores["edge"] = 0.9  # Excellent edge
    elif abs_edge >= 6.0:
        scores["edge"] = 0.7  # Good edge
    elif abs_edge >= 5.5:
        scores["edge"] = 0.5  # Acceptable edge
    else:
        scores["edge"] = 0.0  # Too small

    # ===================================================================
    # TIER 2: Prediction Confidence (Weight: 25%)
    # ===================================================================
    # Model performs best in 15-30 PRA range
    pred_pra = row.get("predicted_pra", 0)

    if 18 <= pred_pra <= 28:
        scores["confidence"] = 1.0  # Sweet spot
    elif 15 <= pred_pra <= 32:
        scores["confidence"] = 0.8  # Good range
    elif 12 <= pred_pra <= 35:
        scores["confidence"] = 0.5  # Acceptable
    else:
        scores["confidence"] = 0.3  # Risky (extreme prediction)

    # ===================================================================
    # TIER 3: Game Context (Weight: 25%)
    # ===================================================================
    # Use predicted PRA as proxy for minutes/role (higher PRA = more minutes)
    # In production, use actual Minutes_Projected feature

    if pred_pra >= 28:
        scores["context"] = 1.0  # Starter with heavy usage
    elif pred_pra >= 20:
        scores["context"] = 0.9  # Regular starter
    elif pred_pra >= 15:
        scores["context"] = 0.6  # Rotation player
    else:
        scores["context"] = 0.4  # Limited role

    # ===================================================================
    # TIER 4: Player Consistency (Weight: 20%)
    # ===================================================================
    # Role players (PRA < 18) more consistent than stars

    if pred_pra < 18:
        scores["consistency"] = 0.9  # Role players more consistent
    elif 18 <= pred_pra <= 28:
        scores["consistency"] = 1.0  # Sweet spot for consistency
    else:
        scores["consistency"] = 0.6  # Stars have higher variance

    # ===================================================================
    # COMBINED QUALITY SCORE (Weighted Average)
    # ===================================================================

    weights = {"edge": 0.30, "confidence": 0.25, "context": 0.25, "consistency": 0.20}

    quality_score = sum(scores[key] * weights[key] for key in scores.keys())

    return quality_score, scores


# ======================================================================
# LOAD DATA
# ======================================================================

print("1. Loading Phase 1 predictions...")
preds_df = pd.read_csv(PHASE1_PREDICTIONS_PATH)
preds_df["GAME_DATE"] = pd.to_datetime(preds_df["GAME_DATE"])
print(f"   ✅ Loaded {len(preds_df):,} predictions")
print(f"   Date range: {preds_df['GAME_DATE'].min()} to {preds_df['GAME_DATE'].max()}")
print(f"   Players: {preds_df['PLAYER_NAME'].nunique()}")
print()

print("2. Loading betting odds...")
odds_df = pd.read_csv(BETTING_LINES_PATH)
odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])
print(f"   ✅ Loaded {len(odds_df):,} betting lines")
print(f"   Date range: {odds_df['event_date'].min()} to {odds_df['event_date'].max()}")
print(f"   Players: {odds_df['player_name'].nunique()}")
print(f"   Bookmakers: {odds_df['bookmaker'].nunique()}")
print()

# ======================================================================
# PREPROCESS FOR MATCHING
# ======================================================================

print("3. Preprocessing for name/date matching...")

# Standardize player names
preds_df["player_lower"] = preds_df["PLAYER_NAME"].str.lower().str.strip()
odds_df["player_lower"] = odds_df["player_name"].str.lower().str.strip()

# Standardize dates
preds_df["game_date"] = preds_df["GAME_DATE"].dt.date
odds_df["game_date"] = odds_df["event_date"].dt.date

print(f"   ✅ Names and dates standardized")
print()

# ======================================================================
# LINE SHOPPING (Best Odds Across Bookmakers)
# ======================================================================

print("4. Line shopping (selecting best odds)...")

odds_best = odds_df.groupby(["player_lower", "game_date"], as_index=False).agg(
    {
        "line": "first",  # Use first available line (typically consistent)
        "over_price": "max",  # Best OVER odds
        "under_price": "max",  # Best UNDER odds
        "bookmaker": lambda x: ", ".join(x.unique()[:3]),  # Show which books
    }
)

print(f"   ✅ Best odds selected for {len(odds_best):,} player-date combinations")
print()

# ======================================================================
# MATCH PREDICTIONS TO ODDS
# ======================================================================

print("5. Matching predictions to betting lines...")

merged = preds_df.merge(
    odds_best,
    left_on=["player_lower", "game_date"],
    right_on=["player_lower", "game_date"],
    how="inner",
)

print(f"   ✅ Matched {len(merged):,} predictions to odds")
print(f"   Match rate: {len(merged)/len(preds_df)*100:.1f}%")
print()

# ======================================================================
# CALCULATE EDGES AND BET SIDES
# ======================================================================

print("6. Calculating edges and bet sides...")

merged["actual_pra"] = merged["actual_pra"]
merged["betting_line"] = merged["line"]
merged["edge"] = merged["predicted_pra"] - merged["betting_line"]
merged["abs_edge"] = merged["edge"].abs()

# Determine bet side (≥3 pts edge threshold)
merged["bet_side"] = "NONE"
merged.loc[merged["edge"] >= 3, "bet_side"] = "OVER"
merged.loc[merged["edge"] <= -3, "bet_side"] = "UNDER"

print(f"   Mean edge: {merged['edge'].mean():.2f} pts")
print(f"   Median edge: {merged['edge'].median():.2f} pts")
print(f"   Bets identified (|edge| ≥ 3): {(merged['bet_side'] != 'NONE').sum():,}")
print()

# ======================================================================
# CALCULATE BET OUTCOMES
# ======================================================================

print("7. Calculating bet outcomes...")


def calculate_bet_result(row):
    """Calculate profit/loss for a bet"""
    if row["bet_side"] == "NONE":
        return 0

    actual = row["actual_pra"]
    line = row["betting_line"]

    # Determine if bet won
    if row["bet_side"] == "OVER":
        won = actual > line
        odds = row["over_price"]
    else:  # UNDER
        won = actual < line
        odds = row["under_price"]

    # Handle push (tie)
    if actual == line:
        return 0

    # Calculate profit/loss
    if won:
        # Convert American odds to profit
        if odds > 0:
            return odds  # e.g., +110 = $110 profit on $100 bet
        else:
            return 100 * (100 / abs(odds))  # e.g., -110 = $90.91 profit
    else:
        return -100  # Loss


merged["bet_result"] = merged.apply(calculate_bet_result, axis=1)
merged["bet_won"] = (merged["bet_result"] > 0) & (merged["bet_side"] != "NONE")
merged["bet_pushed"] = (merged["bet_result"] == 0) & (merged["bet_side"] != "NONE")
merged["bet_lost"] = (merged["bet_result"] < 0) & (merged["bet_side"] != "NONE")

print(f"   ✅ Bet outcomes calculated")
print()

# ======================================================================
# OVERALL PERFORMANCE (ALL BETS WITH EDGE ≥ ±3)
# ======================================================================

bets_all = merged[merged["bet_side"] != "NONE"].copy()

total_bets = len(bets_all)
won_bets = bets_all["bet_won"].sum()
pushed_bets = bets_all["bet_pushed"].sum()
lost_bets = bets_all["bet_lost"].sum()
total_profit = bets_all["bet_result"].sum()
total_wagered = total_bets * 100

win_rate_all = won_bets / (won_bets + lost_bets) if (won_bets + lost_bets) > 0 else 0
roi_all = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

print("=" * 80)
print("OVERALL PERFORMANCE (All bets with |edge| ≥ 3 pts)")
print("=" * 80)
print()
print(f"   Total bets: {total_bets:,}")
print(f"   Wins: {won_bets:,}")
print(f"   Losses: {lost_bets:,}")
print(f"   Pushes: {pushed_bets:,}")
print(f"   Win rate: {win_rate_all*100:.2f}%")
print(f"   ROI: {roi_all:+.2f}%")
print(f"   Total profit: ${total_profit:,.2f}")
print()

# Statistical significance
p_value_all = stats.binomtest(
    won_bets, won_bets + lost_bets, p=0.50, alternative="two-sided"
).pvalue
print(f"   Statistical significance: p = {p_value_all:.4f}")
if p_value_all < 0.05:
    print(f"   ✅ Statistically significant (p < 0.05)")
else:
    print(f"   ⚠️  Not statistically significant (p ≥ 0.05)")
print()

# ======================================================================
# APPLY QUALITY SCORING FOR ULTRA-SELECTIVE FILTERING
# ======================================================================

print("=" * 80)
print("ULTRA-SELECTIVE FILTERING (4-Tier Quality Scoring)")
print("=" * 80)
print()

print("8. Calculating quality scores...")

# Calculate quality scores
quality_data = bets_all.apply(calculate_quality_score, axis=1)
bets_all["quality_score"] = [q[0] for q in quality_data]

# Store individual tier scores
for tier in ["edge", "confidence", "context", "consistency"]:
    bets_all[f"score_{tier}"] = [q[1][tier] for q in quality_data]

print(f"   ✅ Quality scores calculated")
print(f"   Mean quality score: {bets_all['quality_score'].mean():.3f}")
print(f"   Median quality score: {bets_all['quality_score'].median():.3f}")
print()

# ======================================================================
# APPLY ULTRA-SELECTIVE FILTERS
# ======================================================================

print("9. Applying ultra-selective filters...")
print()

# Base filter: Edge range 5-7 pts (proven optimal)
base_filtered = bets_all[
    (bets_all["abs_edge"] >= 5.0)
    & (bets_all["abs_edge"] <= 7.0)
    & (~bets_all["PLAYER_NAME"].isin(STAR_PLAYERS))
].copy()

print(f"   Base filters (edge 5-7 pts, no stars): {len(base_filtered):,} bets")

# Calculate base filtered performance
if len(base_filtered) > 0:
    base_wins = base_filtered["bet_won"].sum()
    base_losses = base_filtered["bet_lost"].sum()
    base_win_rate = base_wins / (base_wins + base_losses) if (base_wins + base_losses) > 0 else 0
    base_roi = (base_filtered["bet_result"].sum() / (len(base_filtered) * 100)) * 100
    print(f"   Win rate: {base_win_rate*100:.2f}%")
    print(f"   ROI: {base_roi:+.2f}%")
print()

# Ultra-selective filter: Quality ≥ 0.75 (top 20%)
ultra_selective = base_filtered[base_filtered["quality_score"] >= 0.75].copy()

print(f"   Ultra-selective (quality ≥ 0.75): {len(ultra_selective):,} bets")
print(
    f"   Reduction: {len(base_filtered):,} → {len(ultra_selective):,} ({len(ultra_selective)/len(base_filtered)*100:.1f}%)"
)
print()

# ======================================================================
# ULTRA-SELECTIVE PERFORMANCE
# ======================================================================

if len(ultra_selective) > 0:
    ultra_wins = ultra_selective["bet_won"].sum()
    ultra_losses = ultra_selective["bet_lost"].sum()
    ultra_pushes = ultra_selective["bet_pushed"].sum()
    ultra_profit = ultra_selective["bet_result"].sum()
    ultra_wagered = len(ultra_selective) * 100

    ultra_win_rate = (
        ultra_wins / (ultra_wins + ultra_losses) if (ultra_wins + ultra_losses) > 0 else 0
    )
    ultra_roi = (ultra_profit / ultra_wagered) * 100

    print("=" * 80)
    print("ULTRA-SELECTIVE PERFORMANCE")
    print("=" * 80)
    print()
    print(f"   📊 Total bets: {len(ultra_selective):,}")
    print(f"   Wins: {ultra_wins:,}")
    print(f"   Losses: {ultra_losses:,}")
    print(f"   Pushes: {ultra_pushes:,}")
    print(f"   Win rate: {ultra_win_rate*100:.2f}%")
    print(f"   ROI: {ultra_roi:+.2f}%")
    print(f"   Total profit: ${ultra_profit:,.2f}")
    print()

    # Statistical significance
    p_value_ultra = stats.binomtest(
        ultra_wins, ultra_wins + ultra_losses, p=0.50, alternative="two-sided"
    ).pvalue
    print(f"   Statistical significance: p = {p_value_ultra:.4f}")
    if p_value_ultra < 0.05:
        print(f"   ✅ Statistically significant (p < 0.05)")
    else:
        print(f"   ⚠️  Not statistically significant (p ≥ 0.05)")
    print()

    # ======================================================================
    # PERFORMANCE BY QUALITY SCORE RANGE
    # ======================================================================

    print("10. Performance by quality score range...")
    print()

    score_bins = [0.75, 0.80, 0.85, 0.90, 1.0]
    for i in range(len(score_bins) - 1):
        bin_data = ultra_selective[
            (ultra_selective["quality_score"] >= score_bins[i])
            & (ultra_selective["quality_score"] < score_bins[i + 1])
        ]

        if len(bin_data) > 0:
            bin_wins = bin_data["bet_won"].sum()
            bin_losses = bin_data["bet_lost"].sum()
            bin_win_rate = bin_wins / (bin_wins + bin_losses) if (bin_wins + bin_losses) > 0 else 0
            print(
                f"   Quality {score_bins[i]:.2f}-{score_bins[i+1]:.2f}: {len(bin_data):4d} bets, {bin_win_rate*100:.1f}% win rate"
            )

    print()

    # ======================================================================
    # PERFORMANCE BY PRA RANGE
    # ======================================================================

    print("11. Performance by PRA range...")
    print()

    pra_bins = [(0, 15), (15, 20), (20, 30), (30, 40), (40, 100)]
    for low, high in pra_bins:
        range_data = ultra_selective[
            (ultra_selective["predicted_pra"] >= low) & (ultra_selective["predicted_pra"] < high)
        ]

        if len(range_data) > 0:
            range_wins = range_data["bet_won"].sum()
            range_losses = range_data["bet_lost"].sum()
            range_win_rate = (
                range_wins / (range_wins + range_losses) if (range_wins + range_losses) > 0 else 0
            )
            print(
                f"   PRA {low:2d}-{high:2d}: {len(range_data):4d} bets, {range_win_rate*100:.1f}% win rate"
            )

    print()

# ======================================================================
# COMPARISON TO BASELINE
# ======================================================================

print("=" * 80)
print("COMPARISON TO BASELINE (FIXED_V2)")
print("=" * 80)
print()

baseline_win_rate = 52.94
baseline_roi = 1.06

print(f"   Baseline (FIXED_V2):")
print(f"     Win rate: {baseline_win_rate:.2f}%")
print(f"     ROI: {baseline_roi:+.2f}%")
print(f"     MAE: 8.83 points")
print()

print(f"   Phase 1 (Overall):")
print(f"     Win rate: {win_rate_all*100:.2f}%")
print(f"     ROI: {roi_all:+.2f}%")
print(f"     MAE: 4.19 points")
print(f"     Improvement: {(win_rate_all*100 - baseline_win_rate):+.2f} pp")
print()

if len(ultra_selective) > 0:
    print(f"   Phase 1 (Ultra-Selective):")
    print(f"     Win rate: {ultra_win_rate*100:.2f}%")
    print(f"     ROI: {ultra_roi:+.2f}%")
    print(f"     Bets: {len(ultra_selective):,}")
    print(f"     Improvement: {(ultra_win_rate*100 - baseline_win_rate):+.2f} pp")
    print()

# ======================================================================
# SAVE RESULTS
# ======================================================================

print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Save all betting simulations
merged.to_csv(OUTPUT_PATH, index=False)
print(f"   ✅ All predictions: {OUTPUT_PATH}")

# Save ultra-selective bets
if len(ultra_selective) > 0:
    ultra_selective.to_csv(ULTRA_SELECTIVE_OUTPUT, index=False)
    print(f"   ✅ Ultra-selective bets: {ULTRA_SELECTIVE_OUTPUT}")

print()
print("=" * 80)
print("✅ PHASE 1 BETTING SIMULATION COMPLETE")
print("=" * 80)
