#!/usr/bin/env python3
"""
Verify Edge Threshold Analysis
===============================

Triple-check the edge threshold analysis by examining actual bet-by-bet performance.
This will validate whether low edges (0-3 pts) truly have higher win rates.

Usage: uv run python scripts/betting/verify_edge_analysis.py
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("VERIFYING EDGE THRESHOLD ANALYSIS")
print("=" * 80)
print()

# ======================================================================
# LOAD DATA
# ======================================================================

print("Loading predictions with odds...")
predictions_df = pd.read_csv("data/results/predictions_ensemble_2024_25_with_odds.csv")
predictions_df["GAME_DATE"] = pd.to_datetime(predictions_df["GAME_DATE"])

print(f"✅ Loaded {len(predictions_df):,} predictions with odds")
print()

# ======================================================================
# CALCULATE EDGES AND RESULTS
# ======================================================================

print("Calculating edges for all predictions...")
print()

# Calculate edge for each prediction
predictions_df["edge"] = predictions_df["prediction"] - predictions_df["line"]

# Determine which side we would bet (OVER or UNDER)
predictions_df["bet_side"] = predictions_df["edge"].apply(lambda x: "OVER" if x > 0 else "UNDER")

# Determine if each side would have won
predictions_df["over_wins"] = predictions_df["actual"] > predictions_df["line"]
predictions_df["under_wins"] = predictions_df["actual"] < predictions_df["line"]

# Determine if our bet would have won (based on our edge direction)
predictions_df["would_win"] = predictions_df.apply(
    lambda row: row["over_wins"] if row["edge"] > 0 else row["under_wins"], axis=1
)

# ======================================================================
# ANALYZE BY EDGE MAGNITUDE BUCKETS
# ======================================================================

print("=" * 80)
print("WIN RATE BY EDGE MAGNITUDE (ALL PREDICTIONS)")
print("=" * 80)
print()

# Create edge magnitude buckets
edge_buckets = [
    (0, 1, "0-1 pts"),
    (1, 2, "1-2 pts"),
    (2, 3, "2-3 pts"),
    (3, 4, "3-4 pts"),
    (4, 5, "4-5 pts"),
    (5, 6, "5-6 pts"),
    (6, 7, "6-7 pts"),
    (7, 8, "7-8 pts"),
    (8, 10, "8-10 pts"),
    (10, 15, "10-15 pts"),
    (15, 100, "15+ pts"),
]

bucket_results = []

for min_edge, max_edge, label in edge_buckets:
    # Get predictions in this edge range (absolute value)
    mask = (predictions_df["edge"].abs() >= min_edge) & (predictions_df["edge"].abs() < max_edge)
    bucket_data = predictions_df[mask]

    if len(bucket_data) == 0:
        continue

    total = len(bucket_data)
    wins = bucket_data["would_win"].sum()
    win_rate = wins / total if total > 0 else 0

    # Separate OVER and UNDER
    over_mask = bucket_data["edge"] > 0
    under_mask = bucket_data["edge"] < 0

    over_total = over_mask.sum()
    over_wins = bucket_data[over_mask]["would_win"].sum()
    over_win_rate = over_wins / over_total if over_total > 0 else 0

    under_total = under_mask.sum()
    under_wins = bucket_data[under_mask]["would_win"].sum()
    under_win_rate = under_wins / under_total if under_total > 0 else 0

    bucket_results.append(
        {
            "edge_range": label,
            "total_predictions": total,
            "wins": wins,
            "win_rate": win_rate,
            "over_predictions": over_total,
            "over_wins": over_wins,
            "over_win_rate": over_win_rate,
            "under_predictions": under_total,
            "under_wins": under_wins,
            "under_win_rate": under_win_rate,
        }
    )

bucket_df = pd.DataFrame(bucket_results)

print("Overall Win Rate by Edge Magnitude:")
print("-" * 80)
for _, row in bucket_df.iterrows():
    print(
        f"{row['edge_range']:>12} | Total: {row['total_predictions']:4} | Wins: {row['wins']:4} | Win Rate: {row['win_rate']*100:5.1f}%"
    )
print()

print("OVER Bets by Edge Magnitude:")
print("-" * 80)
for _, row in bucket_df.iterrows():
    if row["over_predictions"] > 0:
        print(
            f"{row['edge_range']:>12} | Total: {row['over_predictions']:4} | Wins: {row['over_wins']:4} | Win Rate: {row['over_win_rate']*100:5.1f}%"
        )
print()

print("UNDER Bets by Edge Magnitude:")
print("-" * 80)
for _, row in bucket_df.iterrows():
    if row["under_predictions"] > 0:
        print(
            f"{row['edge_range']:>12} | Total: {row['under_predictions']:4} | Wins: {row['under_wins']:4} | Win Rate: {row['under_win_rate']*100:5.1f}%"
        )
print()

# ======================================================================
# CUMULATIVE ANALYSIS (THRESHOLD APPROACH)
# ======================================================================

print("=" * 80)
print("CUMULATIVE WIN RATE BY MINIMUM EDGE THRESHOLD")
print("=" * 80)
print()

thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

threshold_results = []

for threshold in thresholds:
    # Get all predictions with edge >= threshold
    mask = predictions_df["edge"].abs() >= threshold
    threshold_data = predictions_df[mask]

    if len(threshold_data) == 0:
        continue

    total = len(threshold_data)
    wins = threshold_data["would_win"].sum()
    win_rate = wins / total if total > 0 else 0

    # Separate OVER and UNDER
    over_mask = threshold_data["edge"] > 0
    under_mask = threshold_data["edge"] < 0

    over_total = over_mask.sum()
    over_wins = threshold_data[over_mask]["would_win"].sum()
    over_win_rate = over_wins / over_total if over_total > 0 else 0

    under_total = under_mask.sum()
    under_wins = threshold_data[under_mask]["would_win"].sum()
    under_win_rate = under_wins / under_total if under_total > 0 else 0

    threshold_results.append(
        {
            "min_edge": threshold,
            "total_predictions": total,
            "wins": wins,
            "win_rate": win_rate,
            "over_predictions": over_total,
            "over_wins": over_wins,
            "over_win_rate": over_win_rate,
            "under_predictions": under_total,
            "under_wins": under_wins,
            "under_win_rate": under_win_rate,
        }
    )

threshold_df = pd.DataFrame(threshold_results)

print("Win Rate When Betting All Predictions >= Threshold:")
print("-" * 80)
for _, row in threshold_df.iterrows():
    print(
        f"≥ {row['min_edge']:2.0f} pts | Total: {row['total_predictions']:4} | Wins: {row['wins']:4} | Win Rate: {row['win_rate']*100:5.1f}%"
    )
print()

# ======================================================================
# SPECIFIC RANGE ANALYSIS (0-3 pts vs others)
# ======================================================================

print("=" * 80)
print("SPECIFIC RANGE COMPARISON (0-3 pts vs others)")
print("=" * 80)
print()

# 0-3 pts range
mask_0_3 = predictions_df["edge"].abs() < 3
data_0_3 = predictions_df[mask_0_3]
total_0_3 = len(data_0_3)
wins_0_3 = data_0_3["would_win"].sum()
win_rate_0_3 = wins_0_3 / total_0_3 if total_0_3 > 0 else 0

# 3-6 pts range
mask_3_6 = (predictions_df["edge"].abs() >= 3) & (predictions_df["edge"].abs() < 6)
data_3_6 = predictions_df[mask_3_6]
total_3_6 = len(data_3_6)
wins_3_6 = data_3_6["would_win"].sum()
win_rate_3_6 = wins_3_6 / total_3_6 if total_3_6 > 0 else 0

# 6-10 pts range
mask_6_10 = (predictions_df["edge"].abs() >= 6) & (predictions_df["edge"].abs() < 10)
data_6_10 = predictions_df[mask_6_10]
total_6_10 = len(data_6_10)
wins_6_10 = data_6_10["would_win"].sum()
win_rate_6_10 = wins_6_10 / total_6_10 if total_6_10 > 0 else 0

# 10+ pts range
mask_10_plus = predictions_df["edge"].abs() >= 10
data_10_plus = predictions_df[mask_10_plus]
total_10_plus = len(data_10_plus)
wins_10_plus = data_10_plus["would_win"].sum()
win_rate_10_plus = wins_10_plus / total_10_plus if total_10_plus > 0 else 0

print(
    f"0-3 pts edge:   {total_0_3:4} predictions | {wins_0_3:4} wins | {win_rate_0_3*100:5.1f}% win rate"
)
print(
    f"3-6 pts edge:   {total_3_6:4} predictions | {wins_3_6:4} wins | {win_rate_3_6*100:5.1f}% win rate"
)
print(
    f"6-10 pts edge:  {total_6_10:4} predictions | {wins_6_10:4} wins | {win_rate_6_10*100:5.1f}% win rate"
)
print(
    f"10+ pts edge:   {total_10_plus:4} predictions | {wins_10_plus:4} wins | {win_rate_10_plus*100:5.1f}% win rate"
)
print()

# ======================================================================
# EXAMINE SMALL EDGES IN DETAIL
# ======================================================================

print("=" * 80)
print("DETAILED ANALYSIS: SMALL EDGES (0-3 pts)")
print("=" * 80)
print()

small_edge_buckets = [
    (0, 0.5, "0.0-0.5 pts"),
    (0.5, 1.0, "0.5-1.0 pts"),
    (1.0, 1.5, "1.0-1.5 pts"),
    (1.5, 2.0, "1.5-2.0 pts"),
    (2.0, 2.5, "2.0-2.5 pts"),
    (2.5, 3.0, "2.5-3.0 pts"),
]

print("Small Edge Performance:")
print("-" * 80)

for min_edge, max_edge, label in small_edge_buckets:
    mask = (predictions_df["edge"].abs() >= min_edge) & (predictions_df["edge"].abs() < max_edge)
    bucket_data = predictions_df[mask]

    if len(bucket_data) == 0:
        continue

    total = len(bucket_data)
    wins = bucket_data["would_win"].sum()
    win_rate = wins / total if total > 0 else 0

    print(f"{label:>15} | Total: {total:4} | Wins: {wins:4} | Win Rate: {win_rate*100:5.1f}%")

print()

# ======================================================================
# STATISTICAL SIGNIFICANCE TEST
# ======================================================================

print("=" * 80)
print("STATISTICAL SIGNIFICANCE")
print("=" * 80)
print()

from scipy import stats

# Compare 0-3 pts vs 3+ pts
win_rate_0_3_pct = win_rate_0_3
win_rate_3_plus = (wins_3_6 + wins_6_10 + wins_10_plus) / (total_3_6 + total_6_10 + total_10_plus)

# Z-test for proportions
n1 = total_0_3
n2 = total_3_6 + total_6_10 + total_10_plus
p1 = win_rate_0_3
p2 = win_rate_3_plus

# Pooled proportion
p_pooled = (wins_0_3 + wins_3_6 + wins_6_10 + wins_10_plus) / (n1 + n2)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
z = (p1 - p2) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"0-3 pts edge win rate:   {p1*100:.2f}% (n={n1})")
print(f"3+ pts edge win rate:    {p2*100:.2f}% (n={n2})")
print(f"Difference:              {(p1-p2)*100:.2f} percentage points")
print(f"Z-score:                 {z:.3f}")
print(f"P-value:                 {p_value:.4f}")
print()

if p_value < 0.05:
    if p1 > p2:
        print("✅ 0-3 pts edges are SIGNIFICANTLY BETTER (p < 0.05)")
    else:
        print("✅ 3+ pts edges are SIGNIFICANTLY BETTER (p < 0.05)")
else:
    print("⚠️  No statistically significant difference (p >= 0.05)")

print()

# ======================================================================
# SAVE DETAILED RESULTS
# ======================================================================

print("Saving detailed analysis...")

# Save bucket analysis
bucket_df.to_csv("data/results/edge_bucket_analysis.csv", index=False)
print("✅ Saved bucket analysis to data/results/edge_bucket_analysis.csv")

# Save threshold analysis
threshold_df.to_csv("data/results/edge_threshold_win_rates.csv", index=False)
print("✅ Saved threshold analysis to data/results/edge_threshold_win_rates.csv")

# Save predictions with edge analysis
predictions_df.to_csv("data/results/predictions_with_edge_analysis.csv", index=False)
print("✅ Saved detailed predictions to data/results/predictions_with_edge_analysis.csv")

print()
print("=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)
