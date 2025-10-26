#!/usr/bin/env python3
"""
Advanced Edge Bucket Analysis
==============================

Deep dive into edge bucket performance to identify:
1. Non-monotonic win rates
2. Optimal edge thresholds
3. Sample size requirements per bucket
4. Calibration quality by edge magnitude

Usage:
    uv run scripts/validation/analyze_edge_buckets.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("=" * 80)
print("ADVANCED EDGE BUCKET ANALYSIS")
print("=" * 80)
print()


# ======================================================================
# CONFIGURATION
# ======================================================================

PREDICTIONS_FILE = "data/results/FINAL_BACKTEST_predictions_2024_25.csv"
OUTPUT_DIR = "data/validation_results"

# Create synthetic lines for demonstration
# In production, replace with actual sportsbook lines
SYNTHETIC_LINES = True

# Edge bucket configuration
EDGE_BUCKETS = [0, 2, 4, 6, 8, 10, 15, 20, 100]  # Custom bucket edges
MIN_SAMPLES_PER_BUCKET = 30  # Minimum for statistical validity


# ======================================================================
# LOAD DATA
# ======================================================================

print("Loading predictions...")
df = pd.read_csv(PREDICTIONS_FILE)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

print(f"  Total predictions: {len(df):,}")
print(
    f"  Date range: {
        df['GAME_DATE'].min().date()} to {
            df['GAME_DATE'].max().date()}"
)
print()

# Use calibrated predictions if available
if "calibrated_PRA" in df.columns:
    pred_col = "calibrated_PRA"
    print("Using calibrated_PRA column")
else:
    pred_col = "predicted_PRA"
    print("Using predicted_PRA column")
print()


# ======================================================================
# CREATE BETTING SCENARIO
# ======================================================================

if SYNTHETIC_LINES:
    print("Creating synthetic betting lines...")
    # In production, load actual lines from sportsbook API
    np.random.seed(42)
    df["line"] = df[pred_col] - np.random.uniform(-3, 3, size=len(df))
    print("  Note: Using synthetic lines for demonstration")
    print()

# Calculate edge and betting outcome
df["edge"] = df[pred_col] - df["line"]
df["abs_edge"] = df["edge"].abs()
df["win"] = (df["PRA"] > df["line"]).astype(int)
df["profit"] = np.where(df["win"] == 1, 0.91, -1.0)  # -110 odds


# ======================================================================
# BUCKET ANALYSIS
# ======================================================================

print("=" * 80)
print("EDGE BUCKET ANALYSIS")
print("=" * 80)
print()

# Create edge buckets
df["edge_bucket"] = pd.cut(
    df["abs_edge"],
    bins=EDGE_BUCKETS,
    labels=[
        f"{EDGE_BUCKETS[i]:.0f}-{EDGE_BUCKETS[i + 1]:.0f}" for i in range(len(EDGE_BUCKETS) - 1)
    ],  # noqa: E501
)

# Aggregate by bucket
bucket_stats = (
    df.groupby("edge_bucket", observed=True)
    .agg(
        {
            "abs_edge": ["mean", "std", "count"],
            "win": ["mean", "std"],
            "profit": ["sum", "mean"],
            pred_col: "mean",
            "PRA": "mean",
            "line": "mean",
        }
    )
    .reset_index()
)

# Flatten multi-index columns
bucket_stats.columns = ["_".join(col).strip("_") for col in bucket_stats.columns.values]

# Calculate additional metrics
bucket_stats["mae"] = (
    df.groupby("edge_bucket", observed=True)
    .apply(lambda x: np.mean(np.abs(x[pred_col] - x["PRA"])))
    .values
)

bucket_stats["bias"] = (
    df.groupby("edge_bucket", observed=True).apply(lambda x: np.mean(x[pred_col] - x["PRA"])).values
)

bucket_stats["roi"] = bucket_stats["profit_mean"] * 100

# Calculate standard error for win rate (for significance testing)
bucket_stats["win_se"] = bucket_stats["win_std"] / np.sqrt(bucket_stats["abs_edge_count"])

# Calculate 95% CI for win rate
bucket_stats["win_ci_lower"] = bucket_stats["win_mean"] - 1.96 * bucket_stats["win_se"]
bucket_stats["win_ci_upper"] = bucket_stats["win_mean"] + 1.96 * bucket_stats["win_se"]

# Flag buckets with insufficient samples
bucket_stats["sufficient_samples"] = (
    bucket_stats["abs_edge_count"] >= MIN_SAMPLES_PER_BUCKET
)  # noqa: E501

print("Edge Bucket Statistics:")
print("=" * 80)
print(
    bucket_stats[
        [
            "edge_bucket",
            "abs_edge_count",
            "abs_edge_mean",
            "win_mean",
            "win_ci_lower",
            "win_ci_upper",
            "roi",
            "mae",
            "bias",
            "sufficient_samples",
        ]
    ].to_string(index=False)
)
print()


# ======================================================================
# MONOTONICITY ANALYSIS
# ======================================================================

print("=" * 80)
print("MONOTONICITY CHECK")
print("=" * 80)
print()

non_monotonic_issues = []

for i in range(len(bucket_stats) - 1):
    current_wr = bucket_stats.iloc[i]["win_mean"]
    next_wr = bucket_stats.iloc[i + 1]["win_mean"]

    current_bucket = bucket_stats.iloc[i]["edge_bucket"]
    next_bucket = bucket_stats.iloc[i + 1]["edge_bucket"]

    if next_wr < current_wr:
        # Check if statistically significant (overlapping CIs?)
        current_ci_lower = bucket_stats.iloc[i]["win_ci_lower"]
        current_ci_upper = bucket_stats.iloc[i]["win_ci_upper"]
        next_ci_lower = bucket_stats.iloc[i + 1]["win_ci_lower"]
        next_ci_upper = bucket_stats.iloc[i + 1]["win_ci_upper"]

        # CIs overlap = not statistically significant
        overlap = (current_ci_lower <= next_ci_upper) and (next_ci_lower <= current_ci_upper)

        issue = {
            "bucket_pair": f"{current_bucket} → {next_bucket}",
            "wr_drop": current_wr - next_wr,
            "statistically_significant": not overlap,
            "current_wr": current_wr,
            "next_wr": next_wr,
            "current_samples": bucket_stats.iloc[i]["abs_edge_count"],
            "next_samples": bucket_stats.iloc[i + 1]["abs_edge_count"],
        }

        non_monotonic_issues.append(issue)

        if overlap:
            print(
                f"⚠️  Non-monotonic (NOT significant): {current_bucket} → {next_bucket}"
            )  # noqa: E501
        else:
            print(f"❌ Non-monotonic (SIGNIFICANT): {current_bucket} → {next_bucket}")  # noqa: E501

        print(
            f"   WR: {
                current_wr:.4f} → {
                next_wr:.4f} (drop: {
                current_wr -
                next_wr:.4f})"
        )
        print(
            f"   Samples: {bucket_stats.iloc[i]['abs_edge_count']} → {bucket_stats.iloc[i + 1]['abs_edge_count']}"  # noqa: E501
        )
        print()

if not non_monotonic_issues:
    print("✅ Win rate is monotonic across all edge buckets")
    print()
else:
    print(f"Total non-monotonic pairs: {len(non_monotonic_issues)}")
    print()


# ======================================================================
# OPTIMAL EDGE THRESHOLD
# ======================================================================

print("=" * 80)
print("OPTIMAL EDGE THRESHOLD ANALYSIS")
print("=" * 80)
print()

# Calculate cumulative statistics at different edge thresholds
edge_thresholds = np.arange(0, 15.1, 0.5)
threshold_stats = []

for threshold in edge_thresholds:
    bets = df[df["abs_edge"] >= threshold]

    if len(bets) == 0:
        continue

    stats_dict = {
        "threshold": threshold,
        "n_bets": len(bets),
        "win_rate": bets["win"].mean(),
        "roi": bets["profit"].mean() * 100,
        "total_profit": bets["profit"].sum(),
        "avg_edge": bets["abs_edge"].mean(),
        "mae": np.mean(np.abs(bets[pred_col] - bets["PRA"])),
    }

    threshold_stats.append(stats_dict)

threshold_df = pd.DataFrame(threshold_stats)

# Find optimal threshold (maximize total profit with sufficient samples)
min_bets_for_validity = 100
valid_thresholds = threshold_df[threshold_df["n_bets"] >= min_bets_for_validity]

if len(valid_thresholds) > 0:
    optimal_profit_idx = valid_thresholds["total_profit"].idxmax()
    optimal_roi_idx = valid_thresholds["roi"].idxmax()

    print("Optimal Thresholds:")
    print("\nBy Total Profit:")
    print(
        f"  Threshold: {valid_thresholds.loc[optimal_profit_idx, 'threshold']:.1f} pts"
    )  # noqa: E501
    print(f"  Win Rate: {valid_thresholds.loc[optimal_profit_idx, 'win_rate']:.4f}")  # noqa: E501
    print(f"  ROI: {valid_thresholds.loc[optimal_profit_idx, 'roi']:.2f}%")
    print(
        f"  Total Profit: {valid_thresholds.loc[optimal_profit_idx, 'total_profit']:.2f} units"
    )  # noqa: E501
    print(f"  Bets: {valid_thresholds.loc[optimal_profit_idx, 'n_bets']:.0f}")

    print("\nBy ROI:")
    print(
        f"  Threshold: {valid_thresholds.loc[optimal_roi_idx, 'threshold']:.1f} pts"
    )  # noqa: E501
    print(f"  Win Rate: {valid_thresholds.loc[optimal_roi_idx, 'win_rate']:.4f}")
    print(f"  ROI: {valid_thresholds.loc[optimal_roi_idx, 'roi']:.2f}%")
    print(
        f"  Total Profit: {valid_thresholds.loc[optimal_roi_idx, 'total_profit']:.2f} units"
    )  # noqa: E501
    print(f"  Bets: {valid_thresholds.loc[optimal_roi_idx, 'n_bets']:.0f}")
    print()


# ======================================================================
# VISUALIZATION
# ======================================================================

print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"

# ========================================
# Plot 1: Win Rate by Edge Bucket
# ========================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot win rate with error bars
ax.errorbar(
    range(len(bucket_stats)),
    bucket_stats["win_mean"],
    yerr=1.96 * bucket_stats["win_se"],  # 95% CI
    fmt="o-",
    linewidth=2,
    markersize=8,
    capsize=5,
    color="steelblue",
    label="Win Rate (95% CI)",
)

# Breakeven line
ax.axhline(0.5238, color="red", linestyle="--", linewidth=2, label="Breakeven (-110)")

# Mark insufficient sample buckets
insufficient = bucket_stats[~bucket_stats["sufficient_samples"]]
if len(insufficient) > 0:
    insufficient_idx = [i for i, x in enumerate(bucket_stats["sufficient_samples"]) if not x]
    ax.scatter(
        insufficient_idx,
        insufficient["win_mean"],
        s=200,
        facecolors="none",
        edgecolors="orange",
        linewidths=2,
        label=f"< {MIN_SAMPLES_PER_BUCKET} samples",
    )

ax.set_xlabel("Edge Bucket", fontsize=12)
ax.set_ylabel("Win Rate", fontsize=12)
ax.set_title("Win Rate by Edge Bucket (with 95% Confidence Intervals)", fontsize=14)
ax.set_xticks(range(len(bucket_stats)))
ax.set_xticklabels(bucket_stats["edge_bucket"], rotation=45, ha="right")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / "edge_bucket_win_rate.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_path / 'edge_bucket_win_rate.png'}")

# ========================================
# Plot 2: Threshold Performance
# ========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Win rate vs threshold
ax = axes[0, 0]
ax.plot(threshold_df["threshold"], threshold_df["win_rate"], linewidth=2, color="steelblue")
ax.axhline(0.5238, color="red", linestyle="--", label="Breakeven")
ax.set_xlabel("Edge Threshold (pts)", fontsize=11)
ax.set_ylabel("Win Rate", fontsize=11)
ax.set_title("Win Rate vs Edge Threshold", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# ROI vs threshold
ax = axes[0, 1]
ax.plot(threshold_df["threshold"], threshold_df["roi"], linewidth=2, color="green")
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("Edge Threshold (pts)", fontsize=11)
ax.set_ylabel("ROI (%)", fontsize=11)
ax.set_title("ROI vs Edge Threshold", fontsize=12)
ax.grid(True, alpha=0.3)

# Number of bets vs threshold
ax = axes[1, 0]
ax.plot(threshold_df["threshold"], threshold_df["n_bets"], linewidth=2, color="orange")
ax.axhline(
    min_bets_for_validity, color="red", linestyle="--", label=f"Min ({min_bets_for_validity})"
)
ax.set_xlabel("Edge Threshold (pts)", fontsize=11)
ax.set_ylabel("Number of Bets", fontsize=11)
ax.set_title("Sample Size vs Edge Threshold", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Total profit vs threshold
ax = axes[1, 1]
ax.plot(threshold_df["threshold"], threshold_df["total_profit"], linewidth=2, color="purple")
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.set_xlabel("Edge Threshold (pts)", fontsize=11)
ax.set_ylabel("Total Profit (units)", fontsize=11)
ax.set_title("Total Profit vs Edge Threshold", fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / "threshold_optimization.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_path / 'threshold_optimization.png'}")

# ========================================
# Plot 3: Calibration Quality by Edge
# ========================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MAE by edge bucket
ax = axes[0]
ax.bar(range(len(bucket_stats)), bucket_stats["mae"], alpha=0.7, color="steelblue")
ax.set_xlabel("Edge Bucket", fontsize=12)
ax.set_ylabel("MAE (pts)", fontsize=12)
ax.set_title("Prediction Error by Edge Bucket", fontsize=14)
ax.set_xticks(range(len(bucket_stats)))
ax.set_xticklabels(bucket_stats["edge_bucket"], rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")

# Bias by edge bucket
ax = axes[1]
colors = ["red" if x < 0 else "green" for x in bucket_stats["bias"]]
ax.bar(range(len(bucket_stats)), bucket_stats["bias"], alpha=0.7, color=colors)
ax.axhline(0, color="black", linestyle="-", linewidth=1)
ax.set_xlabel("Edge Bucket", fontsize=12)
ax.set_ylabel("Bias (pts)", fontsize=12)
ax.set_title(
    "Prediction Bias by Edge Bucket\n(Negative = Overconfident, Positive = Underconfident)",  # noqa: E501
    fontsize=14,
)
ax.set_xticks(range(len(bucket_stats)))
ax.set_xticklabels(bucket_stats["edge_bucket"], rotation=45, ha="right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_path / "calibration_by_edge.png", dpi=300, bbox_inches="tight")
print(f"✅ Saved: {output_path / 'calibration_by_edge.png'}")


# ======================================================================
# SAVE RESULTS
# ======================================================================

# Save bucket statistics
bucket_stats.to_csv(output_path / "edge_bucket_detailed_stats.csv", index=False)
print(f"✅ Saved: {output_path / 'edge_bucket_detailed_stats.csv'}")

# Save threshold analysis
threshold_df.to_csv(output_path / "edge_threshold_optimization.csv", index=False)
print(f"✅ Saved: {output_path / 'edge_threshold_optimization.csv'}")

# Create summary report
with open(output_path / "edge_bucket_report.txt", "w") as f:
    f.write("EDGE BUCKET ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total predictions: {len(df):,}\n")
    f.write(
        f"Date range: {
            df['GAME_DATE'].min().date()} to {
            df['GAME_DATE'].max().date()}\n"
    )
    f.write(
        f"Overall win rate: {
            df['win'].mean():.4f} ({
            df['win'].mean() *
            100:.2f}%)\n"
    )
    f.write(f"Overall ROI: {df['profit'].mean() * 100:.2f}%\n\n")

    f.write("BUCKET STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(
        bucket_stats[["edge_bucket", "abs_edge_count", "win_mean", "roi", "mae", "bias"]].to_string(
            index=False
        )
    )
    f.write("\n\n")

    f.write("MONOTONICITY ISSUES\n")
    f.write("-" * 80 + "\n")
    if non_monotonic_issues:
        for issue in non_monotonic_issues:
            f.write(
                f"{issue['bucket_pair']}: "
                f"WR drop = {issue['wr_drop']:.4f}, "
                f"Significant = {issue['statistically_significant']}\n"
            )
    else:
        f.write("No monotonicity issues detected.\n")
    f.write("\n")

    f.write("OPTIMAL THRESHOLDS\n")
    f.write("-" * 80 + "\n")
    if len(valid_thresholds) > 0:
        f.write(
            f"\nBy Total Profit: {valid_thresholds.loc[optimal_profit_idx, 'threshold']:.1f} pts\n"  # noqa: E501
        )
        f.write(
            f"  Win Rate: {valid_thresholds.loc[optimal_profit_idx, 'win_rate']:.4f}\n"
        )  # noqa: E501
        f.write(f"  ROI: {valid_thresholds.loc[optimal_profit_idx, 'roi']:.2f}%\n")
        f.write(
            f"  Profit: {valid_thresholds.loc[optimal_profit_idx, 'total_profit']:.2f} units\n"
        )  # noqa: E501

        f.write(
            f"\nBy ROI: {valid_thresholds.loc[optimal_roi_idx, 'threshold']:.1f} pts\n"
        )  # noqa: E501
        f.write(
            f"  Win Rate: {valid_thresholds.loc[optimal_roi_idx, 'win_rate']:.4f}\n"
        )  # noqa: E501
        f.write(f"  ROI: {valid_thresholds.loc[optimal_roi_idx, 'roi']:.2f}%\n")
        f.write(
            f"  Profit: {valid_thresholds.loc[optimal_roi_idx, 'total_profit']:.2f} units\n"
        )  # noqa: E501

print(f"✅ Saved: {output_path / 'edge_bucket_report.txt'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()

print("Key Findings:")
print(f"  1. Monotonicity issues: {len(non_monotonic_issues)}")
print(
    f"  2. Buckets with insufficient samples: {(~bucket_stats['sufficient_samples']).sum()}"
)  # noqa: E501
if len(valid_thresholds) > 0:
    print(
        f"  3. Optimal threshold (profit): {valid_thresholds.loc[optimal_profit_idx, 'threshold']:.1f} pts"  # noqa: E501
    )
    print(
        f"  4. Optimal threshold (ROI): {valid_thresholds.loc[optimal_roi_idx, 'threshold']:.1f} pts"  # noqa: E501
    )

print("\nRecommendations:")
if non_monotonic_issues:
    print("  - Apply isotonic calibration to fix non-monotonic win rates")
if (~bucket_stats["sufficient_samples"]).sum() > 0:
    print(
        f"  - Collect more data for high-edge buckets (< {MIN_SAMPLES_PER_BUCKET} samples)"
    )  # noqa: E501
if len(valid_thresholds) > 0:
    print(
        f"  - Consider using edge threshold of {valid_thresholds.loc[optimal_profit_idx, 'threshold']:.1f} pts for betting"  # noqa: E501
    )
print()
