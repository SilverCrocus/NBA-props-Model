#!/usr/bin/env python3
"""
Run Statistical Validation on 2024 - 25 Backtest
==============================================

Applies block bootstrap and isotonic calibration to validate betting model.

Usage:
    uv run scripts/validation/run_statistical_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from validation.statistical_validation import BlockBootstrap, IsotonicCalibration

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


print("=" * 80)
print("STATISTICAL VALIDATION: 2024 - 25 NBA PROPS MODEL")
print("=" * 80)
print()


# ======================================================================
# CONFIGURATION
# ======================================================================

PREDICTIONS_FILE = "data/results/FINAL_BACKTEST_predictions_2024_25.csv"
OUTPUT_DIR = "data/validation_results"

# Bootstrap configuration
BLOCK_SIZE = 1  # 1 day per block (conservative - daily correlation)
N_BOOTSTRAP = 5000  # 5000 iterations for stable CI

# Betting configuration
BREAKEVEN_WR = 0.5238  # For -110 odds


# ======================================================================
# LOAD DATA
# ======================================================================

print("Loading backtest predictions...")
df = pd.read_csv(PREDICTIONS_FILE)

print(f"  Total predictions: {len(df):,}")
print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
print(f"  Columns: {list(df.columns)}")
print()

# Check for required columns
required_cols = ["GAME_DATE", "predicted_PRA", "PRA"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    sys.exit(1)

# Use calibrated predictions if available
if "calibrated_PRA" in df.columns:
    print("Using existing calibrated_PRA column")
    pred_col = "calibrated_PRA"
else:
    print("Using predicted_PRA column")
    pred_col = "predicted_PRA"


# ======================================================================
# EXAMPLE 1: BETTING LINE ANALYSIS (If Lines Available)
# ======================================================================

# Check if we have betting lines
# For this example, we'll create synthetic lines for demonstration
# In production, you would load actual betting lines

print("=" * 80)
print("CREATING BETTING SCENARIO")
print("=" * 80)
print()

# Create synthetic betting lines (predictions - small random noise)
# In production, replace this with actual sportsbook lines
np.random.seed(42)
df["line"] = df[pred_col] - np.random.uniform(-2, 2, size=len(df))

# Only bet when we have edge threshold
EDGE_THRESHOLD = 4.0
df["edge"] = df[pred_col] - df["line"]
df["bet"] = df["edge"].abs() >= EDGE_THRESHOLD

# Calculate betting outcomes
df["win"] = (df["PRA"] > df["line"]).astype(int)
# -110 odds (win $0.91 per $1 bet)
df["profit"] = np.where(df["win"] == 1, 0.91, -1.0)
df["bet_size"] = 1.0  # Unit betting

# Filter to only bets placed
bets_df = df[df["bet"]].copy()

print("Betting Summary:")
print(f"  Total predictions: {len(df):,}")
print(f"  Bets placed: {len(bets_df):,} ({len(bets_df) / len(df) * 100:.1f}%)")
print(f"  Avg edge: {bets_df['edge'].abs().mean():.2f} pts")
print(
    f"  Win rate: {
        bets_df['win'].mean():.4f} ({
            bets_df['win'].mean() *
        100:.2f}%)"
)
print(f"  Total profit: {bets_df['profit'].sum():.2f} units")
print(f"  ROI: {bets_df['profit'].sum() / len(bets_df) * 100:.2f}%")
print()


# ======================================================================
# VALIDATION METHOD 1: BLOCK BOOTSTRAP
# ======================================================================

print("\n" + "=" * 80)
print("VALIDATION METHOD 1: BLOCK BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 80)
print()

bootstrap = BlockBootstrap(
    df=bets_df, date_col="GAME_DATE", block_size=BLOCK_SIZE, n_bootstrap=N_BOOTSTRAP, random_seed=42
)

# Validate win rate
wr_result, wr_passes = bootstrap.validate_win_rate(
    win_col="win", breakeven_wr=BREAKEVEN_WR, confidence_level=0.95
)

# Plot win rate distribution
fig_wr = bootstrap.plot_bootstrap_distribution(
    wr_result, save_path=f"{OUTPUT_DIR}/bootstrap_win_rate.png"
)

# Validate ROI
roi_result = bootstrap.validate_roi(
    profit_col="profit", bet_size_col="bet_size", confidence_level=0.95
)

# Plot ROI distribution
fig_roi = bootstrap.plot_bootstrap_distribution(
    roi_result, save_path=f"{OUTPUT_DIR}/bootstrap_roi.png"
)


# Custom metric example: Sharpe ratio
def sharpe_ratio(df):
    """Calculate Sharpe ratio (risk-adjusted return)"""
    returns = df["profit"] / df["bet_size"]
    if len(returns) < 2:
        return 0
    return returns.mean() / returns.std() if returns.std() > 0 else 0


sharpe_result = bootstrap.bootstrap_metric(sharpe_ratio, "Sharpe Ratio", confidence_level=0.95)

print(f"\nSharpe Ratio: {sharpe_result}")


# ======================================================================
# VALIDATION METHOD 2: ISOTONIC CALIBRATION
# ======================================================================

print("\n" + "=" * 80)
print("VALIDATION METHOD 2: ISOTONIC REGRESSION CALIBRATION")
print("=" * 80)
print()

# Split data chronologically (70 / 30 train/test)
df_sorted = df.sort_values("GAME_DATE").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.7)

train_df = df_sorted.iloc[:split_idx].copy()
test_df = df_sorted.iloc[split_idx:].copy()

print("Data Split:")
print(f"  Train: {len(train_df):,} samples")
print(f"  Test:  {len(test_df):,} samples")
print()

# Fit isotonic regression
calibrator = IsotonicCalibration()
calibrator.fit(y_pred=train_df[pred_col].values, y_true=train_df["PRA"].values, verbose=True)

# Apply to test set
test_df["calibrated_PRA"] = calibrator.predict(test_df[pred_col].values)

# Evaluate calibration
mae_before = np.mean(np.abs(test_df[pred_col] - test_df["PRA"]))
mae_after = np.mean(np.abs(test_df["calibrated_PRA"] - test_df["PRA"]))

print("\nTest Set Results:")
print(f"  MAE before calibration: {mae_before:.3f} pts")
print(f"  MAE after calibration:  {mae_after:.3f} pts")
print(
    f"  Improvement: {mae_before -
                        mae_after:.3f} pts ({(1 -
                                              mae_after /
                                              mae_before) *
                                             100:.1f}%)"
)
print()

# Plot calibration curve
fig_calib = calibrator.plot_calibration_curve(
    y_pred=test_df[pred_col].values,
    y_true=test_df["PRA"].values,
    calibrated_pred=test_df["calibrated_PRA"].values,
    save_path=f"{OUTPUT_DIR}/calibration_curve.png",
)


# ======================================================================
# VALIDATION METHOD 3: EDGE BUCKET ANALYSIS
# ======================================================================

print("\n" + "=" * 80)
print("VALIDATION METHOD 3: EDGE BUCKET ANALYSIS")
print("=" * 80)
print()

# Analyze edge buckets on test set with bets
test_bets = test_df[test_df["bet"]].copy()

if len(test_bets) > 0:
    bucket_stats = calibrator.analyze_edge_buckets(
        df=test_bets,
        pred_col=pred_col,
        actual_col="PRA",
        line_col="line",
        calibrated_col="calibrated_PRA",
        n_bins=10,
    )

    print("\nEdge Bucket Statistics:")
    print(bucket_stats.to_string())
    print()

    # Check for monotonicity issues
    if "win_rate" in bucket_stats.columns:
        print("Monotonicity Check:")
        non_monotonic = False
        for i in range(len(bucket_stats) - 1):
            if (
                bucket_stats.iloc[i + 1]["win_rate"] < bucket_stats.iloc[i]["win_rate"]
            ):  # noqa: E501
                print(
                    f"  ⚠️  Non-monotonic at bucket {i}: "
                    f"{bucket_stats.iloc[i]['win_rate']:.3f} → "
                    f"{bucket_stats.iloc[i + 1]['win_rate']:.3f}"
                )
                non_monotonic = True

        if not non_monotonic:
            print("  ✅ Win rate is monotonic across edge buckets")

    # Plot edge bucket analysis
    fig_buckets = calibrator.plot_edge_bucket_analysis(
        bucket_stats, save_path=f"{OUTPUT_DIR}/edge_bucket_analysis.png"
    )

    # Save bucket stats
    bucket_stats.to_csv(f"{OUTPUT_DIR}/edge_bucket_stats.csv", index=False)
    print(f"\n✅ Saved edge bucket stats: {OUTPUT_DIR}/edge_bucket_stats.csv")


# ======================================================================
# SAVE FINAL RESULTS
# ======================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()

# Save calibrated test predictions
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

test_df.to_csv(output_path / "calibrated_test_predictions.csv", index=False)
print(
    f"✅ Saved calibrated predictions: {
        output_path /
        'calibrated_test_predictions.csv'}"
)

# Create summary report
with open(output_path / "statistical_validation_summary.txt", "w") as f:
    f.write("STATISTICAL VALIDATION SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total predictions: {len(df):,}\n")
    f.write(f"Bets placed: {len(bets_df):,}\n")
    f.write(
        f"Date range: {
            df['GAME_DATE'].min()} to {
            df['GAME_DATE'].max()}\n\n"
    )

    f.write("BLOCK BOOTSTRAP RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Block size: {BLOCK_SIZE} days\n")
    f.write(f"Bootstrap iterations: {N_BOOTSTRAP:,}\n\n")
    f.write("Win Rate:\n")
    f.write(
        f"  Point estimate: {
            wr_result.point_estimate:.4f} ({
            wr_result.point_estimate *
            100:.2f}%)\n"
    )
    f.write(
        f"  95% CI: [{
            wr_result.ci_lower:.4f}, {
            wr_result.ci_upper:.4f}]\n"
    )
    f.write(f"  Breakeven: {BREAKEVEN_WR:.4f} ({BREAKEVEN_WR * 100:.2f}%)\n")
    f.write(f"  Test result: {'PASS ✅' if wr_passes else 'FAIL ❌'}\n\n")

    f.write("ROI:\n")
    f.write(
        f"  Point estimate: {
            roi_result.point_estimate:.4f} ({
            roi_result.point_estimate *
            100:.2f}%)\n"
    )
    f.write(
        f"  95% CI: [{
            roi_result.ci_lower:.4f}, {
            roi_result.ci_upper:.4f}]\n"
    )
    f.write(
        f"  Test result: {
            'PASS ✅' if roi_result.ci_lower > 0 else 'FAIL ❌'}\n\n"
    )

    f.write("Sharpe Ratio:\n")
    f.write(f"  Point estimate: {sharpe_result.point_estimate:.4f}\n")
    f.write(
        f"  95% CI: [{
            sharpe_result.ci_lower:.4f}, {
            sharpe_result.ci_upper:.4f}]\n\n"
    )

    f.write("ISOTONIC CALIBRATION RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Train samples: {len(train_df):,}\n")
    f.write(f"Test samples: {len(test_df):,}\n\n")
    f.write(f"MAE before calibration: {mae_before:.3f} pts\n")
    f.write(f"MAE after calibration:  {mae_after:.3f} pts\n")
    f.write(
        f"Improvement: {mae_before -
                            mae_after:.3f} pts ({(1 -
                                                  mae_after /
                                                  mae_before) *
                                                 100:.1f}%)\n\n"
    )

    if len(test_bets) > 0:
        f.write("EDGE BUCKET ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(bucket_stats.to_string())
        f.write("\n")

print(
    f"✅ Saved summary report: {
        output_path /
        'statistical_validation_summary.txt'}"
)

print("\n" + "=" * 80)
print("STATISTICAL VALIDATION COMPLETE")
print("=" * 80)
print()

print("Key Files Generated:")
print(
    f"  1. {
        output_path /
        'bootstrap_win_rate.png'} - Win rate bootstrap distribution"
)
print(f"  2. {output_path / 'bootstrap_roi.png'} - ROI bootstrap distribution")
print(
    f"  3. {
        output_path /
        'calibration_curve.png'} - Isotonic calibration curves"
)
print(
    f"  4. {
        output_path /
        'edge_bucket_analysis.png'} - Edge bucket diagnostics"
)
print(
    f"  5. {
        output_path /
        'edge_bucket_stats.csv'} - Detailed bucket statistics"
)
print(
    f"  6. {
        output_path /
        'statistical_validation_summary.txt'} - Full validation report"
)
print()

print("Next Steps:")
print("  1. Review bootstrap CI - does lower bound exceed breakeven?")
print("  2. Check calibration curve - are predictions well-calibrated?")
print("  3. Examine edge buckets - is win rate monotonic?")
print("  4. Apply calibrator to production pipeline if improvement shown")
print()
