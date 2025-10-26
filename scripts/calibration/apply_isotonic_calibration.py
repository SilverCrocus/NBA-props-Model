"""
Apply Isotonic Regression Calibration to Fix Non-Monotone Edge Buckets

Uses walk-forward validation data to:
1. Fit isotonic regression on predicted_PRA → actual_PRA
2. Save calibrator for production use
3. Analyze calibration improvement

Issue: 5 - 6 pt edge bucket underperforms (56.8% WR vs 68.6% for 4 - 5 pts)
Solution: Isotonic regression enforces monotonicity
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def load_walkforward_data():
    """Load walk-forward validation predictions"""
    df = pd.read_csv("data/results/walk_forward_leak_free_2024_25.csv")

    print(f"✅ Loaded {len(df):,} walk-forward predictions")
    print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    print(f"   Players: {df['PLAYER_NAME'].nunique()}")

    return df


def split_temporal(df, split_date="2025 - 02 - 01"):
    """
    Split data temporally for training isotonic calibrator

    Train on earlier data, validate on later data
    """
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    train = df[df["GAME_DATE"] < split_date].copy()
    test = df[df["GAME_DATE"] >= split_date].copy()

    print("\n📊 Temporal Split:")
    print(f"   Train: {len(train):,} predictions (before {split_date})")
    print(f"   Test:  {len(test):,} predictions (after {split_date})")

    return train, test


def fit_isotonic_calibrator(train_df):
    """
    Fit isotonic regression: predicted_PRA → calibrated_PRA

    Enforces monotonicity: higher predictions → higher calibrated values
    """
    print("\n🔧 Fitting Isotonic Regression...")

    X = train_df["predicted_PRA"].values
    y = train_df["PRA"].values

    # Fit isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)

    # Get calibrated predictions on training set
    train_calibrated = iso.predict(X)

    # Calculate improvement
    mae_before = np.abs(X - y).mean()
    mae_after = np.abs(train_calibrated - y).mean()

    print(f"   MAE before calibration: {mae_before:.2f} points")
    print(f"   MAE after calibration:  {mae_after:.2f} points")
    improvement_pct = (1 - mae_after / mae_before) * 100
    print(
        f"   Improvement: {(mae_before -
                              mae_after):.2f} points "
        f"({improvement_pct:.1f}%)"
    )  # noqa: E501

    return iso


def validate_calibrator(iso, test_df):
    """
    Validate calibrator on held-out test set
    """
    print("\n📊 Validating on Test Set...")

    X_test = test_df["predicted_PRA"].values
    y_test = test_df["PRA"].values

    # Get calibrated predictions
    test_calibrated = iso.predict(X_test)

    # Calculate metrics
    mae_before = np.abs(X_test - y_test).mean()
    mae_after = np.abs(test_calibrated - y_test).mean()

    # Correlation
    corr_before = np.corrcoef(X_test, y_test)[0, 1]
    corr_after = np.corrcoef(test_calibrated, y_test)[0, 1]

    mae_delta = mae_after - mae_before
    print(
        f"   MAE before: {
            mae_before:.2f} pts | after: {
            mae_after:.2f} pts "
        f"(Δ={
                mae_delta:+.2f})"
    )
    print(f"   Corr before: {corr_before:.3f} | after: {corr_after:.3f}")

    # Edge bucket analysis
    print("\n📈 Edge Bucket Analysis (Test Set):")

    for df_name, predictions in [("Before", X_test), ("After", test_calibrated)]:
        # Simulate betting with 3 - 7 pt edges
        betting_opps = []

        for i in range(len(test_df)):
            pred = predictions[i]
            actual = y_test[i]

            # Simulate lines around actual value
            for line_offset in [-5, -3, 0, 3, 5]:
                line = actual + line_offset
                edge = pred - line

                if 3 <= abs(edge) <= 7:
                    if edge > 0:  # OVER
                        won = 1 if actual > line else 0
                    else:  # UNDER
                        won = 1 if actual < line else 0

                    betting_opps.append({"abs_edge": abs(edge), "won": won})

        if len(betting_opps) > 0:
            bet_df = pd.DataFrame(betting_opps)

            # Edge buckets
            bins = [3, 4, 5, 6, 7, 10]
            labels = ["3 - 4", "4 - 5", "5 - 6", "6 - 7", "7 - 10"]
            bet_df["bucket"] = pd.cut(bet_df["abs_edge"], bins=bins, labels=labels)

            print(f"\n   {df_name} Calibration:")
            for bucket in labels:
                bucket_df = bet_df[bet_df["bucket"] == bucket]
                if len(bucket_df) > 10:
                    wr = bucket_df["won"].mean()
                    print(
                        f"     {bucket} pts: {
                            wr:.1%} ({
                            len(bucket_df):,        } bets)"
                    )

    return test_calibrated


def save_calibrator(iso, output_path="models/isotonic_calibrator.pkl"):
    """Save calibrator for production use"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(iso, f)

    print(f"\n✅ Calibrator saved to: {output_path}")

    # Create usage example
    usage_example = (
        "# Usage in Production:\n\n"
        "import pickle\n\n"
        "# Load calibrator\n"
        f"with open('{output_path}', 'rb') as f:\n"
        "    calibrator = pickle.load(f)\n\n"
        "# Calibrate predictions\n"
        "predictions_df['calibrated_PRA'] = "
        "calibrator.predict(predictions_df['predicted_PRA'])\n\n"
        "# Use calibrated values for edge calculation\n"
        "edge = calibrated_PRA - line\n"
    )

    with open(output_path.parent / "isotonic_usage.txt", "w") as f:
        f.write(usage_example)

    print(
        f"   Usage example saved to: {
            output_path.parent /
            'isotonic_usage.txt'}"
    )


def analyze_calibration_curve(iso, df, output_dir="data/validation_results"):
    """
    Analyze calibration curve and save diagnostic plots
    """
    print("\n📊 Analyzing Calibration Curve...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions and actuals
    X = df["predicted_PRA"].values
    y = df["PRA"].values

    # Calibrated predictions
    X_cal = iso.predict(X)

    # Create calibration curve plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Before vs After
    ax1 = axes[0]

    # Bin predictions
    bins = np.linspace(0, 80, 41)

    # Before calibration
    digitized = np.digitize(X, bins)
    bin_means_pred = [X[digitized == i].mean() for i in range(1, len(bins))]
    bin_means_actual = [y[digitized == i].mean() for i in range(1, len(bins))]

    ax1.scatter(bin_means_pred, bin_means_actual, alpha=0.6, label="Before", s=50)

    # After calibration
    digitized_cal = np.digitize(X_cal, bins)
    bin_means_cal = [X_cal[digitized_cal == i].mean() for i in range(1, len(bins))]
    bin_means_actual_cal = [y[digitized_cal == i].mean() for i in range(1, len(bins))]

    ax1.scatter(bin_means_cal, bin_means_actual_cal, alpha=0.6, label="After", s=50)

    # Perfect calibration line
    ax1.plot([0, 80], [0, 80], "k--", alpha=0.3, label="Perfect")

    ax1.set_xlabel("Predicted PRA")
    ax1.set_ylabel("Actual PRA")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Residuals
    ax2 = axes[1]

    residuals_before = X - y
    residuals_after = X_cal - y

    ax2.scatter(X, residuals_before, alpha=0.3, s=1, label="Before")
    ax2.scatter(X_cal, residuals_after, alpha=0.3, s=1, label="After")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Predicted PRA")
    ax2.set_ylabel("Residual (Pred - Actual)")
    ax2.set_title("Residual Plot")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "isotonic_calibration_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"   Calibration plots saved to: {plot_path}")

    plt.close()


def main():
    """
    Main calibration pipeline
    """
    print("=" * 80)
    print("ISOTONIC REGRESSION CALIBRATION")
    print("=" * 80)

    # Load data
    df = load_walkforward_data()

    # Temporal split
    train_df, test_df = split_temporal(df, split_date="2025 - 02 - 01")

    # Fit calibrator on training data
    iso_calibrator = fit_isotonic_calibrator(train_df)

    # Validate on test data
    _ = validate_calibrator(iso_calibrator, test_df)

    # Save calibrator
    save_calibrator(iso_calibrator, output_path="models/isotonic_calibrator.pkl")

    # Analyze calibration curve
    analyze_calibration_curve(iso_calibrator, test_df)

    print("\n" + "=" * 80)
    print("✅ ISOTONIC CALIBRATION COMPLETE")
    print("=" * 80)

    print("\n📋 Next Steps:")
    print("   1. Integrate calibrator into production pipeline")
    print("   2. Use calibrated_PRA for edge calculations")
    print("   3. Monitor edge-bucket monotonicity")

    print("\n💡 Usage:")
    print("   import pickle")
    print("   with open('models/isotonic_calibrator.pkl', 'rb') as f:")
    print("       calibrator = pickle.load(f)")
    print("   df['calibrated_PRA'] = calibrator.predict(df['predicted_PRA'])")


if __name__ == "__main__":
    main()
