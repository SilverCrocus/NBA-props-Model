"""
Quantify Diagnostic Impact
Validates root cause analysis with empirical estimates

Author: Data Science Team
Date: 2025 - 10 - 25
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def load_backtest_data():
    """Load backtest results"""
    data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "results"
        / "backtest_walkforward_2024_25.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"Backtest file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} bets from backtest")
    return df


def calculate_basic_metrics(df):
    """Calculate overall performance metrics"""
    # Win rate
    wr = df["win"].mean()

    # ROI
    profit = (df["win"] * (df["entry_odds"] - 1)) - (1 - df["win"])
    roi = profit.sum() / df["stake"].sum() * 100

    # Breakeven
    breakeven = df.apply(lambda x: 1 / x["entry_odds"], axis=1).mean()

    # OWWR (odds-weighted win rate)
    owwr = (df["win"] * df["stake"]).sum() / df["stake"].sum()

    return {
        "win_rate": wr,
        "roi": roi,
        "breakeven": breakeven,
        "owwr": owwr,
        "owwr_gap": owwr - wr,
        "total_profit": profit.sum(),
        "total_staked": df["stake"].sum(),
    }


def analyze_side_bias(df):
    """Quantify side bias impact"""
    print("\n" + "=" * 80)
    print("SIDE BIAS ANALYSIS")
    print("=" * 80)

    overs = df[df["side"] == "OVER"]
    unders = df[df["side"] == "UNDER"]

    over_wr = overs["win"].mean()
    under_wr = unders["win"].mean()
    wr_gap = over_wr - under_wr

    # ROI by side
    def calc_roi(side_df):
        profit = (side_df["win"] * (side_df["entry_odds"] - 1)) - (1 - side_df["win"])
        return profit.sum() / side_df["stake"].sum() * 100

    over_roi = calc_roi(overs)
    under_roi = calc_roi(unders)
    roi_gap = over_roi - under_roi

    # Portfolio-weighted ROI
    over_weight = len(overs) / len(df)
    under_weight = len(unders) / len(df)
    portfolio_roi = over_weight * over_roi + under_weight * under_roi

    print(f"\nOVER Win Rate:  {over_wr:.2%} ({len(overs)} bets)")
    print(f"UNDER Win Rate: {under_wr:.2%} ({len(unders)} bets)")
    print(f"Win Rate Gap:   {wr_gap:.2%}")
    print(f"\nOVER ROI:  {over_roi:+.2f}%")
    print(f"UNDER ROI: {under_roi:+.2f}%")
    print(f"ROI Gap:   {roi_gap:+.2f}%")
    print(f"\nPortfolio-Weighted ROI: {portfolio_roi:+.2f}%")
    print(f"Actual Overall ROI: {calc_roi(df):+.2f}%")

    # Estimate impact
    # If we could fix UNDER to match OVER performance
    fixed_under_roi = over_roi * 0.8  # Conservative estimate
    improved_portfolio_roi = over_weight * over_roi + under_weight * fixed_under_roi  # noqa: E501
    side_bias_impact = improved_portfolio_roi - portfolio_roi

    print("\n--- ESTIMATED SIDE BIAS IMPACT ---")
    print("If UNDER fixed to 80% of OVER ROI:")
    print(f"  Current Portfolio ROI: {portfolio_roi:+.2f}%")
    print(f"  Fixed Portfolio ROI:   {improved_portfolio_roi:+.2f}%")
    print(f"  Potential Gain:        {side_bias_impact:+.2f}% ROI")

    return {
        "over_wr": over_wr,
        "under_wr": under_wr,
        "wr_gap": wr_gap,
        "over_roi": over_roi,
        "under_roi": under_roi,
        "roi_gap": roi_gap,
        "estimated_impact": side_bias_impact,
    }


def analyze_calibration(df):
    """Quantify calibration impact"""
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)

    # Calculate predicted probabilities
    df["pred_prob"] = df.apply(
        lambda row: (
            norm.cdf((row["predicted_pra"] - row["line"]) / row["player_sigma"])
            if row["side"] == "OVER"
            else norm.cdf((row["line"] - row["predicted_pra"]) / row["player_sigma"])
        ),
        axis=1,
    )

    # Bin probabilities
    bins = [0.5, 0.55, 0.60, 0.65, 0.70, 1.0]
    labels = ["50 - 55%", "55 - 60%", "60 - 65%", "65 - 70%", "70%+"]
    df["prob_bin"] = pd.cut(df["pred_prob"], bins=bins, labels=labels)

    # Calibration table
    print("\nCalibration Table:")
    print("-" * 80)
    print(
        f"{
            'Prob Bin':<12} {
            'Pred Prob':<12} {
                'Obs WR':<12} {
                    'Error':<12} {
                        'Count':<12} {
                            'ROI':<12}"
    )
    print("-" * 80)

    total_calib_loss = 0

    for bin_label in labels:
        bin_data = df[df["prob_bin"] == bin_label]
        if len(bin_data) == 0:
            continue

        pred_prob = bin_data["pred_prob"].mean()
        obs_wr = bin_data["win"].mean()
        error = pred_prob - obs_wr

        # ROI for this bin
        profit = (bin_data["win"] * (bin_data["entry_odds"] - 1)) - (1 - bin_data["win"])
        roi = profit.sum() / bin_data["stake"].sum() * 100

        # Estimate loss from miscalibration
        # If we bet expected value based on pred_prob, but win at obs_wr
        expected_roi = (
            pred_prob * (bin_data["entry_odds"].mean() - 1) - (1 - pred_prob)
        ) * 100  # noqa: E501
        calib_loss = roi - expected_roi
        total_calib_loss += (len(bin_data) / len(df)) * calib_loss

        print(
            f"{
                bin_label:<12} {
                pred_prob:<12.2%} {
                obs_wr:<12.2%} {
                    error:<12.2%} {
                        len(bin_data):<12} {
                            roi:<12.2f}%"
        )

    # Expected Calibration Error
    ece = 0
    for bin_label in labels:
        bin_data = df[df["prob_bin"] == bin_label]
        if len(bin_data) > 0:
            weight = len(bin_data) / len(df)
            pred_mean = bin_data["pred_prob"].mean()
            obs_mean = bin_data["win"].mean()
            ece += weight * abs(pred_mean - obs_mean)

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Estimated ROI Loss from Miscalibration: {total_calib_loss:+.2f}%")

    return {"ece": ece, "estimated_impact": total_calib_loss}


def analyze_edge_monotonicity(df):
    """Quantify edge monotonicity impact"""
    print("\n" + "=" * 80)
    print("EDGE MONOTONICITY ANALYSIS")
    print("=" * 80)

    # Model edge
    df["model_edge"] = np.abs(df["predicted_pra"] - df["line"])

    # Edge buckets
    bins = [0, 2, 4, 6, 8, np.inf]
    labels = ["0 - 2", "2 - 4", "4 - 6", "6 - 8", "8+"]
    df["edge_bucket"] = pd.cut(df["model_edge"], bins=bins, labels=labels)

    print("\nEdge Bucket Analysis:")
    print("-" * 80)
    print(
        f"{
            'Edge Bucket':<12} {
            'Win Rate':<12} {
                'ROI':<12} {
                    'Count':<12} {
                        'Avg Stake':<12}"
    )
    print("-" * 80)

    bucket_results = []

    for bucket in labels:
        bucket_data = df[df["edge_bucket"] == bucket]
        if len(bucket_data) == 0:
            continue

        wr = bucket_data["win"].mean()
        profit = (bucket_data["win"] * (bucket_data["entry_odds"] - 1)) - (
            1 - bucket_data["win"]
        )  # noqa: E501
        roi = profit.sum() / bucket_data["stake"].sum() * 100
        avg_stake = bucket_data["stake"].mean()

        bucket_results.append(
            {
                "bucket": bucket,
                "wr": wr,
                "roi": roi,
                "count": len(bucket_data),
                "avg_stake": avg_stake,
            }
        )

        print(
            f"{
                bucket:<12} {
                wr:<12.2%} {
                roi:<12.2f}% {
                    len(bucket_data):<12} {
                        avg_stake:<12.2f}"
        )

    # Check monotonicity
    wr_values = [x["wr"] for x in bucket_results]
    is_monotonic = all(wr_values[i] <= wr_values[i + 1] for i in range(len(wr_values) - 1))

    # Count reversals
    reversals = sum(1 for i in range(len(wr_values) - 1) if wr_values[i] > wr_values[i + 1])

    print(f"\nMonotonic: {is_monotonic}")
    print(f"Number of Reversals: {reversals}")

    # Estimate impact of fixing monotonicity
    # Compare high-edge ROI to what it should be
    high_edge_buckets = [x for x in bucket_results if x["bucket"] in ["6 - 8", "8+"]]
    low_edge_buckets = [x for x in bucket_results if x["bucket"] in ["0 - 2", "2 - 4", "4 - 6"]]

    if high_edge_buckets and low_edge_buckets:
        high_edge_roi = np.mean([x["roi"] for x in high_edge_buckets])
        low_edge_roi = np.mean([x["roi"] for x in low_edge_buckets])

        # High edge should be at least 5% better than low edge
        expected_high_edge_roi = low_edge_roi + 5.0
        monotonicity_loss = expected_high_edge_roi - high_edge_roi

        # Weight by proportion of bets
        high_edge_count = sum([x["count"] for x in high_edge_buckets])
        high_edge_weight = high_edge_count / len(df)

        weighted_impact = monotonicity_loss * high_edge_weight

        print("\n--- ESTIMATED MONOTONICITY IMPACT ---")
        print(f"High Edge ROI (Actual):   {high_edge_roi:+.2f}%")
        print(f"High Edge ROI (Expected): {expected_high_edge_roi:+.2f}%")
        print(f"ROI Loss per High Edge:   {monotonicity_loss:+.2f}%")
        print(f"Weighted Portfolio Loss:  {weighted_impact:+.2f}%")
    else:
        weighted_impact = 0

    return {
        "is_monotonic": is_monotonic,
        "reversals": reversals,
        "estimated_impact": weighted_impact,
    }


def analyze_market_efficiency(df):
    """Test for statistical edge"""
    print("\n" + "=" * 80)
    print("MARKET EFFICIENCY ANALYSIS")
    print("=" * 80)

    df["model_edge"] = np.abs(df["predicted_pra"] - df["line"])

    # Edge correlation with outcomes
    corr, pval = spearmanr(df["model_edge"], df["win"])

    print(f"\nModel Edge vs Win Correlation: {corr:.4f} (p={pval:.4f})")
    print(f"Statistical Significance: {'YES' if pval < 0.05 else 'NO'}")

    if corr > 0.15 and pval < 0.05:
        assessment = "GOOD: Model has true edge"
    elif corr > 0.05:
        assessment = "WEAK: Marginal edge present"
    else:
        assessment = "BAD: No statistical edge detected"

    print(f"Assessment: {assessment}")

    # Estimate impact if we had strong edge
    # Benchmark: edge correlation of 0.15 should add ~3% ROI
    current_corr = max(corr, 0)
    target_corr = 0.15
    corr_gap = target_corr - current_corr

    # Rough estimate: 0.01 correlation = 0.2% ROI
    edge_impact = corr_gap * 20

    print("\n--- ESTIMATED EDGE IMPROVEMENT POTENTIAL ---")
    print(f"Current Correlation:  {current_corr:.4f}")
    print(f"Target Correlation:   {target_corr:.4f}")
    print(f"Correlation Gap:      {corr_gap:.4f}")
    print(f"Potential ROI Gain:   {edge_impact:+.2f}%")

    return {
        "edge_correlation": corr,
        "edge_pval": pval,
        "has_edge": (corr > 0.05 and pval < 0.05),
        "estimated_impact": edge_impact,
    }


def simulate_kelly_reduction(df):
    """Estimate impact of reducing Kelly fraction"""
    print("\n" + "=" * 80)
    print("KELLY FRACTION ANALYSIS")
    print("=" * 80)

    # Calculate what would happen with 1 / 4 Kelly
    df_quarter_kelly = df.copy()
    df_quarter_kelly["stake"] = df["stake"] * 0.25

    # Current ROI
    current_profit = (df["win"] * (df["entry_odds"] - 1)) - (1 - df["win"])
    current_roi = current_profit.sum() / df["stake"].sum() * 100

    # Quarter Kelly ROI
    quarter_profit = (df_quarter_kelly["win"] * (df_quarter_kelly["entry_odds"] - 1)) - (
        1 - df_quarter_kelly["win"]
    )
    quarter_roi = quarter_profit.sum() / df_quarter_kelly["stake"].sum() * 100

    # OWWR analysis
    current_owwr = (df["win"] * df["stake"]).sum() / df["stake"].sum()
    quarter_owwr = (df_quarter_kelly["win"] * df_quarter_kelly["stake"]).sum() / df_quarter_kelly[
        "stake"
    ].sum()

    print("\nCurrent Kelly (Full):")
    print(f"  OWWR:     {current_owwr:.2%}")
    print(f"  ROI:      {current_roi:+.2f}%")
    print(f"  Variance: {df['stake'].std():.3f}")

    print("\nQuarter Kelly (0.25x):")
    print(f"  OWWR:     {quarter_owwr:.2%}")
    print(f"  ROI:      {quarter_roi:+.2f}%")
    print(f"  Variance: {df_quarter_kelly['stake'].std():.3f}")

    kelly_impact = quarter_roi - current_roi

    print("\n--- KELLY REDUCTION IMPACT ---")
    print(f"ROI Change: {kelly_impact:+.2f}%")
    print(
        f"Variance Reduction: {(1 - df_quarter_kelly['stake'].std() / df['stake'].std()) * 100:.1f}%"  # noqa: E501
    )

    return {
        "current_roi": current_roi,
        "quarter_kelly_roi": quarter_roi,
        "estimated_impact": kelly_impact,
    }


def main():
    """Main execution"""
    print("=" * 80)
    print("DIAGNOSTIC IMPACT QUANTIFICATION")
    print("=" * 80)

    # Load data
    df = load_backtest_data()

    # Basic metrics
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    metrics = calculate_basic_metrics(df)
    for key, value in metrics.items():
        if isinstance(value, float):
            if "rate" in key or "gap" in key:
                print(f"{key:20s}: {value:.2%}")
            else:
                print(f"{key:20s}: {value:+.2f}")
        else:
            print(f"{key:20s}: {value}")

    # Run analyses
    results = {}
    results["side_bias"] = analyze_side_bias(df)
    results["calibration"] = analyze_calibration(df)
    results["edge_monotonicity"] = analyze_edge_monotonicity(df)
    results["market_efficiency"] = analyze_market_efficiency(df)
    results["kelly_reduction"] = simulate_kelly_reduction(df)

    # Summary
    print("\n" + "=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)
    print("\nEstimated ROI Impact by Root Cause:")
    print("-" * 80)

    total_impact = 0

    print(f"{'Root Cause':<30s} {'Estimated Impact':<20s}")
    print("-" * 80)

    for key, data in results.items():
        if "estimated_impact" in data:
            impact = data["estimated_impact"]
            total_impact += impact
            print(f"{key.replace('_', ' ').title():<30s} {impact:+.2f}% ROI")

    print("-" * 80)
    print(f"{'TOTAL POTENTIAL IMPROVEMENT':<30s} {total_impact:+.2f}% ROI")
    print(f"{'Current ROI':<30s} {metrics['roi']:+.2f}%")
    print(f"{'Projected ROI (Fixed)':<30s} {metrics['roi'] + total_impact:+.2f}%")  # noqa: E501
    print("=" * 80)

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "validation_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    summary_path = output_dir / "diagnostic_impact_summary.txt"

    with open(summary_path, "w") as f:
        f.write("DIAGNOSTIC IMPACT QUANTIFICATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Overall Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n\nEstimated Impacts:\n")
        for key, data in results.items():
            if "estimated_impact" in data:
                f.write(f"  {key}: {data['estimated_impact']:+.2f}% ROI\n")

        f.write(f"\n\nTotal Potential Improvement: {total_impact:+.2f}% ROI\n")
        f.write(f"Projected ROI: {metrics['roi'] + total_impact:+.2f}%\n")

    print(f"\n✓ Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
