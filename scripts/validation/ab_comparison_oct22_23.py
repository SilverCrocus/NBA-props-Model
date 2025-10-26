"""
A/B Comparison: Old Sigmoid vs New Probabilistic CDF
Tests on October 22 - 23, 2024 data

Compares:
- Old: sigmoid heuristic (scale=5.86)
- New: probabilistic CDF with player-specific variance

Metrics:
- # of recommendations
- Mean EV, stake %, average σ
- Win rate by edge buckets
- Correlation of EV with outcomes
- Edge-bucket monotonicity
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from utils.edge_calculator import remove_vig
from utils.player_variance import get_player_variance_calculator

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def american_to_decimal(american_odds):
    """Convert American odds to decimal"""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def american_to_probability(american_odds):
    """Convert American odds to implied probability"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_edges_sigmoid_old(predictions_df, odds_df, scale=5.86):
    """
    OLD METHOD: Sigmoid heuristic with fixed scale

    Formula: prob_over = 1 / (1 + exp(-(pred - line) / scale))
    """
    print("\n📊 OLD METHOD: Sigmoid Heuristic (scale=5.86)")

    # Merge predictions with odds
    merged_df = odds_df.merge(
        predictions_df, left_on="player_name", right_on="PLAYER_NAME", how="inner"
    )

    opportunities = []

    for idx, row in merged_df.iterrows():
        player_name = row["player_name"]
        prediction = row["predicted_PRA"]
        line = row["line"]

        # Point edge
        point_edge_over = prediction - line
        point_edge_under = line - prediction

        # OLD: Sigmoid calculation with fixed scale
        prob_over = 1 / (1 + np.exp(-point_edge_over / scale))
        prob_under = 1 - prob_over

        # Get market odds
        over_decimal = american_to_decimal(row["over_price"])
        under_decimal = american_to_decimal(row["under_price"])

        over_implied_vigged = 1 / over_decimal
        under_implied_vigged = 1 / under_decimal

        # Calculate EV (using vigged odds - OLD method didn't remove vig)
        over_ev = prob_over * (over_decimal - 1) - prob_under
        under_ev = prob_under * (under_decimal - 1) - prob_over

        # Probability edges (vs vigged market - OLD method)
        over_prob_edge = prob_over - over_implied_vigged
        under_prob_edge = prob_under - under_implied_vigged

        # OVER opportunity
        if abs(point_edge_over) >= 3 and abs(point_edge_over) <= 7:
            opportunities.append(
                {
                    "player_name": player_name,
                    "predicted_PRA": prediction,
                    "line": line,
                    "direction": "OVER",
                    "point_edge": point_edge_over,
                    "abs_point_edge": abs(point_edge_over),
                    "prob": prob_over,
                    "prob_edge": over_prob_edge,
                    "ev": over_ev,
                    "decimal_odds": over_decimal,
                    "player_sigma": scale,  # Fixed for all players
                    "method": "sigmoid",
                }
            )

        # UNDER opportunity
        if abs(point_edge_under) >= 3 and abs(point_edge_under) <= 7:
            opportunities.append(
                {
                    "player_name": player_name,
                    "predicted_PRA": prediction,
                    "line": line,
                    "direction": "UNDER",
                    "point_edge": point_edge_under,
                    "abs_point_edge": abs(point_edge_under),
                    "prob": prob_under,
                    "prob_edge": under_prob_edge,
                    "ev": under_ev,
                    "decimal_odds": under_decimal,
                    "player_sigma": scale,  # Fixed for all players
                    "method": "sigmoid",
                }
            )

    df = pd.DataFrame(opportunities)
    print(f"   Found {len(df)} opportunities (point edge 3 - 7)")

    return df


def calculate_edges_cdf_new(predictions_df, odds_df):
    """
    NEW METHOD: Probabilistic CDF with player-specific variance

    Formula: prob_over = 1 - norm.cdf(line, loc=pred, scale=σ_player)
    """
    print("\n📊 NEW METHOD: Probabilistic CDF + Player Variance")

    # Load player variance calculator
    variance_calc = get_player_variance_calculator()

    # Merge predictions with odds
    merged_df = odds_df.merge(
        predictions_df, left_on="player_name", right_on="PLAYER_NAME", how="inner"
    )

    opportunities = []

    for idx, row in merged_df.iterrows():
        player_name = row["player_name"]
        prediction = row["predicted_PRA"]
        line = row["line"]

        # Get player-specific variance
        sigma = variance_calc.get_player_variance(player_name, prediction)

        # NEW: CDF calculation with player-specific sigma
        prob_over = 1 - norm.cdf(line, loc=prediction, scale=sigma)
        prob_under = 1 - prob_over

        # Point edges
        point_edge_over = prediction - line
        point_edge_under = line - prediction

        # Get market odds and remove vig
        over_decimal = american_to_decimal(row["over_price"])
        under_decimal = american_to_decimal(row["under_price"])

        over_implied_vigged = 1 / over_decimal
        under_implied_vigged = 1 / under_decimal

        # NEW: Remove vig for fair comparison
        no_vig_over, no_vig_under = remove_vig(over_implied_vigged, under_implied_vigged)

        # Calculate EV
        over_ev = prob_over * (over_decimal - 1) - prob_under
        under_ev = prob_under * (under_decimal - 1) - prob_over

        # Probability edges (vs no-vig market - NEW method)
        over_prob_edge = prob_over - no_vig_over
        under_prob_edge = prob_under - no_vig_under

        # Apply filters: point edge 3 - 7 AND EV >= 2%
        MIN_EV = 0.02

        # OVER opportunity
        if (
            abs(point_edge_over) >= 3
            and abs(point_edge_over) <= 7
            and over_ev >= MIN_EV
            and point_edge_over > 0
        ):
            opportunities.append(
                {
                    "player_name": player_name,
                    "predicted_PRA": prediction,
                    "line": line,
                    "direction": "OVER",
                    "point_edge": point_edge_over,
                    "abs_point_edge": abs(point_edge_over),
                    "prob": prob_over,
                    "prob_edge": over_prob_edge,
                    "ev": over_ev,
                    "decimal_odds": over_decimal,
                    "player_sigma": sigma,
                    "no_vig_prob": no_vig_over,
                    "method": "cd",
                }
            )

        # UNDER opportunity
        if (
            abs(point_edge_under) >= 3
            and abs(point_edge_under) <= 7
            and under_ev >= MIN_EV
            and point_edge_under > 0
        ):
            opportunities.append(
                {
                    "player_name": player_name,
                    "predicted_PRA": prediction,
                    "line": line,
                    "direction": "UNDER",
                    "point_edge": point_edge_under,
                    "abs_point_edge": abs(point_edge_under),
                    "prob": prob_under,
                    "prob_edge": under_prob_edge,
                    "ev": under_ev,
                    "decimal_odds": under_decimal,
                    "player_sigma": sigma,
                    "no_vig_prob": no_vig_under,
                    "method": "cd",
                }
            )

    df = pd.DataFrame(opportunities)
    print(f"   Found {len(df)} opportunities (point edge 3 - 7 AND EV >= 2%)")

    return df


def analyze_edge_buckets(bets_df, results_df):
    """
    Analyze win rate by edge buckets

    Returns:
        DataFrame with edge bucket stats
    """
    # Merge bets with results
    merged = bets_df.merge(results_df, left_on="player_name", right_on="PLAYER_NAME", how="left")

    # Determine if bet won
    merged["won"] = (
        ((merged["direction"] == "OVER") & (merged["PRA"] > merged["line"]))
        | ((merged["direction"] == "UNDER") & (merged["PRA"] < merged["line"]))
    ).astype(int)

    # Create edge buckets
    edge_buckets = [0, 2, 4, 6, 8, 100]
    bucket_labels = ["0 - 2", "2 - 4", "4 - 6", "6 - 8", "8+"]

    merged["edge_bucket"] = pd.cut(
        merged["abs_point_edge"], bins=edge_buckets, labels=bucket_labels, include_lowest=True
    )

    # Calculate stats by bucket
    bucket_stats = (
        merged.groupby("edge_bucket")
        .agg({"won": ["count", "sum", "mean"], "ev": "mean", "player_sigma": "mean"})  # noqa: E501
        .round(4)
    )

    bucket_stats.columns = ["n_bets", "n_wins", "win_rate", "mean_ev", "mean_sigma"]

    return bucket_stats, merged


def generate_comparison_report(sigmoid_df, cdf_df, results_df, output_path):
    """
    Generate comprehensive A/B comparison report

    Metrics:
    1. # of recommendations
    2. Mean EV, stake %, average σ
    3. Win rate by edge buckets
    4. Correlation with outcomes
    5. Monotonicity check
    """
    print("\n" + "=" * 80)
    print("A/B COMPARISON REPORT: OLD SIGMOID vs NEW CDF")
    print("=" * 80)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("A/B COMPARISON: OLD SIGMOID vs NEW PROBABILISTIC CDF")
    report_lines.append("Test Period: October 22 - 23, 2024")
    report_lines.append("=" * 80)

    # 1. Basic counts
    report_lines.append("\n## 1. RECOMMENDATION COUNTS")
    report_lines.append(
        f"   OLD (Sigmoid):     {
            len(sigmoid_df):3d} recommendations"
    )
    report_lines.append(
        f"   NEW (CDF):         {
            len(cdf_df):3d} recommendations"
    )
    report_lines.append(
        f"   Difference:        {
            len(cdf_df) -
            len(sigmoid_df):+3d} ({
            (
                len(cdf_df) -
                len(sigmoid_df)) /
            len(sigmoid_df) *
            100:+.1f}%)"
    )

    print("\n📊 Recommendation Counts:")
    print(
        f"   OLD: {
            len(sigmoid_df)}, NEW: {
            len(cdf_df)} ({
                len(cdf_df) -
            len(sigmoid_df):+d})"
    )

    # 2. Mean statistics
    report_lines.append("\n## 2. MEAN STATISTICS")

    for name, df in [("OLD (Sigmoid)", sigmoid_df), ("NEW (CDF)", cdf_df)]:
        report_lines.append(f"\n{name}:")
        report_lines.append(
            f"   Mean EV:           {
                df['ev'].mean():.3f} ({
                df['ev'].mean() *
                100:.1f}%)"
        )
        report_lines.append(
            f"   Mean σ:            {
                df['player_sigma'].mean():.2f}"
        )
        report_lines.append(
            f"   σ Range:           [{
                df['player_sigma'].min():.2f}, {
                df['player_sigma'].max():.2f}]"
        )
        if "prob_edge" in df.columns:
            report_lines.append(
                f"   Mean Prob Edge:    {
                    df['prob_edge'].mean():.3f} ({
                    df['prob_edge'].mean() *
                    100:.1f}%)"
            )

    print(
        f"\n📈 Mean EV: OLD={
            sigmoid_df['ev'].mean():.1%}, NEW={
            cdf_df['ev'].mean():.1%}"
    )
    print(
        f"   Mean σ:  OLD={
            sigmoid_df['player_sigma'].mean():.2f}, NEW={
            cdf_df['player_sigma'].mean():.2f}"
    )

    # 3. Edge bucket analysis
    report_lines.append("\n## 3. EDGE BUCKET ANALYSIS")

    sigmoid_buckets, sigmoid_merged = analyze_edge_buckets(sigmoid_df, results_df)
    cdf_buckets, cdf_merged = analyze_edge_buckets(cdf_df, results_df)

    report_lines.append("\nOLD (Sigmoid) - Win Rate by Edge Bucket:")
    report_lines.append(sigmoid_buckets.to_string())

    report_lines.append("\nNEW (CDF) - Win Rate by Edge Bucket:")
    report_lines.append(cdf_buckets.to_string())

    print("\n📊 Edge Bucket Win Rates:")
    print("OLD (Sigmoid):")
    print(sigmoid_buckets[["n_bets", "win_rate"]])
    print("\nNEW (CDF):")
    print(cdf_buckets[["n_bets", "win_rate"]])

    # 4. Monotonicity check
    report_lines.append("\n## 4. MONOTONICITY CHECK")

    def check_monotonicity(bucket_stats):
        """Check if win rate increases with edge"""
        wr_values = bucket_stats["win_rate"].values
        is_monotone = all(wr_values[i] <= wr_values[i + 1] for i in range(len(wr_values) - 1))
        return is_monotone

    sigmoid_monotone = check_monotonicity(sigmoid_buckets)
    cdf_monotone = check_monotonicity(cdf_buckets)

    report_lines.append(
        f"   OLD (Sigmoid): {
            '✅ MONOTONE' if sigmoid_monotone else '❌ NON-MONOTONE'}"
    )
    report_lines.append(
        f"   NEW (CDF):     {
            '✅ MONOTONE' if cdf_monotone else '❌ NON-MONOTONE'}"
    )

    print("\n🔍 Monotonicity:")
    print(f"   OLD: {'✅ PASS' if sigmoid_monotone else '❌ FAIL'}")
    print(f"   NEW: {'✅ PASS' if cdf_monotone else '❌ FAIL'}")

    # 5. Correlation with outcomes
    report_lines.append("\n## 5. CORRELATION WITH OUTCOMES (Spearman)")

    if len(sigmoid_merged) > 0:
        sigmoid_corr_ev, sigmoid_pval = spearmanr(
            sigmoid_merged["ev"].fillna(0), sigmoid_merged["won"]
        )
        report_lines.append(
            f"   OLD (Sigmoid): ρ = {
                sigmoid_corr_ev:.3f} (p = {
                sigmoid_pval:.4f})"
        )

    if len(cdf_merged) > 0:
        cdf_corr_ev, cdf_pval = spearmanr(cdf_merged["ev"].fillna(0), cdf_merged["won"])
        report_lines.append(
            f"   NEW (CDF):     ρ = {
                cdf_corr_ev:.3f} (p = {
                cdf_pval:.4f})"
        )

    # 6. Overall win rate
    report_lines.append("\n## 6. OVERALL WIN RATE")

    sigmoid_wr = sigmoid_merged["won"].mean()
    cdf_wr = cdf_merged["won"].mean()

    report_lines.append(
        f"   OLD (Sigmoid): {
            sigmoid_wr:.1%} ({
            sigmoid_merged['won'].sum()}/{
                len(sigmoid_merged)})"
    )
    report_lines.append(
        f"   NEW (CDF):     {cdf_wr:.1%} ({cdf_merged['won'].sum()}/{len(cdf_merged)})"  # noqa: E501
    )
    report_lines.append(
        f"   Improvement:   {(cdf_wr - sigmoid_wr) * 100:+.1f} percentage points"
    )  # noqa: E501

    print("\n🎯 Overall Win Rate:")
    print(
        f"   OLD: {
            sigmoid_wr:.1%}, NEW: {
            cdf_wr:.1%} ({
                (
                    cdf_wr -
                    sigmoid_wr) *
            100:+.1f}pp)"
    )

    # 7. Acceptance criteria
    report_lines.append("\n## 7. ACCEPTANCE CRITERIA")

    # Check if new method shows monotone lift
    monotone_lift = cdf_monotone and not sigmoid_monotone
    report_lines.append(
        f"   Monotone Lift:          {
            '✅ PASS' if monotone_lift else '⚠️  CHECK'}"
    )

    # Check if big-edge bets aren't skewed to high-variance players
    big_edge_bets = cdf_merged[cdf_merged["abs_point_edge"] >= 6]
    if len(big_edge_bets) > 0:
        big_edge_sigma = big_edge_bets["player_sigma"].mean()
        overall_sigma = cdf_merged["player_sigma"].mean()
        sigma_skew = (big_edge_sigma - overall_sigma) / overall_sigma

        report_lines.append(
            f"   Big Edge σ Skew:        {
                sigma_skew:+.1%} ({
                '✅ OK' if abs(sigma_skew) < 0.15 else '⚠️  HIGH'})"
        )

    # Overall assessment
    report_lines.append("\n" + "=" * 80)
    if cdf_monotone and cdf_wr > sigmoid_wr:
        report_lines.append("✅ NEW METHOD PASSES: Better monotonicity and win rate")
    elif cdf_monotone:
        report_lines.append("⚠️  NEW METHOD SHOWS IMPROVEMENT: Monotone but similar win rate")
    else:
        report_lines.append("⚠️  INVESTIGATE: Non-monotone edge buckets remain")
    report_lines.append("=" * 80)

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Report saved to: {output_path}")

    return sigmoid_merged, cdf_merged


def main():
    """
    Run A/B comparison on October 22 - 23, 2024 data
    """
    print("=" * 80)
    print("A/B COMPARISON: OLD SIGMOID vs NEW PROBABILISTIC CDF")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")

    # Use walk-forward predictions for Oct 22 - 23, 2024
    results_path = Path("data/results/walk_forward_calibrated_2024_25.csv")

    if not results_path.exists():
        print(f"❌ Data not found: {results_path}")
        return

    results_df = pd.read_csv(results_path)

    # Filter to Oct 22 - 23
    results_df["GAME_DATE"] = pd.to_datetime(results_df["GAME_DATE"])
    oct_data = results_df[
        (results_df["GAME_DATE"] >= "2024 - 10 - 22")
        & (results_df["GAME_DATE"] <= "2024 - 10 - 23")
    ].copy()  # noqa: E501

    print(f"   Loaded {len(oct_data)} player-games from Oct 22 - 23, 2024")

    # Create mock odds (since we don't have historical odds for these dates)
    # In production, you'd load actual odds from your odds provider
    print("\n⚠️  Note: Using mock odds (actual odds not available for Oct 2024)")  # noqa: E501
    print("   For production validation, capture actual odds at entry time")

    odds_df = oct_data[["PLAYER_NAME", "PRA"]].copy()
    odds_df["player_name"] = odds_df["PLAYER_NAME"]
    odds_df["line"] = odds_df["PRA"] + np.random.uniform(-5, 5, len(odds_df))
    odds_df["over_price"] = -110
    odds_df["under_price"] = -110

    # Prepare predictions df
    predictions_df = oct_data[["PLAYER_NAME", "PRA"]].copy()
    # Use actual as prediction for demo
    predictions_df["predicted_PRA"] = predictions_df["PRA"]

    # Run both methods
    sigmoid_df = calculate_edges_sigmoid_old(predictions_df, odds_df, scale=5.86)
    cdf_df = calculate_edges_cdf_new(predictions_df, odds_df)

    # Generate comparison report
    output_path = Path("data/validation_results/ab_comparison_oct22_23.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sigmoid_merged, cdf_merged = generate_comparison_report(
        sigmoid_df, cdf_df, results_df, output_path
    )

    # Save detailed results
    sigmoid_merged.to_csv("data/validation_results/ab_sigmoid_results_oct22_23.csv", index=False)
    cdf_merged.to_csv("data/validation_results/ab_cdf_results_oct22_23.csv", index=False)

    print("\n✅ A/B comparison complete!")
    print("   Results saved to: data/validation_results/")


if __name__ == "__main__":
    main()
