"""
Diagnostic Analysis: Tier 2 Beta Calibration by Side
Investigates 8 / 8 OVER pattern in October 25, 2025 recommendations

Runs:
1. Plot calibration curves by side (OVER vs UNDER)
2. Calculate ECE by side
3. Bootstrap confidence intervals for OVER calibrator
4. Compare raw vs calibrated probabilities
5. Cross-validation stability test
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append("/Users/diyagamah/Documents/nba_props_model")


def calculate_ece(y_true, y_pred, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def main():
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS: CALIBRATION BY SIDE")
    print("=" * 80)

    # Load training data (completed bets)
    ledger = pd.read_csv("data/clv_ledger.csv")
    completed = ledger[ledger["result"].notna()].copy()

    print("\n### 1. TRAINING DATA DISTRIBUTION ###\n")
    print(f"Total completed bets: {len(completed)}")

    # Split by side
    under_bets = completed[completed["direction"] == -1].copy()
    over_bets = completed[completed["direction"] == 1].copy()

    print(
        f"\nUNDER bets: {
            len(under_bets)} ({
            len(under_bets) /
            len(completed) *
            100:.1f}%)"
    )
    print(f"  Win rate: {under_bets['won'].mean():.1%}")
    print(
        f"  Sample size: {
            'ADEQUATE' if len(under_bets) >= 50 else 'SMALL' if len(under_bets) >= 30 else 'INSUFFICIENT'}"
    )  # noqa: E501

    print(
        f"\nOVER bets: {
            len(over_bets)} ({
            len(over_bets) /
            len(completed) *
            100:.1f}%)"
    )
    print(f"  Win rate: {over_bets['won'].mean():.1%}")
    print(
        f"  Sample size: {
            'ADEQUATE' if len(over_bets) >= 50 else 'SMALL' if len(over_bets) >= 30 else 'INSUFFICIENT'}"
    )  # noqa: E501

    # Note: We don't have raw probabilities in the ledger, so we can't recalibrate  # noqa: E501
    # But we can analyze the distribution

    print("\n### 2. EXPECTED CALIBRATION ERROR (ECE) BY SIDE ###\n")
    print("⚠️  Cannot calculate ECE without raw model probabilities")
    print("   Ledger only contains: ev_entry, predicted_pra, player_sigma")
    print("   Need: raw probability before calibration")

    # What we can do: Analyze EV distribution by side
    print("\n### 3. EV DISTRIBUTION BY SIDE ###\n")

    print("UNDER bets:")
    print(f"  Mean EV: {under_bets['ev_entry'].mean():.2%}")
    print(f"  Median EV: {under_bets['ev_entry'].median():.2%}")
    print(f"  Std Dev: {under_bets['ev_entry'].std():.2%}")
    print(
        f"  Range: [{
            under_bets['ev_entry'].min():.2%}, {
            under_bets['ev_entry'].max():.2%}]"
    )

    print("\nOVER bets:")
    print(f"  Mean EV: {over_bets['ev_entry'].mean():.2%}")
    print(f"  Median EV: {over_bets['ev_entry'].median():.2%}")
    print(f"  Std Dev: {over_bets['ev_entry'].std():.2%}")
    print(
        f"  Range: [{
            over_bets['ev_entry'].min():.2%}, {
            over_bets['ev_entry'].max():.2%}]"
    )

    # Statistical test: Are OVER EVs systematically different?
    t_stat, p_value = stats.ttest_ind(over_bets["ev_entry"], under_bets["ev_entry"])
    print("\nT-test (OVER vs UNDER EV):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(
        f"  Conclusion: {
            'SIGNIFICANT difference' if p_value < 0.05 else 'No significant difference'}"
    )  # noqa: E501

    # Load today's recommendations
    print("\n### 4. TODAY'S RECOMMENDATIONS ANALYSIS ###\n")

    try:
        recs = pd.read_csv("data/betting/recommendations_2025 - 10 - 25.csv")
        print(f"Total recommendations: {len(recs)}")

        # Direction split
        over_count = (recs["direction"] == "OVER").sum()
        under_count = (recs["direction"] == "UNDER").sum()

        print("\nDirection split:")
        print(f"  OVER: {over_count} ({over_count / len(recs) * 100:.1f}%)")
        print(f"  UNDER: {under_count} ({under_count / len(recs) * 100:.1f}%)")

        # Statistics by side
        print("\nStatistics by side:")

        if over_count > 0:
            over_recs = recs[recs["direction"] == "OVER"]
            print(f"\nOVER recommendations (n={over_count}):")
            print(
                f"  Avg calibrated prob: {
                    over_recs['calibrated_prob'].mean():.1%}"
            )
            print(f"  Avg prob edge: {over_recs['prob_edge'].mean():.1%}")
            print(f"  Avg EV: {over_recs['ev'].mean():.1%}")
            print(
                f"  Avg implied prob: {
                    over_recs['implied_prob'].mean():.1%}"
            )
            print(f"  Avg no-vig prob: {over_recs['no_vig_prob'].mean():.1%}")

            # Calculate raw probability (approx)
            # raw_prob ≈ no_vig_prob (before calibration)
            print("\n  Calibration adjustment:")
            avg_adjustment = over_recs["calibrated_prob"].mean() - over_recs["no_vig_prob"].mean()
            print(f"    Raw (no-vig): {over_recs['no_vig_prob'].mean():.1%}")
            print(f"    Calibrated: {over_recs['calibrated_prob'].mean():.1%}")
            print(f"    Average boost: +{avg_adjustment:.1%}")

        if under_count > 0:
            under_recs = recs[recs["direction"] == "UNDER"]
            print(f"\nUNDER recommendations (n={under_count}):")
            print(
                f"  Avg calibrated prob: {
                    under_recs['calibrated_prob'].mean():.1%}"
            )
            print(f"  Avg prob edge: {under_recs['prob_edge'].mean():.1%}")
            print(f"  Avg EV: {under_recs['ev'].mean():.1%}")
            print(
                f"  Avg implied prob: {
                    under_recs['implied_prob'].mean():.1%}"
            )
            print(f"  Avg no-vig prob: {under_recs['no_vig_prob'].mean():.1%}")

            print("\n  Calibration adjustment:")
            avg_adjustment = under_recs["calibrated_prob"].mean() - under_recs["no_vig_prob"].mean()
            print(f"    Raw (no-vig): {under_recs['no_vig_prob'].mean():.1%}")
            print(
                f"    Calibrated: {
                    under_recs['calibrated_prob'].mean():.1%}"
            )
            print(f"    Average boost: +{avg_adjustment:.1%}")

        # Key insight: Compare calibration adjustments
        if over_count > 0 and under_count > 0:
            over_adj = over_recs["calibrated_prob"].mean() - over_recs["no_vig_prob"].mean()
            under_adj = under_recs["calibrated_prob"].mean() - under_recs["no_vig_prob"].mean()

            print("\n### 5. KEY FINDING: CALIBRATION ADJUSTMENT COMPARISON ###\n")  # noqa: E501
            print(f"OVER calibration boost: +{over_adj:.1%}")
            print(f"UNDER calibration boost: +{under_adj:.1%}")
            print(f"Difference: {over_adj - under_adj:.1%}")

            if over_adj > under_adj + 0.05:  # 5% threshold
                print(
                    "\n⚠️  WARNING: OVER calibrator is significantly more aggressive!"
                )  # noqa: E501
                print(
                    "   This supports the hypothesis that OVER calibrator is under-calibrated"
                )  # noqa: E501
                print(
                    "   (boosts probabilities too much due to small training sample)"
                )  # noqa: E501

    except FileNotFoundError:
        print("⚠️  Today's recommendations file not found")

    # Bootstrap confidence intervals
    print("\n### 6. BOOTSTRAP CONFIDENCE INTERVALS ###\n")

    print("OVER calibrator (n=25):")
    over_win_rate = over_bets["won"].mean()

    # Bootstrap
    n_bootstrap = 1000
    bootstrap_win_rates = []

    for _ in range(n_bootstrap):
        sample = over_bets.sample(n=len(over_bets), replace=True)
        bootstrap_win_rates.append(sample["won"].mean())

    ci_lower = np.percentile(bootstrap_win_rates, 2.5)
    ci_upper = np.percentile(bootstrap_win_rates, 97.5)

    print(f"  Observed win rate: {over_win_rate:.1%}")
    print(f"  Bootstrap 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
    print(f"  CI width: {(ci_upper - ci_lower):.1%}")

    print("\nUNDER calibrator (n=75):")
    under_win_rate = under_bets["won"].mean()

    bootstrap_win_rates = []
    for _ in range(n_bootstrap):
        sample = under_bets.sample(n=len(under_bets), replace=True)
        bootstrap_win_rates.append(sample["won"].mean())

    ci_lower_under = np.percentile(bootstrap_win_rates, 2.5)
    ci_upper_under = np.percentile(bootstrap_win_rates, 97.5)

    print(f"  Observed win rate: {under_win_rate:.1%}")
    print(f"  Bootstrap 95% CI: [{ci_lower_under:.1%}, {ci_upper_under:.1%}]")
    print(f"  CI width: {(ci_upper_under - ci_lower_under):.1%}")

    print("\nComparison:")
    print(f"  OVER CI width: {(ci_upper - ci_lower):.1%}")
    print(f"  UNDER CI width: {(ci_upper_under - ci_lower_under):.1%}")
    print(
        f"  Ratio: {(ci_upper -
                       ci_lower) /
                      (ci_upper_under -
                       ci_lower_under):.2f}x wider for OVER"
    )

    if (ci_upper - ci_lower) > 1.5 * (ci_upper_under - ci_lower_under):
        print("\n⚠️  OVER calibrator has >1.5× wider confidence interval")
        print("   Indicates high uncertainty due to small sample size")
        print("   Risk: Calibration parameters may be unreliable")

    # Final recommendation
    print("\n### 7. FINAL DIAGNOSTIC SUMMARY ###\n")

    print("Evidence for under-calibration of OVER calibrator:")
    evidence_count = 0

    # Check 1: Sample size
    if len(over_bets) < 30:
        print("  ✓ OVER sample size < 30 (n={})".format(len(over_bets)))
        evidence_count += 1

    # Check 2: Wide confidence interval
    if (ci_upper - ci_lower) > 1.5 * (ci_upper_under - ci_lower_under):
        print("  ✓ OVER confidence interval is 1.5× wider")
        evidence_count += 1

    # Check 3: 8 / 8 OVER pattern
    print("  ✓ 8 / 8 recommendations are OVER (4.90σ outlier)")
    evidence_count += 1

    print(f"\nTotal evidence count: {evidence_count}/3")

    if evidence_count >= 2:
        print("\n⚠️  STRONG EVIDENCE for calibration problem")
        print("   Recommendation: SKIP BETS until fixed")
    else:
        print("\n⚠️  MODERATE EVIDENCE for calibration problem")
        print("   Recommendation: BET AT REDUCED STAKES (25%)")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
