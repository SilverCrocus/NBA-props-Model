#!/usr/bin/env python3
"""
Advanced Betting Diagnostics (No Closing Lines Required)
========================================================

Comprehensive diagnostic framework for betting model validation without needing
closing lines. Uses proxy metrics and statistical tests to identify issues.

Usage:
    uv run python scripts/validation/advanced_diagnostics.py

Author: Claude Code
Date: 2025 - 10 - 24
"""

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


class AdvancedBettingDiagnostics:
    """
    Advanced diagnostics for betting model validation

    Diagnostic Dimensions:
    1. Model Edge Correlation - Does model have predictive power?
    2. Stake Overconfidence - Is Kelly too aggressive?
    3. Edge Bucket Monotonicity - Does higher edge → higher WR?
    4. Side Bias - OVER vs UNDER asymmetry
    5. Calibration (ECE) - Probability accuracy
    6. Price Distribution - Odds discipline
    7. Sharp Ratio Proxy - Model edge as CLV proxy
    """

    def __init__(self, ledger_path: str = "data/clv_ledger.csv"):
        """Initialize diagnostics"""
        self.ledger_path = Path(ledger_path)

        if not self.ledger_path.exists():
            raise FileNotFoundError(
                f"CLV ledger not found: {
                    self.ledger_path}"
            )

        # Load data
        self.ledger = pd.read_csv(self.ledger_path)

        # Filter to bets with results
        self.results_df = self.ledger[~self.ledger["won"].isna()].copy()

        if len(self.results_df) == 0:
            raise ValueError("No bet results available for diagnostics")

        # Calculate derived metrics
        self._calculate_derived_metrics()

        print(f"✅ Loaded {len(self.results_df)} bets with results")

    def _calculate_derived_metrics(self):
        """Calculate derived metrics for analysis"""
        df = self.results_df

        # Model edge (predicted - line) * direction
        df["model_edge"] = (df["predicted_pra"] - df["line"]) * df["direction"]

        # Breakeven rate
        df["breakeven_rate"] = 1 / df["entry_odds_dec"]

        # Edge bins (for monotonicity check)
        df["edge_bin"] = pd.cut(
            df["model_edge"].abs(),
            bins=[0, 2, 4, 6, 8, 100],
            labels=["0 - 2", "2 - 4", "4 - 6", "6 - 8", "8+"],
        )

        # Stake quartiles
        df["stake_quartile"] = pd.qcut(
            df["stake"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"], duplicates="drop"
        )

        # Calibrated probability (from model)
        # P(OVER) = 1 - CDF(line | μ=pred, σ=sigma)
        from scipy.stats import norm

        if "player_sigma" in df.columns and df["player_sigma"].notna().any():
            df["calibrated_prob"] = df.apply(
                lambda row: (
                    1 - norm.cdf(row["line"], loc=row["predicted_pra"], scale=row["player_sigma"])
                    if row["direction"] == 1
                    else norm.cdf(  # noqa: E501
                        row["line"], loc=row["predicted_pra"], scale=row["player_sigma"]
                    )
                ),
                axis=1,
            )
        else:
            # Fallback if sigma missing
            df["calibrated_prob"] = np.nan

        self.results_df = df

    def run_all_diagnostics(self) -> Dict:
        """Run all diagnostic tests and return results"""
        results = {
            "summary": self._get_summary_stats(),
            "edge_correlation": self._test_edge_correlation(),
            "stake_overconfidence": self._test_stake_overconfidence(),
            "edge_monotonicity": self._test_edge_monotonicity(),
            "side_bias": self._test_side_bias(),
            "calibration": self._test_calibration(),
            "price_distribution": self._analyze_price_distribution(),
            "sharp_ratio": self._calculate_sharp_ratio(),
            "health_score": None,  # Will calculate after all tests
        }

        # Calculate overall health score
        results["health_score"] = self._calculate_health_score(results)

        return results

    def _get_summary_stats(self) -> Dict:
        """Get basic summary statistics"""
        df = self.results_df

        total_bets = len(df)
        wins = int(df["won"].sum())
        losses = total_bets - wins
        win_rate = df["won"].mean()

        total_staked = df["stake"].sum()
        total_profit = df["profit"].sum()
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        avg_odds = df["entry_odds_dec"].mean()
        breakeven_rate = 1 / avg_odds

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "breakeven_rate": breakeven_rate,
            "edge_vs_breakeven": (win_rate - breakeven_rate) * 100,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": roi,
            "avg_odds": avg_odds,
        }

    def _test_edge_correlation(self) -> Dict:
        """
        Test 1: Model Edge Correlation

        Does larger predicted edge lead to higher win rate?
        This is the #1 test for model validity.

        Method: Spearman correlation between model_edge and win outcome

        Interpretation:
        - ρ > 0.15: Model has predictive power (GOOD)
        - ρ 0.05 - 0.15: Weak signal (MARGINAL)
        - ρ < 0.05: No edge, wins are luck (BAD)
        """
        df = self.results_df

        # Spearman correlation (model_edge vs won)
        corr, p_value = spearmanr(df["model_edge"], df["won"])

        # Win rate by edge quintile
        edge_quintiles = pd.qcut(
            df["model_edge"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
        )
        wr_by_quintile = df.groupby(edge_quintiles)["won"].agg(["mean", "count"])

        # Status
        if corr > 0.15:
            status = "PASS"
            message = "✅ Model has predictive power"
        elif corr > 0.05:
            status = "MARGINAL"
            message = "⚠️  Weak edge signal"
        else:
            status = "FAIL"
            message = "❌ No statistical edge"

        return {
            "correlation": corr,
            "p_value": p_value,
            "wr_by_quintile": wr_by_quintile.to_dict(),
            "status": status,
            "message": message,
        }

    def _test_stake_overconfidence(self) -> Dict:
        """
        Test 2: Stake Overconfidence

        Are we over-staking on low-confidence bets (Kelly overconfidence)?

        Method:
        - Spearman correlation between stake and win
        - Win rate by stake quartile
        - OWWR (Observed Win rate Weighted by Risk)

        Interpretation:
        - corr > 0: Higher stakes → higher WR (GOOD)
        - corr ~ 0: No relationship (NEUTRAL)
        - corr < 0: Higher stakes → lower WR (BAD - Kelly overconfidence)
        """
        df = self.results_df

        # Spearman correlation (stake vs won)
        corr, p_value = spearmanr(df["stake"], df["won"])

        # Win rate by stake quartile
        wr_by_quartile = df.groupby("stake_quartile")["won"].agg(["mean", "count"])

        # OWWR (Observed Win rate Weighted by Risk)
        # If OWWR < WR, we're overconfident (betting more on losers)
        weighted_wins = (df["stake"] * df["won"]).sum()
        total_stake = df["stake"].sum()
        owwr = weighted_wins / total_stake if total_stake > 0 else 0

        actual_wr = df["won"].mean()
        owwr_gap = (owwr - actual_wr) * 100

        # Status
        if owwr_gap > -2:
            status = "PASS"
            message = "✅ Stake sizing aligned with outcomes"
        elif owwr_gap > -5:
            status = "MARGINAL"
            message = "⚠️  Slight Kelly overconfidence"
        else:
            status = "FAIL"
            message = "❌ Severe Kelly overconfidence - reduce Kelly to 0.1×"

        return {
            "stake_win_correlation": corr,
            "p_value": p_value,
            "wr_by_quartile": wr_by_quartile.to_dict(),
            "owwr": owwr,
            "actual_wr": actual_wr,
            "owwr_gap": owwr_gap,
            "status": status,
            "message": message,
        }

    def _test_edge_monotonicity(self) -> Dict:
        """
        Test 3: Edge Bucket Monotonicity

        Does win rate increase monotonically with predicted edge?
        Non-monotonic pattern indicates calibration issues.

        Method: Check WR across edge bins (0 - 2, 2 - 4, 4 - 6, 6 - 8, 8+)

        Interpretation:
        - Monotonic increasing: Well-calibrated (GOOD)
        - Non-monotonic: Needs isotonic regression (BAD)
        """
        df = self.results_df

        # Win rate by edge bin
        wr_by_edge = df.groupby("edge_bin", observed=True)["won"].agg(["mean", "count"])

        # Check monotonicity
        wr_values = wr_by_edge["mean"].values
        is_monotonic = all(wr_values[i] <= wr_values[i + 1] for i in range(len(wr_values) - 1))

        # Calculate violations (how many times WR decreases)
        violations = sum(1 for i in range(len(wr_values) - 1) if wr_values[i] > wr_values[i + 1])

        # Status
        if is_monotonic:
            status = "PASS"
            message = "✅ Edge buckets are monotonic"
        elif violations <= 1:
            status = "MARGINAL"
            message = "⚠️  Minor calibration issues (1 violation)"
        else:
            status = "FAIL"
            message = f"❌ Non-monotonic ({violations} violations) - apply isotonic regression"  # noqa: E501

        return {
            "wr_by_edge_bin": wr_by_edge.to_dict(),
            "is_monotonic": is_monotonic,
            "violations": violations,
            "status": status,
            "message": message,
        }

    def _test_side_bias(self) -> Dict:
        """
        Test 4: Side Bias Analysis

        Is there asymmetry between OVER and UNDER performance?

        Method: Compare WR, ROI, and calibration for OVER vs UNDER

        Interpretation:
        - WR gap < 5%: No bias (GOOD)
        - WR gap 5 - 10%: Moderate bias (MARGINAL)
        - WR gap > 10%: Severe bias (BAD)
        """
        df = self.results_df

        # Split by side
        over_bets = df[df["side"] == "OVER"]
        under_bets = df[df["side"] == "UNDER"]

        # Calculate metrics for each side
        over_wr = over_bets["won"].mean() if len(over_bets) > 0 else 0
        under_wr = under_bets["won"].mean() if len(under_bets) > 0 else 0

        over_roi = (
            (over_bets["profit"].sum() / over_bets["stake"].sum() * 100)
            if len(over_bets) > 0
            else 0
        )
        under_roi = (
            (under_bets["profit"].sum() / under_bets["stake"].sum() * 100)
            if len(under_bets) > 0
            else 0
        )

        # WR gap
        wr_gap = abs(over_wr - under_wr) * 100

        # Status
        if wr_gap < 5:
            status = "PASS"
            message = "✅ No significant side bias"
        elif wr_gap < 10:
            status = "MARGINAL"
            message = f"⚠️  Moderate {wr_gap:.1f}% gap - monitor closely"
        else:
            status = "FAIL"
            message = f"❌ Severe {
                wr_gap:.1f}% gap - fix side-specific calibration"

        return {
            "over_count": len(over_bets),
            "under_count": len(under_bets),
            "over_wr": over_wr,
            "under_wr": under_wr,
            "over_roi": over_roi,
            "under_roi": under_roi,
            "wr_gap": wr_gap,
            "status": status,
            "message": message,
        }

    def _test_calibration(self) -> Dict:
        """
        Test 5: Calibration (Expected Calibration Error)

        Are predicted probabilities accurate?

        Method: ECE (Expected Calibration Error) - bins predicted probs vs actual frequencies  # noqa: E501

        Interpretation:
        - ECE < 0.05: Well-calibrated (GOOD)
        - ECE 0.05 - 0.10: Moderate miscalibration (MARGINAL)
        - ECE > 0.10: Severe miscalibration (BAD)
        """
        df = self.results_df

        if "calibrated_prob" not in df.columns or df["calibrated_prob"].isna().all():
            return {
                "ece": None,
                "status": "N/A",
                "message": "⚠️  No calibration data (player_sigma missing)",
            }

        # Remove NaN probabilities
        df_cal = df[df["calibrated_prob"].notna()].copy()

        if len(df_cal) == 0:
            return {"ece": None, "status": "N/A", "message": "⚠️  No valid calibration data"}

        # Bin predicted probabilities
        bins = [0, 0.4, 0.5, 0.6, 0.7, 1.0]
        df_cal["prob_bin"] = pd.cut(df_cal["calibrated_prob"], bins=bins)

        # Calculate ECE
        ece = 0
        calibration_by_bin = []

        for bin_label in df_cal["prob_bin"].cat.categories:
            bin_data = df_cal[df_cal["prob_bin"] == bin_label]

            if len(bin_data) == 0:
                continue

            pred_prob = bin_data["calibrated_prob"].mean()
            actual_freq = bin_data["won"].mean()
            bin_size = len(bin_data)

            # ECE contribution
            ece += (bin_size / len(df_cal)) * abs(pred_prob - actual_freq)

            calibration_by_bin.append(
                {
                    "bin": str(bin_label),
                    "count": bin_size,
                    "pred_prob": pred_prob,
                    "actual_freq": actual_freq,
                    "gap": pred_prob - actual_freq,
                }
            )

        # Status
        if ece < 0.05:
            status = "PASS"
            message = "✅ Model well-calibrated"
        elif ece < 0.10:
            status = "MARGINAL"
            message = f"⚠️  Moderate miscalibration (ECE={ece:.3f})"
        else:
            status = "FAIL"
            message = f"❌ Severe miscalibration (ECE={
                ece:.3f}) - apply isotonic regression"

        return {
            "ece": ece,
            "calibration_by_bin": calibration_by_bin,
            "status": status,
            "message": message,
        }

    def _analyze_price_distribution(self) -> Dict:
        """
        Test 6: Price Distribution Analysis

        Are we getting good prices? Avoiding short odds?

        Method: Analyze odds distribution and ROI by odds bucket

        Interpretation:
        - Avg odds > 1.90: Good price discipline (GOOD)
        - Avg odds 1.80 - 1.90: Acceptable (MARGINAL)
        - Avg odds < 1.80: Too many short odds (BAD)
        """
        df = self.results_df

        # Odds buckets
        df["odds_bucket"] = pd.cut(
            df["entry_odds_dec"],
            bins=[0, 1.70, 1.85, 2.00, 10.0],
            labels=["<1.70", "1.70 - 1.85", "1.85 - 2.00", "2.00+"],
        )

        # ROI by odds bucket
        roi_by_odds = (
            df.groupby("odds_bucket", observed=True)
            .apply(
                lambda x: {
                    "count": len(x),
                    "wr": x["won"].mean(),
                    "roi": (
                        (x["profit"].sum() / x["stake"].sum() * 100) if x["stake"].sum() > 0 else 0
                    ),
                }
            )
            .to_dict()
        )

        avg_odds = df["entry_odds_dec"].mean()

        # Status
        if avg_odds >= 1.90:
            status = "PASS"
            message = f"✅ Good price discipline (avg {avg_odds:.2f})"
        elif avg_odds >= 1.80:
            status = "MARGINAL"
            message = f"⚠️  Acceptable odds (avg {avg_odds:.2f})"
        else:
            status = "FAIL"
            message = f"❌ Too many short odds (avg {avg_odds:.2f})"

        return {
            "avg_odds": avg_odds,
            "roi_by_odds": roi_by_odds,
            "status": status,
            "message": message,
        }

    def _calculate_sharp_ratio(self) -> Dict:
        """
        Test 7: Sharp Ratio (CLV Proxy)

        Uses model_edge as a proxy for CLV.

        Sharp Ratio = % of bets where model_edge correctly predicts outcome direction  # noqa: E501

        Method:
        - "Sharp" bet: model_edge > 0 and won=1, OR model_edge < 0 and won=0
        - SR > 55%: Sharp (GOOD)
        - SR 50 - 55%: Marginal (NEUTRAL)
        - SR < 50%: Dull (BAD)
        """
        df = self.results_df

        # Define "sharp" bets
        # Correct: (edge > 0 and won) OR (edge < 0 and lost)
        df["is_sharp"] = ((df["model_edge"] > 0) & (df["won"] == 1)) | (
            (df["model_edge"] < 0) & (df["won"] == 0)
        )

        sharp_pct = df["is_sharp"].mean()

        # Status
        if sharp_pct > 0.55:
            status = "PASS"
            message = f"✅ Sharp ({
                sharp_pct:.1%}) - model edges align with outcomes"
        elif sharp_pct > 0.50:
            status = "MARGINAL"
            message = f"⚠️  Marginal ({sharp_pct:.1%})"
        else:
            status = "FAIL"
            message = f"❌ Dull ({sharp_pct:.1%}) - model edges misaligned"

        return {"sharp_ratio": sharp_pct, "status": status, "message": message}

    def _calculate_health_score(self, results: Dict) -> Dict:
        """
        Calculate overall model health score (0 - 7)

        1 point for each test that passes
        """
        score = 0
        max_score = 7

        tests = [
            "edge_correlation",
            "stake_overconfidence",
            "edge_monotonicity",
            "side_bias",
            "calibration",
            "price_distribution",
            "sharp_ratio",
        ]

        for test in tests:
            if test in results and results[test]["status"] == "PASS":
                score += 1

        # Overall status
        if score >= 6:
            status = "✅ EXCELLENT"
        elif score >= 4:
            status = "⚠️  GOOD"
        elif score >= 2:
            status = "⚠️  MARGINAL"
        else:
            status = "❌ POOR"

        return {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score * 100),
            "status": status,
        }

    def print_report(self):
        """Print comprehensive diagnostic report"""
        results = self.run_all_diagnostics()

        print("\n" + "=" * 80)
        print("ADVANCED BETTING DIAGNOSTICS REPORT")
        print("=" * 80)

        # Summary
        summary = results["summary"]
        print(f"\n📊 Summary ({summary['total_bets']} bets):")
        print(f"   Win Rate:          {summary['win_rate']:.1%}")
        print(f"   Breakeven Rate:    {summary['breakeven_rate']:.1%}")
        print(f"   Edge:              {summary['edge_vs_breakeven']:+.1f}%")
        print(f"   ROI:               {summary['roi']:+.2f}%")

        # Test Results
        print("\n🔬 Diagnostic Tests:")

        tests = [
            ("1. Model Edge Correlation", results["edge_correlation"]),
            ("2. Stake Overconfidence", results["stake_overconfidence"]),
            ("3. Edge Monotonicity", results["edge_monotonicity"]),
            ("4. Side Bias", results["side_bias"]),
            ("5. Calibration (ECE)", results["calibration"]),
            ("6. Price Distribution", results["price_distribution"]),
            ("7. Sharp Ratio", results["sharp_ratio"]),
        ]

        for test_name, test_result in tests:
            status_symbol = {"PASS": "✅", "MARGINAL": "⚠️ ", "FAIL": "❌", "N/A": "⊘ "}.get(
                test_result["status"], "?"
            )

            print(f"\n   {test_name}:")
            print(f"      Status: {status_symbol} {test_result['status']}")
            print(f"      {test_result['message']}")

        # Health Score
        health = results["health_score"]
        print("\n🎯 Model Health Score:")
        print(
            f"   Score:  {health['score']}/{health['max_score']} ({health['percentage']:.0f}%)"
        )  # noqa: E501
        print(f"   Status: {health['status']}")

        # Recommendations
        print("\n💡 Recommendations:")
        self._print_recommendations(results)

        print("\n" + "=" * 80)

    def _print_recommendations(self, results: Dict):
        """Print actionable recommendations based on diagnostic results"""
        recommendations = []

        # Edge correlation
        if results["edge_correlation"]["status"] != "PASS":
            recommendations.append(
                "❌ Model has no statistical edge - improve features (TS%, L3, opponent stats)"  # noqa: E501
            )

        # Stake overconfidence
        if results["stake_overconfidence"]["status"] == "FAIL":
            recommendations.append("❌ Reduce Kelly fraction to 0.1× (currently too aggressive)")

        # Edge monotonicity
        if results["edge_monotonicity"]["status"] != "PASS":
            recommendations.append("❌ Apply isotonic regression for calibration")

        # Side bias
        side_bias = results["side_bias"]
        if side_bias["status"] == "FAIL":
            better_side = (
                "OVER" if side_bias["over_wr"] > side_bias["under_wr"] else "UNDER"
            )  # noqa: E501
            recommendations.append(f"❌ Bet {better_side} only until side bias fixed")

        # Calibration
        if results["calibration"]["status"] == "FAIL":
            recommendations.append("❌ Severe miscalibration - retrain with isotonic regression")

        # Price distribution
        if results["price_distribution"]["status"] == "FAIL":
            recommendations.append("❌ Avoid short odds (<1.80) - raise minimum EV threshold")

        # Sharp ratio
        if results["sharp_ratio"]["status"] != "PASS":
            recommendations.append("❌ Model edges don't align with outcomes - review predictions")

        if not recommendations:
            print("   ✅ Model is performing well - continue current strategy")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")


def main():
    """Main execution"""
    print("=" * 80)
    print("ADVANCED BETTING DIAGNOSTICS")
    print("=" * 80)

    try:
        diagnostics = AdvancedBettingDiagnostics()
        diagnostics.print_report()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
