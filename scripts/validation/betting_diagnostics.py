"""
Comprehensive Betting Model Diagnostics
Analyzes betting performance without closing lines using proxy CLV metrics

Author: Elite Data Science Team
Date: 2025 - 10 - 25
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, spearmanr
from sklearn.metrics import brier_score_loss, log_loss

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"


class BettingDiagnostics:
    """
    Comprehensive diagnostic framework for betting model evaluation

    Provides 7 diagnostic dimensions:
    1. Proxy CLV Analysis (no closing lines needed)
    2. Stake Diagnostics (Kelly overconfidence test)
    3. Side Bias Analysis (OVER vs UNDER)
    4. Edge Bucket Monotonicity
    5. Probability Calibration
    6. Root Cause Analysis (WR > BE but ROI < 0)
    7. Advanced Diagnostics (sigma calibration, market efficiency)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize diagnostics

        Args:
            df: DataFrame with columns:
                - entry_odds (decimal)
                - predicted_pra
                - actual_pra
                - line
                - side (OVER/UNDER)
                - stake
                - win (1 / 0)
                - player_sigma
                - entry_ev
                - date (optional)
        """
        self.df = df.copy()
        self._prepare_data()
        self.results = {}

    def _prepare_data(self):
        """Calculate derived metrics"""
        df = self.df

        # Model edge (proxy for CLV)
        df["model_edge"] = np.abs(df["predicted_pra"] - df["line"])

        # Prediction error
        df["prediction_error"] = np.abs(df["actual_pra"] - df["predicted_pra"])

        # Implied probability from odds
        df["implied_prob"] = 1 / df["entry_odds"]

        # Implied line movement (proxy for sharp money)
        df["ilm"] = np.where(
            df["side"] == "OVER", df["predicted_pra"] - df["line"], df["line"] - df["predicted_pra"]
        )

        # Predicted win probability (using normal distribution)
        df["pred_win_prob"] = df.apply(
            lambda row: (
                norm.cdf((row["predicted_pra"] - row["line"]) / row["player_sigma"])
                if row["side"] == "OVER"
                else norm.cdf(  # noqa: E501
                    (row["line"] - row["predicted_pra"]) / row["player_sigma"]
                )
            ),
            axis=1,
        )

        # Z-scores for sigma calibration
        df["z_score"] = (df["actual_pra"] - df["predicted_pra"]) / df["player_sigma"]

        # Market edge
        df["market_edge"] = df["pred_win_prob"] - df["implied_prob"]

        # Stake quartiles
        df["stake_quartile"] = pd.qcut(df["stake"], q=4, labels=[1, 2, 3, 4], duplicates="drop")

        # Edge buckets
        bins = [0, 2, 4, 6, 8, np.inf]
        labels = ["0 - 2", "2 - 4", "4 - 6", "6 - 8", "8+"]
        df["edge_bucket"] = pd.cut(df["model_edge"], bins=bins, labels=labels)

        # Probability bins
        prob_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 1.0]
        prob_labels = ["50 - 55", "55 - 60", "60 - 65", "65 - 70", "70+"]
        df["prob_bin"] = pd.cut(df["pred_win_prob"], bins=prob_bins, labels=prob_labels)

        self.df = df

    def calculate_proxy_clv(self) -> dict:
        """
        Diagnostic 1: Proxy CLV metrics without closing lines

        Uses model edge and implied line movement as proxies for sharp money
        """
        df = self.df

        # Model edge correlation with outcomes
        edge_corr, edge_pval = spearmanr(df["model_edge"], df["win"])

        # Sharp ratio (bets where ILM > 0)
        sharp_moves = df[df["ilm"] > 0]
        sharp_ratio = sharp_moves["win"].mean() if len(sharp_moves) > 0 else 0

        # Edge realization rate
        edge_real = (df["win"] * df["model_edge"]).sum() / df["model_edge"].sum()

        # ILM analysis
        ilm_pos = df[df["ilm"] > 0]
        ilm_neg = df[df["ilm"] < 0]
        ilm_pos_wr = ilm_pos["win"].mean() if len(ilm_pos) > 0 else 0
        ilm_neg_wr = ilm_neg["win"].mean() if len(ilm_neg) > 0 else 0

        results = {
            "edge_correlation": edge_corr,
            "edge_corr_pval": edge_pval,
            "sharp_ratio": sharp_ratio,
            "edge_realization": edge_real,
            "ilm_positive_wr": ilm_pos_wr,
            "ilm_negative_wr": ilm_neg_wr,
            "ilm_count_pos": len(ilm_pos),
            "ilm_count_neg": len(ilm_neg),
            "assessment": self._assess_clv(edge_corr, sharp_ratio, edge_real),
        }

        self.results["proxy_clv"] = results
        return results

    def _assess_clv(self, edge_corr, sharp_ratio, edge_real):
        """Assess CLV quality"""
        if edge_corr > 0.15 and sharp_ratio > 0.60:
            return "GOOD: Model has true edge vs market"
        elif edge_corr > 0.05 and sharp_ratio > 0.55:
            return "ACCEPTABLE: Weak edge present"
        else:
            return "BAD: No statistical edge detected"

    def calculate_stake_diagnostics(self) -> dict:
        """
        Diagnostic 2: Kelly overconfidence test

        Tests if heavier stakes correlate with worse outcomes
        """
        df = self.df

        # Stake-win correlation
        stake_corr, stake_pval = spearmanr(df["stake"], df["win"])

        # Win rate by quartile
        wr_by_quartile = df.groupby("stake_quartile")["win"].mean().to_dict()

        # ROI by quartile
        def calc_roi(x):
            profit = (x["win"] * (x["entry_odds"] - 1)) - (1 - x["win"])
            return (profit.sum() / x["stake"].sum() * 100) if x["stake"].sum() > 0 else 0

        roi_by_quartile = df.groupby("stake_quartile").apply(calc_roi).to_dict()

        # Weighted analysis
        total_staked_wins = df[df["win"] == 1]["stake"].sum()
        total_staked_losses = df[df["win"] == 0]["stake"].sum()

        heavy_loss_ratio = total_staked_losses / df["stake"].sum()
        light_win_ratio = total_staked_wins / df["stake"].sum()
        imbalance = heavy_loss_ratio - light_win_ratio

        # Monotonicity check
        wr_values = [wr_by_quartile.get(i, 0) for i in [1, 2, 3, 4]]
        is_monotonic = all(wr_values[i] <= wr_values[i + 1] for i in range(len(wr_values) - 1))

        results = {
            "stake_win_corr": stake_corr,
            "stake_corr_pval": stake_pval,
            "wr_by_quartile": wr_by_quartile,
            "roi_by_quartile": roi_by_quartile,
            "loss_win_imbalance": imbalance,
            "is_monotonic": is_monotonic,
            "assessment": self._assess_kelly(stake_corr, is_monotonic, imbalance),
        }

        self.results["stake_diagnostics"] = results
        return results

    def _assess_kelly(self, stake_corr, is_monotonic, imbalance):
        """Assess Kelly calibration"""
        if stake_corr < -0.10 or not is_monotonic:
            return "BAD: Kelly overconfident - reduce to 1 / 4 or 1 / 2 Kelly"
        elif abs(stake_corr) < 0.05 and is_monotonic:
            return "GOOD: Kelly is well-calibrated"
        else:
            return "ACCEPTABLE: Minor Kelly adjustment may help"

    def calculate_side_bias(self) -> dict:
        """
        Diagnostic 3: OVER vs UNDER analysis

        Identifies if model has side-specific bias
        """
        df = self.df

        overs = df[df["side"] == "OVER"]
        unders = df[df["side"] == "UNDER"]

        # Win rates
        over_wr = overs["win"].mean() if len(overs) > 0 else 0
        under_wr = unders["win"].mean() if len(unders) > 0 else 0
        wr_gap = over_wr - under_wr

        # Edge distribution
        over_edge = overs["model_edge"].mean() if len(overs) > 0 else 0
        under_edge = unders["model_edge"].mean() if len(unders) > 0 else 0

        # Statistical test
        if len(overs) > 0 and len(unders) > 0:
            t_stat, edge_pval = stats.ttest_ind(overs["model_edge"], unders["model_edge"])
        else:
            edge_pval = 1.0

        # Odds analysis
        over_win_odds = (
            overs[overs["win"] == 1]["entry_odds"].mean()
            if len(overs[overs["win"] == 1]) > 0
            else 0
        )
        under_win_odds = (
            unders[unders["win"] == 1]["entry_odds"].mean()
            if len(unders[unders["win"] == 1]) > 0
            else 0
        )

        # ROI by side
        def calc_side_roi(side_df):
            if len(side_df) == 0:
                return 0
            profit = (side_df["win"] * (side_df["entry_odds"] - 1)) - (1 - side_df["win"])
            return (
                (profit.sum() / side_df["stake"].sum() * 100) if side_df["stake"].sum() > 0 else 0
            )

        over_roi = calc_side_roi(overs)
        under_roi = calc_side_roi(unders)

        results = {
            "over_wr": over_wr,
            "under_wr": under_wr,
            "wr_gap": wr_gap,
            "over_mean_edge": over_edge,
            "under_mean_edge": under_edge,
            "edge_diff_pval": edge_pval,
            "over_win_odds": over_win_odds,
            "under_win_odds": under_win_odds,
            "over_roi": over_roi,
            "under_roi": under_roi,
            "sample_ratio": len(overs) / len(unders) if len(unders) > 0 else 0,
            "assessment": self._assess_side_bias(wr_gap, len(overs), len(unders)),
        }

        self.results["side_bias"] = results
        return results

    def _assess_side_bias(self, wr_gap, n_overs, n_unders):
        """Assess side bias severity"""
        if abs(wr_gap) > 0.10:
            return "BAD: Large side bias - apply side-specific isotonic regression"  # noqa: E501
        elif abs(wr_gap) > 0.05 and min(n_overs, n_unders) > 30:
            return "WARNING: Moderate side bias - monitor and calibrate"
        else:
            return "GOOD: No significant side bias"

    def calculate_edge_monotonicity(self) -> dict:
        """
        Diagnostic 4: Edge bucket analysis

        Verifies that larger edges win more (calibration check)
        """
        df = self.df

        # Bucket statistics
        bucket_stats = (
            df.groupby("edge_bucket")
            .agg(
                {
                    "win": ["count", "mean", "sum"],
                    "stake": "mean",
                    "entry_odds": "mean",
                    "model_edge": "mean",
                }
            )
            .round(4)
        )

        # Monotonicity test
        bucket_means = df.groupby("edge_bucket")["win"].mean()
        bucket_nums = range(len(bucket_means))

        if len(bucket_means) > 1:
            mono_corr, mono_pval = spearmanr(bucket_nums, bucket_means)
        else:
            mono_corr, mono_pval = 0, 1.0

        # ROI by bucket
        def calc_bucket_roi(x):
            profit = (x["win"] * (x["entry_odds"] - 1)) - (1 - x["win"])
            return (profit.sum() / x["stake"].sum() * 100) if x["stake"].sum() > 0 else 0

        roi_by_bucket = df.groupby("edge_bucket").apply(calc_bucket_roi)

        # Check for reversals
        wr_list = bucket_means.tolist()
        has_reversal = any(wr_list[i] > wr_list[i + 1] for i in range(len(wr_list) - 1))

        results = {
            "bucket_stats": bucket_stats,
            "bucket_win_rates": bucket_means.to_dict(),
            "roi_by_bucket": roi_by_bucket.to_dict(),
            "monotonicity_corr": mono_corr,
            "monotonicity_pval": mono_pval,
            "has_reversal": has_reversal,
            "is_monotonic": mono_corr > 0.70 and mono_pval < 0.05,
            "assessment": self._assess_monotonicity(mono_corr, has_reversal),
        }

        self.results["edge_monotonicity"] = results
        return results

    def _assess_monotonicity(self, mono_corr, has_reversal):
        """Assess edge monotonicity"""
        if mono_corr > 0.90 and not has_reversal:
            return "GOOD: Strong monotonic relationship"
        elif mono_corr > 0.70:
            return "ACCEPTABLE: Weak monotonicity - consider isotonic regression"  # noqa: E501
        else:
            return "BAD: Non-monotonic - APPLY ISOTONIC REGRESSION"

    def calculate_calibration(self) -> dict:
        """
        Diagnostic 5: Probability calibration analysis

        Tests if predicted probabilities match observed frequencies
        """
        df = self.df

        # Calibration table
        calib_stats = (
            df.groupby("prob_bin")
            .agg(
                {
                    "pred_win_prob": "mean",
                    "win": ["mean", "count"],  # noqa: E501
                    "model_edge": "mean",
                }
            )
            .round(4)
        )

        # Brier score
        brier = brier_score_loss(df["win"], df["pred_win_prob"])

        # Log loss
        # Clip probabilities to avoid log(0)
        pred_probs_clipped = np.clip(df["pred_win_prob"], 1e-10, 1 - 1e-10)
        logloss = log_loss(df["win"], pred_probs_clipped)

        # Expected Calibration Error (ECE)
        ece = 0
        for bin_label in df["prob_bin"].dropna().unique():
            bin_data = df[df["prob_bin"] == bin_label]
            if len(bin_data) > 0:
                bin_weight = len(bin_data) / len(df)
                pred_mean = bin_data["pred_win_prob"].mean()
                obs_mean = bin_data["win"].mean()
                calib_error = abs(pred_mean - obs_mean)
                ece += bin_weight * calib_error

        # Maximum calibration error
        max_error = 0
        for bin_label in df["prob_bin"].dropna().unique():
            bin_data = df[df["prob_bin"] == bin_label]
            if len(bin_data) > 0:
                error = abs(bin_data["pred_win_prob"].mean() - bin_data["win"].mean())
                max_error = max(max_error, error)

        results = {
            "calibration_table": calib_stats,
            "brier_score": brier,
            "log_loss": logloss,
            "ece": ece,
            "max_calibration_error": max_error,
            "assessment": self._assess_calibration(brier, ece),
        }

        self.results["calibration"] = results
        return results

    def _assess_calibration(self, brier, ece):
        """Assess probability calibration"""
        if brier < 0.20 and ece < 0.05:
            return "GOOD: Well-calibrated probabilities"
        elif brier < 0.25 and ece < 0.10:
            return "ACCEPTABLE: Minor calibration issues"
        else:
            return "BAD: Poor calibration - recalibrate with isotonic regression"  # noqa: E501

    def diagnose_roi_mystery(self) -> dict:
        """
        Diagnostic 6: Root cause of WR > BE but ROI < 0

        Identifies why model wins more than breakeven but loses money
        """
        df = self.df

        # Basic metrics
        traditional_wr = df["win"].mean()
        breakeven_wr = df["implied_prob"].mean()

        # Odds-weighted win rate
        owwr = (df["win"] * df["stake"]).sum() / df["stake"].sum()
        owwr_gap = owwr - traditional_wr

        # Odds analysis
        mean_odds_wins = df[df["win"] == 1]["entry_odds"].mean()
        mean_odds_losses = df[df["win"] == 0]["entry_odds"].mean()
        odds_imbalance = mean_odds_wins - mean_odds_losses

        # ROI calculation
        profit = (df["win"] * (df["entry_odds"] - 1)) - (1 - df["win"])
        actual_roi = (profit.sum() / df["stake"].sum() * 100) if df["stake"].sum() > 0 else 0

        # Expected ROI from entry EV
        expected_roi = df["entry_ev"].mean() * 100
        calibration_gap = actual_roi - expected_roi

        # Variance drag (Kelly risk)
        kelly_fractions = df["market_edge"] / (df["entry_odds"] - 1)
        avg_kelly = kelly_fractions.mean()
        kelly_variance = kelly_fractions.var()

        # Heavy loss analysis
        losses = df[df["win"] == 0]
        heavy_losses = losses.nlargest(10, "stake")
        heavy_loss_impact = heavy_losses["stake"].sum() / df["stake"].sum()

        results = {
            "traditional_wr": traditional_wr,
            "breakeven_wr": breakeven_wr,
            "wr_vs_breakeven": traditional_wr - breakeven_wr,
            "owwr": owwr,
            "owwr_gap": owwr_gap,
            "mean_odds_wins": mean_odds_wins,
            "mean_odds_losses": mean_odds_losses,
            "odds_imbalance": odds_imbalance,
            "actual_roi": actual_roi,
            "expected_roi": expected_roi,
            "calibration_gap": calibration_gap,
            "avg_kelly_fraction": avg_kelly,
            "kelly_variance": kelly_variance,
            "heavy_loss_impact": heavy_loss_impact,
            "assessment": self._assess_roi_mystery(owwr_gap, odds_imbalance, calibration_gap),
        }

        self.results["roi_mystery"] = results
        return results

    def _assess_roi_mystery(self, owwr_gap, odds_imbalance, calib_gap):
        """Diagnose ROI mystery"""
        diagnoses = []

        if owwr_gap < -0.02:
            diagnoses.append("Heavy staking on losses (Kelly overconfidence)")
        if odds_imbalance < -0.05:
            diagnoses.append("Winning on shorter odds (ROI dilution)")
        if calib_gap < -5:
            diagnoses.append("Model probabilities overconfident")
        if not diagnoses:
            diagnoses.append("Small sample variance or bad luck")

        return " | ".join(diagnoses)

    def calculate_advanced_diagnostics(self) -> dict:
        """
        Diagnostic 7: Advanced statistical tests

        Sigma calibration, market efficiency, prediction errors
        """
        df = self.df

        # Prediction error analysis
        wins_error = df[df["win"] == 1]["prediction_error"].mean()
        losses_error = df[df["win"] == 0]["prediction_error"].mean()
        error_ratio = wins_error / losses_error if losses_error > 0 else np.nan

        # Sigma calibration
        z_mean = df["z_score"].mean()
        z_std = df["z_score"].std()
        z_median = df["z_score"].median()

        # Normality test
        if len(df) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(df["z_score"])
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan  # noqa: F841

        sigma_calibrated = 0.8 < z_std < 1.2

        # Market efficiency test
        edge_corr, edge_pval = spearmanr(df["market_edge"], df["win"])
        has_true_edge = edge_corr > 0.15 and edge_pval < 0.05

        # Sequence analysis (runs test for randomness)
        wins_sequence = df["win"].values
        if len(wins_sequence) > 1:
            runs = 1 + sum(
                wins_sequence[i] != wins_sequence[i - 1] for i in range(1, len(wins_sequence))
            )
            n_wins = wins_sequence.sum()
            n_losses = len(wins_sequence) - n_wins

            if n_wins > 0 and n_losses > 0:
                expected_runs = 1 + (2 * n_wins * n_losses) / (n_wins + n_losses)
                var_runs = (
                    2
                    * n_wins
                    * n_losses
                    * (2 * n_wins * n_losses - n_wins - n_losses)  # noqa: E501
                ) / (
                    (n_wins + n_losses) ** 2 * (n_wins + n_losses - 1)
                )  # noqa: E501
                z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
                runs_pval = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            else:
                runs_pval = 1.0
        else:
            runs_pval = 1.0

        results = {
            "wins_pred_error": wins_error,
            "losses_pred_error": losses_error,
            "error_ratio": error_ratio,
            "z_mean": z_mean,
            "z_std": z_std,
            "z_median": z_median,
            "z_shapiro_pval": shapiro_p,
            "sigma_calibrated": sigma_calibrated,
            "market_edge_corr": edge_corr,
            "market_edge_pval": edge_pval,
            "has_true_edge": has_true_edge,
            "runs_test_pval": runs_pval,
            "sequence_random": runs_pval > 0.05,
            "assessment": self._assess_advanced(sigma_calibrated, has_true_edge),
        }

        self.results["advanced"] = results
        return results

    def _assess_advanced(self, sigma_calibrated, has_true_edge):
        """Assess advanced diagnostics"""
        issues = []
        if not sigma_calibrated:
            issues.append("Sigma miscalibrated")
        if not has_true_edge:
            issues.append("No statistical edge vs market")

        if not issues:
            return "GOOD: Advanced diagnostics pass"
        else:
            return "WARNING: " + ", ".join(issues)

    def run_all_diagnostics(self) -> dict:
        """Run all diagnostic tests"""
        print("Running comprehensive betting diagnostics...")
        print("=" * 80)

        self.calculate_proxy_clv()
        self.calculate_stake_diagnostics()
        self.calculate_side_bias()
        self.calculate_edge_monotonicity()
        self.calculate_calibration()
        self.diagnose_roi_mystery()
        self.calculate_advanced_diagnostics()

        print("✓ All diagnostics complete")
        return self.results

    def generate_report(self, save_path: str = None):
        """Generate comprehensive text report"""
        df = self.df

        if not self.results:
            self.run_all_diagnostics()

        report = []
        report.append("=" * 80)
        report.append("NBA PROPS MODEL DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"\nSample Size: {len(df)} bets")
        if "date" in df.columns:
            report.append(
                f"Date Range: {
                    df['date'].min()} to {
                    df['date'].max()}"
            )

        # Overall performance
        overall_wr = df["win"].mean()
        profit = (df["win"] * (df["entry_odds"] - 1)) - (1 - df["win"])
        overall_roi = (profit.sum() / df["stake"].sum() * 100) if df["stake"].sum() > 0 else 0

        report.append(f"\nOverall Win Rate: {overall_wr:.2%}")
        report.append(f"Overall ROI: {overall_roi:.2f}%")
        report.append(f"Total Profit/Loss: {profit.sum():.2f} units")

        # Diagnostic 1: Proxy CLV
        clv = self.results["proxy_clv"]
        report.append("\n" + "=" * 80)
        report.append("1. PROXY CLV ANALYSIS (No Closing Lines)")
        report.append("=" * 80)
        report.append(
            f"Model Edge Correlation: {
                clv['edge_correlation']:.3f} (p={
                clv['edge_corr_pval']:.3f})"
        )
        report.append("  ✓ GOOD if > 0.15    ✗ BAD if < 0.05")
        report.append(f"\nSharp Ratio: {clv['sharp_ratio']:.2%}")
        report.append("  ✓ GOOD if > 60%     ✗ BAD if < 55%")
        report.append(f"\nEdge Realization: {clv['edge_realization']:.3f}")
        report.append("  ✓ GOOD if > 0.55    ✗ BAD if < 0.50")
        report.append("\nImplied Line Movement:")
        report.append(
            f"  Positive ILM: {
                clv['ilm_count_pos']} bets, WR = {
                clv['ilm_positive_wr']:.2%}"
        )
        report.append(
            f"  Negative ILM: {
                clv['ilm_count_neg']} bets, WR = {
                clv['ilm_negative_wr']:.2%}"
        )
        report.append(f"\n{clv['assessment']}")

        # Diagnostic 2: Stake
        stake = self.results["stake_diagnostics"]
        report.append("\n" + "=" * 80)
        report.append("2. STAKE DIAGNOSTICS (Kelly Overconfidence Test)")
        report.append("=" * 80)
        report.append(
            f"Stake-Win Correlation: {
                stake['stake_win_corr']:.3f} (p={
                stake['stake_corr_pval']:.3f})"
        )
        report.append("  ✓ GOOD if ≈ 0       ✗ BAD if < -0.10")
        report.append("\nWin Rate by Stake Quartile:")
        for q, wr in stake["wr_by_quartile"].items():
            report.append(f"  Q{q}: {wr:.2%}")
        report.append(
            f"Monotonic: {
                '✓ YES' if stake['is_monotonic'] else '✗ NO'}"
        )
        report.append("\nROI by Stake Quartile:")
        for q, roi_val in stake["roi_by_quartile"].items():
            report.append(f"  Q{q}: {roi_val:.2f}%")
        report.append(f"\nLoss/Win Imbalance: {stake['loss_win_imbalance']:.2%}")
        report.append("  ✓ GOOD if < 10%     ✗ BAD if > 20%")
        report.append(f"\n{stake['assessment']}")

        # Diagnostic 3: Side Bias
        side = self.results["side_bias"]
        report.append("\n" + "=" * 80)
        report.append("3. SIDE BIAS ANALYSIS (OVER vs UNDER)")
        report.append("=" * 80)
        report.append(
            f"OVER Win Rate: {side['over_wr']:.2%} ({int(df[df['side'] == 'OVER']['win'].sum())}/{len(df[df['side'] == 'OVER'])} bets)"  # noqa: E501
        )
        report.append(
            f"UNDER Win Rate: {side['under_wr']:.2%} ({int(df[df['side'] == 'UNDER']['win'].sum())}/{len(df[df['side'] == 'UNDER'])} bets)"  # noqa: E501
        )
        report.append(f"Win Rate Gap: {side['wr_gap']:.2%}")
        report.append("  ⚠ WARNING if gap > 5%")
        report.append("\nROI by Side:")
        report.append(f"  OVER: {side['over_roi']:.2f}%")
        report.append(f"  UNDER: {side['under_roi']:.2f}%")
        report.append(f"\nMean Edge - OVER: {side['over_mean_edge']:.2f} pts")
        report.append(f"Mean Edge - UNDER: {side['under_mean_edge']:.2f} pts")
        report.append(f"Edge Difference p-value: {side['edge_diff_pval']:.3f}")
        report.append(f"\nSample Ratio (OVER/UNDER): {side['sample_ratio']:.2f}")
        report.append(f"\n{side['assessment']}")

        # Diagnostic 4: Edge Monotonicity
        edge = self.results["edge_monotonicity"]
        report.append("\n" + "=" * 80)
        report.append("4. EDGE BUCKET MONOTONICITY")
        report.append("=" * 80)
        report.append("\nWin Rate by Edge Bucket:")
        for bucket, wr in edge["bucket_win_rates"].items():
            roi = edge["roi_by_bucket"].get(bucket, 0)
            report.append(f"  {bucket} pts: WR={wr:.2%}, ROI={roi:.2f}%")
        report.append(
            f"\nMonotonicity Correlation: {
                edge['monotonicity_corr']:.3f} (p={
                edge['monotonicity_pval']:.3f})"
        )
        report.append("  ✓ GOOD if > 0.90    ✗ BAD if < 0.70")
        report.append(
            f"Has Reversal: {
                '✗ YES' if edge['has_reversal'] else '✓ NO'}"
        )
        report.append(f"\n{edge['assessment']}")

        # Diagnostic 5: Calibration
        calib = self.results["calibration"]
        report.append("\n" + "=" * 80)
        report.append("5. PROBABILITY CALIBRATION")
        report.append("=" * 80)
        report.append(f"\nBrier Score: {calib['brier_score']:.4f}")
        report.append("  ✓ GOOD if < 0.20    ✗ BAD if > 0.25")
        report.append(f"\nLog Loss: {calib['log_loss']:.4f}")
        report.append("  ✓ GOOD if < 0.65    ✗ BAD if > 0.70")
        report.append(
            f"\nExpected Calibration Error (ECE): {
                calib['ece']:.4f}"
        )
        report.append("  ✓ GOOD if < 0.05    ✗ BAD if > 0.10")
        report.append(
            f"\nMax Calibration Error: {
                calib['max_calibration_error']:.4f}"
        )
        report.append(f"\n{calib['assessment']}")

        # Diagnostic 6: ROI Mystery
        roi = self.results["roi_mystery"]
        report.append("\n" + "=" * 80)
        report.append("6. ROOT CAUSE: WR > BE but ROI < 0")
        report.append("=" * 80)
        report.append(f"Traditional Win Rate: {roi['traditional_wr']:.2%}")
        report.append(f"Breakeven Win Rate: {roi['breakeven_wr']:.2%}")
        report.append(f"WR vs Breakeven: {roi['wr_vs_breakeven']:.2%}")
        report.append(f"\nOdds-Weighted Win Rate: {roi['owwr']:.2%}")
        report.append(f"OWWR Gap: {roi['owwr_gap']:.2%}")
        report.append("  ⚠ If negative: Heavy stakes on losses")
        report.append(f"\nMean Odds - Wins: {roi['mean_odds_wins']:.3f}")
        report.append(f"Mean Odds - Losses: {roi['mean_odds_losses']:.3f}")
        report.append(f"Odds Imbalance: {roi['odds_imbalance']:.3f}")
        report.append("  ⚠ If negative: Winning on shorter odds")
        report.append(f"\nExpected ROI: {roi['expected_roi']:.2f}%")
        report.append(f"Actual ROI: {roi['actual_roi']:.2f}%")
        report.append(f"Calibration Gap: {roi['calibration_gap']:.2f}%")
        report.append("  ⚠ If gap > 5%: Model overconfident")
        report.append(f"\nAvg Kelly Fraction: {roi['avg_kelly_fraction']:.4f}")
        report.append(
            f"Heavy Loss Impact: {
                roi['heavy_loss_impact']:.2%} of capital"
        )
        report.append(f"\n{roi['assessment']}")

        # Diagnostic 7: Advanced
        adv = self.results["advanced"]
        report.append("\n" + "=" * 80)
        report.append("7. ADVANCED DIAGNOSTICS")
        report.append("=" * 80)
        report.append(f"Prediction Error - Wins: {adv['wins_pred_error']:.2f} pts")
        report.append(f"Prediction Error - Losses: {adv['losses_pred_error']:.2f} pts")
        report.append(f"Error Ratio (Wins/Losses): {adv['error_ratio']:.3f}")
        report.append("  ✓ GOOD if < 1.0 (more accurate on wins)")
        report.append("\nZ-Score Statistics:")
        report.append(f"  Mean: {adv['z_mean']:.3f} (should be ≈ 0)")
        report.append(f"  Std: {adv['z_std']:.3f} (should be ≈ 1)")
        report.append(f"  Median: {adv['z_median']:.3f}")
        report.append(
            f"Sigma Calibrated: {
                '✓ YES' if adv['sigma_calibrated'] else '✗ NO'}"
        )
        report.append("  ✓ GOOD if std ∈ [0.8, 1.2]")
        report.append("\nMarket Efficiency:")
        report.append(
            f"  Edge Correlation: {
                adv['market_edge_corr']:.3f} (p={
                adv['market_edge_pval']:.3f})"
        )
        report.append(
            f"  Has True Edge: {
                '✓ YES' if adv['has_true_edge'] else '✗ NO'}"
        )
        report.append("\nSequence Randomness:")
        report.append(f"  Runs Test p-value: {adv['runs_test_pval']:.3f}")
        report.append(
            f"  Random: {
                '✓ YES' if adv['sequence_random'] else '✗ NO (possible clustering)'}"
        )  # noqa: E501
        report.append(f"\n{adv['assessment']}")

        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("ACTIONABLE RECOMMENDATIONS")
        report.append("=" * 80)

        recommendations = self._generate_recommendations()
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        else:
            report.append(
                "✓ Model is well-calibrated. Small sample variance likely cause."
            )  # noqa: E501

        report.append("\n" + "=" * 80)
        report.append("END REPORT")
        report.append("=" * 80)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {save_path}")

        return report_text

    def _generate_recommendations(self) -> list:
        """Generate prioritized recommendations"""
        recommendations = []

        # Critical issues
        if self.results["proxy_clv"]["edge_correlation"] < 0.05:
            recommendations.append(
                "❌ CRITICAL: No edge correlation - improve features or stop betting"  # noqa: E501
            )

        if self.results["stake_diagnostics"]["stake_win_corr"] < -0.10:
            recommendations.append(
                "❌ CRITICAL: Kelly overconfidence - reduce to 1 / 4 Kelly immediately"  # noqa: E501
            )

        # High priority
        if self.results["side_bias"]["wr_gap"] > 0.10:
            recommendations.append(
                "⚠ HIGH: Large side bias - apply side-specific isotonic regression"  # noqa: E501
            )

        if not self.results["edge_monotonicity"]["is_monotonic"]:
            recommendations.append("⚠ HIGH: Non-monotonic edges - apply isotonic regression")

        if self.results["calibration"]["ece"] > 0.10:
            recommendations.append(
                "⚠ HIGH: Poor calibration (ECE > 0.10) - recalibrate probabilities"  # noqa: E501
            )

        # Medium priority
        if self.results["roi_mystery"]["owwr_gap"] < -0.02:
            recommendations.append("⚠ MEDIUM: Heavy staking on losses - reduce Kelly fraction")

        if not self.results["advanced"]["sigma_calibrated"]:
            recommendations.append("⚠ MEDIUM: Player variance miscalibrated - adjust sigma model")

        if not self.results["advanced"]["has_true_edge"]:
            recommendations.append(
                "⚠ MEDIUM: No statistical edge vs market - improve model or pause"  # noqa: E501
            )

        if self.results["roi_mystery"]["calibration_gap"] < -5:
            recommendations.append("⚠ MEDIUM: Model overconfident - recalibrate probabilities")

        # Low priority
        if (
            self.results["side_bias"]["sample_ratio"] < 0.3
            or self.results["side_bias"]["sample_ratio"] > 3.0
        ):
            recommendations.append("ℹ LOW: Sample imbalance - collect more data on minority side")

        return recommendations

    def create_visualizations(self, save_dir: str = None):
        """Create diagnostic visualizations"""
        if not self.results:
            self.run_all_diagnostics()

        df = self.df

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle("NBA Props Model Diagnostics", fontsize=16, fontweight="bold")

        # 1. Edge vs Win Rate
        ax = axes[0, 0]
        edge_wr = df.groupby("edge_bucket")["win"].agg(["mean", "count"])
        edge_wr["mean"].plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
        ax.set_title("Win Rate by Model Edge", fontweight="bold")
        ax.set_xlabel("Edge Bucket (Points)")
        ax.set_ylabel("Win Rate")
        ax.axhline(0.533, color="r", linestyle="--", label="Breakeven (53.3%)", linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        for i, (idx, row) in enumerate(edge_wr.iterrows()):
            ax.text(i, row["mean"] + 0.02, f"n={int(row['count'])}", ha="center", fontsize=9)

        # 2. Stake vs Win Rate
        ax = axes[0, 1]
        if "stake_quartile" in df.columns and df["stake_quartile"].notna().any():
            stake_wr = df.groupby("stake_quartile")["win"].agg(["mean", "count"])
            stake_wr["mean"].plot(kind="bar", ax=ax, color="coral", alpha=0.7)
            ax.set_title("Win Rate by Stake Quartile", fontweight="bold")
            ax.set_xlabel("Stake Quartile (1=Low, 4=High)")
            ax.set_ylabel("Win Rate")
            ax.axhline(0.533, color="r", linestyle="--", linewidth=2)
            ax.grid(True, alpha=0.3)
            for i, (idx, row) in enumerate(stake_wr.iterrows()):
                ax.text(i, row["mean"] + 0.02, f"n={int(row['count'])}", ha="center", fontsize=9)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient stake variation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # 3. OVER vs UNDER
        ax = axes[0, 2]
        side_wr = df.groupby("side")["win"].agg(["mean", "count"])
        __bars = side_wr["mean"].plot(  # noqa: F841
            kind="bar", ax=ax, color=["green", "red"], alpha=0.7
        )
        ax.set_title("Win Rate by Side", fontweight="bold")
        ax.set_ylabel("Win Rate")
        ax.axhline(0.533, color="black", linestyle="--", linewidth=2)
        ax.grid(True, alpha=0.3)
        for i, (idx, row) in enumerate(side_wr.iterrows()):
            ax.text(i, row["mean"] + 0.02, f"n={int(row['count'])}", ha="center", fontsize=9)

        # 4. Calibration Curve
        ax = axes[1, 0]
        calib_data = df.groupby("prob_bin").agg({"pred_win_prob": "mean", "win": ["mean", "count"]})
        ax.scatter(
            calib_data[("pred_win_prob", "mean")],
            calib_data[("win", "mean")],
            s=calib_data[("win", "count")] * 10,
            alpha=0.6,
            color="blue",
        )
        ax.plot([0.5, 1.0], [0.5, 1.0], "r--", label="Perfect Calibration", linewidth=2)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Win Rate")
        ax.set_title("Reliability Diagram", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.0, 1.0)

        # 5. ROI by Edge Bucket
        ax = axes[1, 1]
        edge_roi = self.results["edge_monotonicity"]["roi_by_bucket"]
        edge_roi_series = pd.Series(edge_roi)
        edge_roi_series.plot(kind="bar", ax=ax, color="purple", alpha=0.7)
        ax.set_title("ROI by Edge Bucket", fontweight="bold")
        ax.set_xlabel("Edge Bucket (Points)")
        ax.set_ylabel("ROI (%)")
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.grid(True, alpha=0.3)

        # 6. Prediction Error Distribution
        ax = axes[1, 2]
        wins = df[df["win"] == 1]["prediction_error"]
        losses = df[df["win"] == 0]["prediction_error"]
        ax.hist(
            wins,
            bins=20,
            alpha=0.5,
            label=f"Wins (n={len(wins)})",
            color="green",
            edgecolor="black",
        )
        ax.hist(
            losses,
            bins=20,
            alpha=0.5,
            label=f"Losses (n={len(losses)})",
            color="red",
            edgecolor="black",
        )
        ax.set_xlabel("Prediction Error (Points)")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Error: Wins vs Losses", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 7. Z-Score Distribution
        ax = axes[2, 0]
        z_scores = df["z_score"].dropna()
        ax.hist(z_scores, bins=30, edgecolor="black", alpha=0.7, density=True, color="skyblue")

        # Overlay normal distribution
        x = np.linspace(z_scores.min(), z_scores.max(), 100)
        ax.plot(x, stats.norm.pdf(x, 0, 1), "r--", linewidth=2, label="N(0,1)")
        ax.axvline(0, color="black", linestyle="--", label="Mean", linewidth=2)
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("Density")
        ax.set_title("Z-Score Distribution (Sigma Calibration)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Odds Distribution: Wins vs Losses
        ax = axes[2, 1]
        wins_odds = df[df["win"] == 1]["entry_odds"]
        losses_odds = df[df["win"] == 0]["entry_odds"]
        ax.hist(
            wins_odds,
            bins=20,
            alpha=0.5,
            label=f"Wins (μ={wins_odds.mean():.2f})",
            color="green",
            edgecolor="black",
        )
        ax.hist(
            losses_odds,
            bins=20,
            alpha=0.5,
            label=f"Losses (μ={losses_odds.mean():.2f})",
            color="red",
            edgecolor="black",
        )
        ax.set_xlabel("Entry Odds (Decimal)")
        ax.set_ylabel("Frequency")
        ax.set_title("Odds Distribution: Wins vs Losses", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 9. Cumulative ROI
        ax = axes[2, 2]
        if "date" in df.columns:
            df_sorted = df.sort_values("date").reset_index(drop=True)
        else:
            df_sorted = df.reset_index(drop=True)

        cumulative_pl = (
            (df_sorted["win"] * (df_sorted["entry_odds"] - 1))
            - (1 - df_sorted["win"])  # noqa: E501
        ).cumsum()
        ax.plot(range(len(df_sorted)), cumulative_pl, linewidth=2, color="navy")
        ax.axhline(0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Bet Number")
        ax.set_ylabel("Cumulative P&L (Units)")
        ax.set_title("Cumulative P&L Over Time", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.fill_between(
            range(len(df_sorted)),
            cumulative_pl,
            0,
            where=(cumulative_pl >= 0),
            color="green",
            alpha=0.2,
        )
        ax.fill_between(
            range(len(df_sorted)),
            cumulative_pl,
            0,
            where=(cumulative_pl < 0),
            color="red",
            alpha=0.2,
        )

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / "diagnostic_plots.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Diagnostic plots saved to: {save_path}")

        return fig


def main():
    """Main execution function"""
    import sys

    # Load data
    data_path = Path(__file__).parent.parent.parent / "data" / "clv_ledger.csv"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure clv_ledger.csv exists in data/ directory")
        sys.exit(1)

    print(f"Loading betting data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} bets")
    print(f"Columns: {df.columns.tolist()}")

    # Validate required columns
    required_cols = [
        "entry_odds",
        "predicted_pra",
        "actual_pra",
        "line",
        "side",
        "stake",
        "win",
        "player_sigma",
        "entry_ev",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)

    # Initialize diagnostics
    diagnostics = BettingDiagnostics(df)

    # Run all diagnostics
    print("\nRunning comprehensive diagnostics...")
    results = diagnostics.run_all_diagnostics()

    # Generate report
    output_dir = Path(__file__).parent.parent.parent / "data" / "validation_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    report_path = output_dir / "betting_diagnostics_report.txt"
    report_text = diagnostics.generate_report(save_path=str(report_path))

    # Print to console
    print("\n" + report_text)

    # Create visualizations
    print("\nGenerating diagnostic visualizations...")
    diagnostics.create_visualizations(save_dir=str(output_dir))

    # Save results as JSON
    json_path = output_dir / "betting_diagnostics_results.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {
                k: (
                    float(v)
                    if isinstance(v, (np.integer, np.floating))
                    else (str(v) if isinstance(v, pd.DataFrame) else v)
                )
                for k, v in value.items()
            }
        else:
            json_results[key] = value

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {json_path}")
    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}")
    print("\nView results:")
    print(f"  - Text report: {report_path}")
    print(f"  - Visualizations: {output_dir / 'diagnostic_plots.png'}")
    print(f"  - JSON results: {json_path}")


if __name__ == "__main__":
    main()
