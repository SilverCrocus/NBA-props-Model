#!/usr/bin/env python3
"""
Statistical Validation Module for Sports Betting Models
========================================================

Implements rigorous statistical validation methods:
1. Block Bootstrap Confidence Intervals (respects temporal correlation)
2. Isotonic Regression Calibration (fixes non-monotonic edge buckets)
3. Edge-bucket analysis and diagnostics

Author: Statistical Validation Framework
Date: October 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class BootstrapResult:
    """Results from block bootstrap analysis"""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    block_size: int
    bootstrap_samples: np.ndarray

    def __str__(self):
        return (
            f"{self.metric_name}: {self.point_estimate:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(SE: {self.std_error:.4f})"
        )


@dataclass
class CalibrationResult:
    """Results from isotonic regression calibration"""

    calibrator: IsotonicRegression
    calibrated_predictions: np.ndarray
    pre_calibration_mae: float
    post_calibration_mae: float
    pre_calibration_bins: pd.DataFrame
    post_calibration_bins: pd.DataFrame


class BlockBootstrap:
    """
    Block Bootstrap for Time Series Data

    Handles temporal correlation by resampling blocks of consecutive days.
    This preserves the correlation structure within game slates while allowing
    for statistical inference.

    Key Concepts:
    - Block Size: Choose based on correlation structure (typically 1 - 7 days)
    - Bootstrap Samples: 2,000 - 5,000 for stable CI estimates
    - Temporal Blocks: Maintain chronological order within blocks
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "GAME_DATE",
        block_size: int = 1,
        n_bootstrap: int = 5000,
        random_seed: int = 42,
    ):
        """
        Initialize block bootstrap validator

        Parameters:
        -----------
        df : pd.DataFrame
            Betting results with one row per bet
        date_col : str
            Column name for game dates
        block_size : int
            Number of days per block (1 = daily blocks, 7 = weekly)
        n_bootstrap : int
            Number of bootstrap iterations (recommend 5000)
        random_seed : int
            Random seed for reproducibility
        """
        self.df = df.copy()
        self.date_col = date_col
        self.block_size = block_size
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(random_seed)

        # Convert dates and sort
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Create blocks
        self.unique_dates = sorted(self.df[date_col].unique())
        self.blocks = self._create_blocks()

        print("Block Bootstrap Configuration:")
        print(f"  Total bets: {len(self.df):,}")
        print(f"  Unique dates: {len(self.unique_dates)}")
        print(f"  Block size: {block_size} days")
        print(f"  Number of blocks: {len(self.blocks)}")
        print(f"  Bootstrap iterations: {n_bootstrap:,}")

    def _create_blocks(self) -> List[pd.DataFrame]:
        """Create temporal blocks of data"""
        blocks = []

        for i in range(0, len(self.unique_dates), self.block_size):
            block_dates = self.unique_dates[i : i + self.block_size]
            block_df = self.df[self.df[self.date_col].isin(block_dates)]

            if len(block_df) > 0:
                blocks.append(block_df)

        return blocks

    def _resample_blocks(self) -> pd.DataFrame:
        """Resample blocks with replacement"""
        n_blocks = len(self.blocks)
        block_indices = self.rng.choice(n_blocks, size=n_blocks, replace=True)

        resampled_dfs = [self.blocks[idx] for idx in block_indices]
        return pd.concat(resampled_dfs, ignore_index=True)

    def bootstrap_metric(
        self, metric_func, metric_name: str, confidence_level: float = 0.95
    ) -> BootstrapResult:
        """
        Bootstrap a custom metric function

        Parameters:
        -----------
        metric_func : callable
            Function that takes DataFrame and returns scalar metric
            Example: lambda df: (df['win'].sum() / len(df))
        metric_name : str
            Name of the metric for reporting
        confidence_level : float
            Confidence level for CI (default 0.95 for 95% CI)

        Returns:
        --------
        BootstrapResult with point estimate and confidence interval
        """
        # Calculate point estimate on original data
        point_estimate = metric_func(self.df)

        # Bootstrap sampling
        bootstrap_samples = np.zeros(self.n_bootstrap)

        for i in range(self.n_bootstrap):
            resampled_df = self._resample_blocks()
            bootstrap_samples[i] = metric_func(resampled_df)

            if (i + 1) % 1000 == 0:
                print(f"  Bootstrap iteration {i + 1:,}/{self.n_bootstrap:,}")

        # Calculate confidence interval (percentile method)
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
        std_error = np.std(bootstrap_samples)

        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            n_bootstrap=self.n_bootstrap,
            block_size=self.block_size,
            bootstrap_samples=bootstrap_samples,
        )

    def validate_win_rate(
        self, win_col: str = "win", breakeven_wr: float = 0.5238, confidence_level: float = 0.95
    ) -> Tuple[BootstrapResult, bool]:  # noqa: E501
        """
        Validate win rate with block bootstrap

        Parameters:
        -----------
        win_col : str
            Column indicating win (1) or loss (0)
        breakeven_wr : float
            Breakeven win rate threshold (0.5238 for -110 odds)
        confidence_level : float
            Confidence level for CI

        Returns:
        --------
        (BootstrapResult, passes_test)
        """
        print(f"\n{'=' * 60}")
        print("WIN RATE VALIDATION (Block Bootstrap)")
        print(f"{'=' * 60}")

        def win_rate_metric(df):
            return df[win_col].mean()

        result = self.bootstrap_metric(win_rate_metric, "Win Rate", confidence_level)

        print("\nResults:")
        print(
            f"  Point Estimate: {
                result.point_estimate:.4f} ({
                result.point_estimate *
                100:.2f}%)"
        )
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  Std Error: {result.std_error:.4f}")
        print(
            f"  Breakeven WR: {
                breakeven_wr:.4f} ({
                breakeven_wr *
                100:.2f}%)"
        )

        passes = result.ci_lower > breakeven_wr

        if passes:
            print(
                f"\n✅ PASS: Lower bound ({
                    result.ci_lower:.4f}) > breakeven ({
                    breakeven_wr:.4f})"
            )
            print(
                f"   Model is statistically profitable at {
                    confidence_level *
                    100:.0f}% confidence"
            )
        else:
            print(
                f"\n❌ FAIL: Lower bound ({
                    result.ci_lower:.4f}) ≤ breakeven ({
                    breakeven_wr:.4f})"
            )
            print(
                f"   Cannot prove profitability at {
                    confidence_level *
                    100:.0f}% confidence"
            )

        return result, passes

    def validate_roi(
        self,
        profit_col: str = "profit",
        bet_size_col: str = "bet_size",
        confidence_level: float = 0.95,
    ) -> BootstrapResult:
        """
        Validate ROI with block bootstrap

        Parameters:
        -----------
        profit_col : str
            Column with profit/loss per bet
        bet_size_col : str
            Column with bet size (for ROI calculation)
        confidence_level : float
            Confidence level for CI

        Returns:
        --------
        BootstrapResult for ROI
        """
        print(f"\n{'=' * 60}")
        print("ROI VALIDATION (Block Bootstrap)")
        print(f"{'=' * 60}")

        def roi_metric(df):
            total_profit = df[profit_col].sum()
            total_risked = df[bet_size_col].sum()
            return total_profit / total_risked if total_risked > 0 else 0

        result = self.bootstrap_metric(roi_metric, "ROI", confidence_level)

        print("\nResults:")
        print(
            f"  Point Estimate: {
                result.point_estimate:.4f} ({
                result.point_estimate *
                100:.2f}%)"
        )
        print(f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"  Std Error: {result.std_error:.4f}")

        if result.ci_lower > 0:
            print("\n✅ Lower bound > 0: Statistically profitable")
        else:
            print("\n❌ Lower bound ≤ 0: Cannot prove profitability")

        return result

    def plot_bootstrap_distribution(self, result: BootstrapResult, save_path: Optional[str] = None):
        """Plot bootstrap distribution with confidence intervals"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(result.bootstrap_samples, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

        # Point estimate
        ax.axvline(
            result.point_estimate,
            color="darkblue",
            linestyle="-",
            linewidth=2,
            label=f"Point Estimate: {result.point_estimate:.4f}",
        )

        # Confidence interval
        ax.axvline(
            result.ci_lower,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
        )
        ax.axvline(result.ci_upper, color="red", linestyle="--", linewidth=2)

        # Fill CI region
        ax.axvspan(result.ci_lower, result.ci_upper, alpha=0.2, color="red")

        ax.set_xlabel(result.metric_name, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            f"Bootstrap Distribution: {result.metric_name}\n"
            f"({result.n_bootstrap:,} iterations, block size = {result.block_size})",  # noqa: E501
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved plot: {save_path}")

        return fig


class IsotonicCalibration:
    """
    Isotonic Regression for Probability Calibration

    Fixes non-monotonic edge buckets by learning monotonic mapping
    from predicted edges to empirical win rates.

    Key Concepts:
    - Monotonicity: Larger edges should have higher win rates
    - Out-of-sample: Fit on validation set, apply to test set
    - Diagnostic: Compare pre/post calibration bins
    """

    def __init__(self):
        """Initialize isotonic calibration"""
        self.calibrator = None
        self.is_fitted = False

    def fit(
        self, y_pred: np.ndarray, y_true: np.ndarray, verbose: bool = True
    ) -> "IsotonicCalibration":
        """
        Fit isotonic regression on validation data

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted PRA values
        y_true : np.ndarray
            Actual PRA values
        verbose : bool
            Print fitting summary

        Returns:
        --------
        self (fitted calibrator)
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("ISOTONIC REGRESSION CALIBRATION")
            print(f"{'=' * 60}")
            print(f"  Training samples: {len(y_pred):,}")

        # Fit isotonic regression
        # Map predicted PRA -> actual PRA (monotonic)
        self.calibrator = IsotonicRegression(
            y_min=y_true.min(), y_max=y_true.max(), increasing=True, out_of_bounds="clip"
        )

        self.calibrator.fit(y_pred, y_true)
        self.is_fitted = True

        # Evaluate on training data
        calibrated_pred = self.calibrator.predict(y_pred)

        mae_before = np.mean(np.abs(y_pred - y_true))
        mae_after = np.mean(np.abs(calibrated_pred - y_true))

        if verbose:
            print("\nCalibration Performance (Training Set):")
            print(f"  MAE before: {mae_before:.3f} pts")
            print(f"  MAE after: {mae_after:.3f} pts")
            print(
                f"  Improvement: {mae_before -
                                    mae_after:.3f} pts ({(1 -
                                                          mae_after /
                                                          mae_before) *
                                                         100:.1f}%)"
            )

        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self.calibrator.predict(y_pred)

    def analyze_edge_buckets(
        self,
        df: pd.DataFrame,
        pred_col: str = "predicted_PRA",
        actual_col: str = "PRA",
        line_col: str = "line",
        calibrated_col: Optional[str] = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Analyze win rate by edge buckets

        Parameters:
        -----------
        df : pd.DataFrame
            Betting data with predictions and actuals
        pred_col : str
            Column with predicted PRA
        actual_col : str
            Column with actual PRA
        line_col : str
            Column with betting line (if available, else None)
        calibrated_col : str
            Column with calibrated predictions (optional)
        n_bins : int
            Number of edge bins

        Returns:
        --------
        pd.DataFrame with edge bucket analysis
        """
        df_copy = df.copy()

        # Calculate edge
        if line_col and line_col in df_copy.columns:
            df_copy["edge"] = df_copy[pred_col] - df_copy[line_col]
        else:
            # If no line, use prediction deviation from mean
            df_copy["edge"] = df_copy[pred_col] - df_copy[pred_col].mean()

        # Calculate win
        df_copy["win"] = (
            (df_copy[actual_col] > df_copy[line_col]).astype(int)
            if line_col in df_copy.columns
            else None
        )

        # Create bins
        df_copy["edge_bin"] = pd.qcut(df_copy["edge"], q=n_bins, labels=False, duplicates="drop")

        # Aggregate by bin
        bucket_stats = df_copy.groupby("edge_bin").agg(
            {
                "edge": ["mean", "min", "max", "count"],
                pred_col: "mean",
                actual_col: "mean",
            }
        )

        # Flatten column names
        bucket_stats.columns = ["_".join(col).strip() for col in bucket_stats.columns.values]
        bucket_stats = bucket_stats.reset_index()

        # Calculate MAE and bias
        bucket_stats["mae"] = (
            df_copy.groupby("edge_bin")
            .apply(lambda x: np.mean(np.abs(x[pred_col] - x[actual_col])))
            .values
        )

        bucket_stats["bias"] = (
            df_copy.groupby("edge_bin").apply(lambda x: np.mean(x[pred_col] - x[actual_col])).values
        )

        # Win rate if available
        if "win" in df_copy.columns and df_copy["win"].notna().sum() > 0:
            bucket_stats["win_rate"] = df_copy.groupby("edge_bin")["win"].mean().values

        # Calibrated stats if available
        if calibrated_col and calibrated_col in df_copy.columns:
            bucket_stats[f"{calibrated_col}_mean"] = (
                df_copy.groupby("edge_bin")[calibrated_col].mean().values
            )
            bucket_stats["mae_calibrated"] = (
                df_copy.groupby("edge_bin")
                .apply(lambda x: np.mean(np.abs(x[calibrated_col] - x[actual_col])))
                .values
            )

        return bucket_stats

    def plot_calibration_curve(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        calibrated_pred: Optional[np.ndarray] = None,
        n_bins: int = 20,
        save_path: Optional[str] = None,
    ):
        """
        Plot calibration curve (predicted vs actual by bins)

        Shows if model is well-calibrated (points near diagonal)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Create bins
        df = pd.DataFrame({"predicted": y_pred, "actual": y_true})

        df["pred_bin"] = pd.qcut(df["predicted"], q=n_bins, labels=False, duplicates="drop")

        bin_stats = (
            df.groupby("pred_bin").agg({"predicted": "mean", "actual": "mean"}).reset_index()
        )

        # Plot 1: Before calibration
        ax = axes[0]
        ax.scatter(bin_stats["predicted"], bin_stats["actual"], s=100, alpha=0.7, color="steelblue")

        # Diagonal line (perfect calibration)
        min_val = min(bin_stats["predicted"].min(), bin_stats["actual"].min())
        max_val = max(bin_stats["predicted"].max(), bin_stats["actual"].max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Calibration"
        )

        ax.set_xlabel("Predicted PRA (binned)", fontsize=12)
        ax.set_ylabel("Actual PRA", fontsize=12)
        ax.set_title("Before Isotonic Calibration", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: After calibration
        if calibrated_pred is not None:
            df["calibrated"] = calibrated_pred
            df["calib_bin"] = pd.qcut(df["calibrated"], q=n_bins, labels=False, duplicates="drop")

            calib_stats = (
                df.groupby("calib_bin").agg({"calibrated": "mean", "actual": "mean"}).reset_index()
            )

            ax = axes[1]
            ax.scatter(
                calib_stats["calibrated"], calib_stats["actual"], s=100, alpha=0.7, color="green"
            )

            min_val = min(calib_stats["calibrated"].min(), calib_stats["actual"].min())
            max_val = max(calib_stats["calibrated"].max(), calib_stats["actual"].max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=2,
                label="Perfect Calibration",
            )

            ax.set_xlabel("Calibrated PRA (binned)", fontsize=12)
            ax.set_ylabel("Actual PRA", fontsize=12)
            ax.set_title("After Isotonic Calibration", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved calibration plot: {save_path}")

        return fig

    def plot_edge_bucket_analysis(
        self, bucket_stats: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Plot edge bucket analysis showing monotonicity issues
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Win rate by edge bucket
        if "win_rate" in bucket_stats.columns:
            ax = axes[0, 0]
            ax.plot(
                bucket_stats["edge_mean"],
                bucket_stats["win_rate"],
                "o-",
                linewidth=2,
                markersize=8,
                color="steelblue",
            )
            ax.axhline(0.5238, color="red", linestyle="--", label="Breakeven (52.38%)")
            ax.set_xlabel("Edge (pts)", fontsize=12)
            ax.set_ylabel("Win Rate", fontsize=12)
            ax.set_title("Win Rate by Edge Bucket", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: MAE by edge bucket
        ax = axes[0, 1]
        ax.plot(
            bucket_stats["edge_mean"],
            bucket_stats["mae"],
            "o-",
            linewidth=2,
            markersize=8,
            color="orange",
        )

        if "mae_calibrated" in bucket_stats.columns:
            ax.plot(
                bucket_stats["edge_mean"],
                bucket_stats["mae_calibrated"],
                "o-",
                linewidth=2,
                markersize=8,
                color="green",
                label="Calibrated",
            )
            ax.legend()

        ax.set_xlabel("Edge (pts)", fontsize=12)
        ax.set_ylabel("MAE (pts)", fontsize=12)
        ax.set_title("MAE by Edge Bucket", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Plot 3: Bias by edge bucket
        ax = axes[1, 0]
        ax.plot(
            bucket_stats["edge_mean"],
            bucket_stats["bias"],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Edge (pts)", fontsize=12)
        ax.set_ylabel("Bias (pts)", fontsize=12)
        ax.set_title("Prediction Bias by Edge Bucket", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Plot 4: Sample count by edge bucket
        ax = axes[1, 1]
        ax.bar(bucket_stats["edge_mean"], bucket_stats["edge_count"], alpha=0.7, color="steelblue")
        ax.set_xlabel("Edge (pts)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Sample Count by Edge Bucket", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  Saved edge bucket plot: {save_path}")

        return fig


def run_full_validation(
    predictions_df: pd.DataFrame,
    pred_col: str = "predicted_PRA",
    actual_col: str = "PRA",
    line_col: Optional[str] = None,
    date_col: str = "GAME_DATE",
    output_dir: str = "data/validation_results",
    block_size: int = 1,
    n_bootstrap: int = 5000,
) -> Dict:
    """
    Run complete statistical validation pipeline

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Betting predictions with columns: GAME_DATE, predicted_PRA, PRA, [line]
    pred_col : str
        Column with predicted values
    actual_col : str
        Column with actual values
    line_col : str
        Column with betting lines (optional)
    date_col : str
        Column with game dates
    output_dir : str
        Directory to save validation results
    block_size : int
        Block size for bootstrap (days)
    n_bootstrap : int
        Number of bootstrap iterations

    Returns:
    --------
    dict with all validation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("STATISTICAL VALIDATION PIPELINE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {output_dir}")

    results = {}

    # Prepare data
    df = predictions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Calculate wins if line available
    if line_col and line_col in df.columns:
        df["edge"] = df[pred_col] - df[line_col]
        df["win"] = (df[actual_col] > df[line_col]).astype(int)
        df["profit"] = np.where(df["win"] == 1, 0.91, -1.0)  # -110 odds
        df["bet_size"] = 1.0  # Unit betting for analysis

        has_betting_data = True
    else:
        has_betting_data = False
        print("\nNote: No betting line column found. Skipping betting validation.")  # noqa: E501

    # ========================================
    # 1. BLOCK BOOTSTRAP VALIDATION
    # ========================================
    if has_betting_data:
        bootstrap = BlockBootstrap(
            df=df, date_col=date_col, block_size=block_size, n_bootstrap=n_bootstrap
        )

        # Win Rate
        wr_result, wr_passes = bootstrap.validate_win_rate(win_col="win", breakeven_wr=0.5238)
        results["win_rate"] = wr_result
        results["win_rate_passes"] = wr_passes

        # Plot win rate distribution
        bootstrap.plot_bootstrap_distribution(
            wr_result, save_path=output_path / "bootstrap_win_rate.png"
        )

        # ROI
        roi_result = bootstrap.validate_roi(profit_col="profit", bet_size_col="bet_size")
        results["roi"] = roi_result

        # Plot ROI distribution
        bootstrap.plot_bootstrap_distribution(
            roi_result, save_path=output_path / "bootstrap_roi.png"
        )

    # ========================================
    # 2. ISOTONIC CALIBRATION
    # ========================================
    print(f"\n{'=' * 60}")
    print("ISOTONIC REGRESSION CALIBRATION")
    print(f"{'=' * 60}")

    # Split data: 70% train, 30% test
    df_sorted = df.sort_values(date_col)
    split_idx = int(len(df_sorted) * 0.7)

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    print("\nData split:")
    print(
        f"  Train: {
            len(train_df):,    } samples ({
            train_df[date_col].min()} to {
                train_df[date_col].max()})"
    )
    print(
        f"  Test:  {
            len(test_df):,    } samples ({
            test_df[date_col].min()} to {
                test_df[date_col].max()})"
    )

    # Fit calibrator on train set
    calibrator = IsotonicCalibration()
    calibrator.fit(
        y_pred=train_df[pred_col].values, y_true=train_df[actual_col].values, verbose=True
    )

    # Apply to test set
    test_df["calibrated_PRA"] = calibrator.predict(test_df[pred_col].values)

    # Evaluate on test set
    mae_before = np.mean(np.abs(test_df[pred_col] - test_df[actual_col]))
    mae_after = np.mean(np.abs(test_df["calibrated_PRA"] - test_df[actual_col]))

    print("\nTest Set Performance:")
    print(f"  MAE before: {mae_before:.3f} pts")
    print(f"  MAE after: {mae_after:.3f} pts")
    print(
        f"  Improvement: {mae_before -
                            mae_after:.3f} pts ({(1 -
                                                  mae_after /
                                                  mae_before) *
                                                 100:.1f}%)"
    )

    results["calibrator"] = calibrator
    results["mae_before"] = mae_before
    results["mae_after"] = mae_after

    # Plot calibration curve
    calibrator.plot_calibration_curve(
        y_pred=test_df[pred_col].values,
        y_true=test_df[actual_col].values,
        calibrated_pred=test_df["calibrated_PRA"].values,
        save_path=output_path / "calibration_curve.png",
    )

    # ========================================
    # 3. EDGE BUCKET ANALYSIS
    # ========================================
    if has_betting_data:
        print(f"\n{'=' * 60}")
        print("EDGE BUCKET ANALYSIS")
        print(f"{'=' * 60}")

        # Before calibration
        bucket_stats_before = calibrator.analyze_edge_buckets(
            df=test_df,
            pred_col=pred_col,
            actual_col=actual_col,
            line_col=line_col,
            calibrated_col="calibrated_PRA",
            n_bins=10,
        )

        print("\nEdge Bucket Statistics (Test Set):")
        print(bucket_stats_before.to_string())

        # Save bucket stats
        bucket_stats_before.to_csv(output_path / "edge_bucket_stats.csv", index=False)

        # Plot edge buckets
        calibrator.plot_edge_bucket_analysis(
            bucket_stats_before, save_path=output_path / "edge_bucket_analysis.png"
        )

        results["edge_buckets"] = bucket_stats_before

    # ========================================
    # 4. SAVE RESULTS
    # ========================================

    # Save calibrated predictions
    test_df.to_csv(output_path / "calibrated_predictions.csv", index=False)
    print(
        f"\n✅ Saved calibrated predictions: {
            output_path /
            'calibrated_predictions.csv'}"
    )

    # Save summary report
    with open(output_path / "validation_report.txt", "w") as f:
        f.write("STATISTICAL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        if has_betting_data:
            f.write("BLOCK BOOTSTRAP RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Win Rate: {wr_result}\n")
            f.write(f"Passes test: {wr_passes}\n\n")
            f.write(f"ROI: {roi_result}\n\n")

        f.write("ISOTONIC CALIBRATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"MAE before: {mae_before:.3f} pts\n")
        f.write(f"MAE after: {mae_after:.3f} pts\n")
        f.write(f"Improvement: {mae_before - mae_after:.3f} pts\n\n")

        if has_betting_data:
            f.write("EDGE BUCKET ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(bucket_stats_before.to_string())

    print(
        f"✅ Saved validation report: {
            output_path /
            'validation_report.txt'}"
    )

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")

    return results
