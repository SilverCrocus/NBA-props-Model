"""
Isotonic Regression Calibrator for NBA Props Predictions.

This module provides calibration to improve betting performance by ensuring
predicted probabilities match observed frequencies (perfect calibration).

The "large edges underperform" problem occurs when the model is overconfident.
Calibration fixes this by adjusting predictions to match historical accuracy.

Author: NBA Props Model - Phase 1 Week 1
Date: October 14, 2025
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for NBA props predictions.

    This calibrator learns a monotonic mapping from raw model predictions
    to calibrated probabilities that match observed frequencies.

    Usage:
        # Train
        calibrator = IsotonicCalibrator()
        calibrator.fit(predictions, actuals, lines)

        # Apply
        calibrated_preds = calibrator.transform(new_predictions)

        # Evaluate
        metrics = calibrator.evaluate(val_predictions, val_actuals, val_lines)
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Initialize calibrator.

        Args:
            out_of_bounds: How to handle predictions outside training range.
                'clip': Clip to [min, max] (recommended for betting)
                'nan': Return NaN (for research)
        """
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds, increasing=True)
        self.is_fitted = False
        self.training_metrics = {}

    def _convert_to_binary(
        self, predictions: np.ndarray, actuals: np.ndarray, lines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert PRA predictions to binary outcomes (Over/Under).

        Args:
            predictions: Predicted PRA values
            actuals: Actual PRA values
            lines: Betting lines (thresholds)

        Returns:
            predicted_probs: P(Over) for each prediction
            binary_outcomes: 1 if actual > line, 0 otherwise
        """
        # Calculate probability of going over the line
        # Use simple heuristic: P(over) = 1 / (1 + exp(-(pred - line)))
        # This creates a sigmoid centered at the line
        differences = predictions - lines
        predicted_probs = 1 / (1 + np.exp(-differences / 3.0))  # 3.0 is scale parameter

        # Binary outcomes: 1 if actual > line, 0 otherwise
        binary_outcomes = (actuals > lines).astype(int)

        return predicted_probs, binary_outcomes

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Fit the isotonic calibrator on training data.

        Args:
            predictions: Model predictions (PRA values)
            actuals: Actual PRA values
            lines: Betting lines
            validation_split: Fraction to hold out for calibration validation

        Returns:
            Dictionary of training metrics
        """
        logger.info("Fitting isotonic calibrator...")

        # Convert to arrays
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)
        lines = np.asarray(lines)

        # Split into calibration train and validation
        n_samples = len(predictions)
        n_train = int(n_samples * (1 - validation_split))

        # Use temporal split (not random) to respect time-series nature
        train_preds = predictions[:n_train]
        train_actuals = actuals[:n_train]
        train_lines = lines[:n_train]

        val_preds = predictions[n_train:]
        val_actuals = actuals[n_train:]
        val_lines = lines[n_train:]

        # Convert to binary probabilities
        train_probs, train_outcomes = self._convert_to_binary(
            train_preds, train_actuals, train_lines
        )
        val_probs, val_outcomes = self._convert_to_binary(val_preds, val_actuals, val_lines)

        # Fit isotonic regression
        self.calibrator.fit(train_probs, train_outcomes)
        self.is_fitted = True

        # Calculate metrics before calibration
        train_brier_before = brier_score_loss(train_outcomes, train_probs)
        val_brier_before = brier_score_loss(val_outcomes, val_probs)

        # Apply calibration
        train_probs_calibrated = self.calibrator.transform(train_probs)
        val_probs_calibrated = self.calibrator.transform(val_probs)

        # Calculate metrics after calibration
        train_brier_after = brier_score_loss(train_outcomes, train_probs_calibrated)
        val_brier_after = brier_score_loss(val_outcomes, val_probs_calibrated)

        # Store metrics
        self.training_metrics = {
            "n_train": n_train,
            "n_val": len(val_preds),
            "train_brier_before": train_brier_before,
            "train_brier_after": train_brier_after,
            "train_brier_improvement": train_brier_before - train_brier_after,
            "val_brier_before": val_brier_before,
            "val_brier_after": val_brier_after,
            "val_brier_improvement": val_brier_before - val_brier_after,
        }

        logger.info(f"✅ Calibrator fitted on {n_train:,} samples")
        logger.info(
            f"   Train Brier: {train_brier_before:.4f} → {train_brier_after:.4f} "
            f"(Δ {self.training_metrics['train_brier_improvement']:+.4f})"
        )
        logger.info(
            f"   Val Brier: {val_brier_before:.4f} → {val_brier_after:.4f} "
            f"(Δ {self.training_metrics['val_brier_improvement']:+.4f})"
        )

        return self.training_metrics

    def transform(self, predictions: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new predictions.

        Args:
            predictions: Model predictions (PRA values)
            lines: Betting lines

        Returns:
            Calibrated probabilities P(Over)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform. Call fit() first.")

        predictions = np.asarray(predictions)
        lines = np.asarray(lines)

        # Convert to probabilities
        raw_probs, _ = self._convert_to_binary(
            predictions, predictions, lines
        )  # Use preds as dummy actuals

        # Apply calibration
        calibrated_probs = self.calibrator.transform(raw_probs)

        return calibrated_probs

    def evaluate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        confidence_thresholds: list = [0.55, 0.60, 0.65, 0.70],
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality and betting performance.

        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            lines: Betting lines
            confidence_thresholds: Thresholds for edge analysis

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)
        lines = np.asarray(lines)

        # Get raw and calibrated probabilities
        raw_probs, binary_outcomes = self._convert_to_binary(predictions, actuals, lines)
        calibrated_probs = self.calibrator.transform(raw_probs)

        # Overall metrics
        metrics = {
            "brier_before": brier_score_loss(binary_outcomes, raw_probs),
            "brier_after": brier_score_loss(binary_outcomes, calibrated_probs),
            "log_loss_before": log_loss(binary_outcomes, raw_probs, labels=[0, 1]),
            "log_loss_after": log_loss(binary_outcomes, calibrated_probs, labels=[0, 1]),
        }

        # Win rate by confidence threshold
        for threshold in confidence_thresholds:
            # Raw predictions
            mask_raw = raw_probs >= threshold
            if mask_raw.sum() > 0:
                win_rate_raw = binary_outcomes[mask_raw].mean()
                n_bets_raw = mask_raw.sum()
            else:
                win_rate_raw = 0.0
                n_bets_raw = 0

            # Calibrated predictions
            mask_cal = calibrated_probs >= threshold
            if mask_cal.sum() > 0:
                win_rate_cal = binary_outcomes[mask_cal].mean()
                n_bets_cal = mask_cal.sum()
            else:
                win_rate_cal = 0.0
                n_bets_cal = 0

            metrics[f"win_rate_raw_{threshold:.0%}"] = win_rate_raw
            metrics[f"n_bets_raw_{threshold:.0%}"] = n_bets_raw
            metrics[f"win_rate_cal_{threshold:.0%}"] = win_rate_cal
            metrics[f"n_bets_cal_{threshold:.0%}"] = n_bets_cal

        return metrics

    def plot_calibration_curve(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        save_path: str = None,
        n_bins: int = 10,
    ):
        """
        Plot calibration curve (reliability diagram).

        Args:
            predictions: Model predictions
            actuals: Actual outcomes
            lines: Betting lines
            save_path: Path to save plot
            n_bins: Number of bins for reliability diagram
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)
        lines = np.asarray(lines)

        # Get probabilities
        raw_probs, binary_outcomes = self._convert_to_binary(predictions, actuals, lines)
        calibrated_probs = self.calibrator.transform(raw_probs)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Before calibration
        ax1 = axes[0]
        self._plot_reliability_diagram(
            raw_probs, binary_outcomes, n_bins, ax1, "Before Calibration"
        )

        # Plot 2: After calibration
        ax2 = axes[1]
        self._plot_reliability_diagram(
            calibrated_probs, binary_outcomes, n_bins, ax2, "After Calibration"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"✅ Calibration curve saved to {save_path}")

        return fig

    def _plot_reliability_diagram(
        self, probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int, ax: plt.Axes, title: str
    ):
        """Helper function to plot reliability diagram."""
        # Bin probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate mean predicted and observed probability per bin
        bin_means_pred = []
        bin_means_obs = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means_pred.append(probabilities[mask].mean())
                bin_means_obs.append(outcomes[mask].mean())
                bin_counts.append(mask.sum())

        # Plot
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
        ax.scatter(
            bin_means_pred,
            bin_means_obs,
            s=[c / 10 for c in bin_counts],
            alpha=0.7,
            label="Observed",
        )
        ax.plot(bin_means_pred, bin_means_obs, "b-", alpha=0.5)

        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Observed Frequency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    def save(self, path: str):
        """Save calibrator to disk."""
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {"calibrator": self.calibrator, "training_metrics": self.training_metrics}, f
            )

        logger.info(f"✅ Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str) -> "IsotonicCalibrator":
        """Load calibrator from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        calibrator = cls()
        calibrator.calibrator = data["calibrator"]
        calibrator.training_metrics = data["training_metrics"]
        calibrator.is_fitted = True

        logger.info(f"✅ Calibrator loaded from {path}")
        return calibrator


def simulate_betting_performance(
    calibrated_probs: np.ndarray,
    actuals: np.ndarray,
    lines: np.ndarray,
    confidence_threshold: float = 0.55,
    odds: float = -110,
) -> Dict[str, float]:
    """
    Simulate betting performance with calibrated probabilities.

    Args:
        calibrated_probs: Calibrated P(Over) probabilities
        actuals: Actual PRA values
        lines: Betting lines
        confidence_threshold: Minimum confidence to place bet
        odds: Betting odds (default -110 means risk $110 to win $100)

    Returns:
        Dictionary of betting metrics
    """
    # Determine which bets to place
    over_bets = calibrated_probs >= confidence_threshold
    under_bets = (1 - calibrated_probs) >= confidence_threshold

    # Calculate outcomes
    actual_over = actuals > lines

    # Over bets
    over_wins = over_bets & actual_over
    over_losses = over_bets & ~actual_over

    # Under bets
    under_wins = under_bets & ~actual_over
    under_losses = under_bets & actual_over

    # Total bets
    n_over_bets = over_bets.sum()
    n_under_bets = under_bets.sum()
    n_total_bets = n_over_bets + n_under_bets

    # Wins and losses
    n_wins = over_wins.sum() + under_wins.sum()
    n_losses = over_losses.sum() + under_losses.sum()

    # Win rate
    win_rate = n_wins / n_total_bets if n_total_bets > 0 else 0.0

    # ROI calculation (assuming -110 odds: risk 1.1 units to win 1 unit)
    risk_per_bet = 1.1  # Risk $110 to win $100
    win_per_bet = 1.0

    total_risked = n_total_bets * risk_per_bet
    total_won = n_wins * win_per_bet
    total_lost = n_losses * risk_per_bet
    net_profit = total_won - total_lost
    roi = (net_profit / total_risked * 100) if total_risked > 0 else 0.0

    return {
        "n_over_bets": n_over_bets,
        "n_under_bets": n_under_bets,
        "n_total_bets": n_total_bets,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "total_risked": total_risked,
        "net_profit": net_profit,
        "roi": roi,
        "confidence_threshold": confidence_threshold,
    }
