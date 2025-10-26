"""
Beta Calibrator Without Clipping (with Platt Fallback)

Corrected implementation that:
- Returns raw prob_cal for metrics (no clipping)
- Applies clipping only at bet-time in production
- Uses Beta calibration for sufficient data (n >= 30)
- Falls back to Platt scaling for small samples (n < 30)
- Preserves original row order for validation metrics

Mathematical Background:
- Beta Calibration: Learns a and b parameters of Beta distribution
  P_cal = Beta(a, b).cdf(P_raw) where a, b fit via maximum likelihood
- Platt Scaling: Learns logistic regression P_cal = 1 / (1 + exp(a + b*logit(P_raw)))  # noqa: E501
- Beta is more flexible (3+ parameters) but needs more data
- Platt is more stable (2 parameters) for small samples

Author: NBA Props Model
Date: October 25, 2025
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class BetaCalibrator:
    """
    Beta calibration for probability predictions (with Platt fallback).

    Fits separate calibrators for OVER and UNDER sides.
    Uses Beta calibration when n >= 30, otherwise uses Platt scaling.

    Attributes:
        calibrator_over: Calibrator for OVER bets
        calibrator_under: Calibrator for UNDER bets
        metadata: Training metadata and diagnostics
    """

    def __init__(
        self, min_samples_beta: int = 30, calibrator_type: Literal["auto", "beta", "platt"] = "auto"
    ):
        """
        Initialize Beta Calibrator.

        Args:
            min_samples_beta: Minimum samples required for Beta (else use Platt)  # noqa: E501
            calibrator_type: 'auto' (hybrid), 'beta' (force Beta), 'platt' (force Platt)
        """
        self.min_samples_beta = min_samples_beta
        self.calibrator_type = calibrator_type
        self.calibrator_over = None
        self.calibrator_under = None
        self.metadata = {
            "n_over": 0,
            "n_under": 0,
            "calibrator_type_over": None,
            "calibrator_type_under": None,
            "ece_over_before": np.nan,
            "ece_over_after": np.nan,
            "ece_under_before": np.nan,
            "ece_under_after": np.nan,
            "brier_over_before": np.nan,
            "brier_over_after": np.nan,
            "brier_under_before": np.nan,
            "brier_under_after": np.nan,
        }
        self.is_fitted = False

    def fit(
        self, probs_raw: np.ndarray, outcomes: np.ndarray, sides: np.ndarray, validate: bool = True
    ) -> Dict[str, float]:
        """
        Fit calibrators for OVER and UNDER sides.

        Args:
            probs_raw: Raw predicted probabilities (uncalibrated)
            outcomes: Binary outcomes (1 = win, 0 = loss)
            sides: Array of 'OVER' or 'UNDER' labels
            validate: If True, runs validation and computes metrics

        Returns:
            Dictionary of training metrics

        Raises:
            ValueError: If insufficient data or invalid inputs

        Example:
            >>> cal = BetaCalibrator()
            >>> probs = np.array([0.6, 0.7, 0.55, 0.65])
            >>> outcomes = np.array([1, 1, 0, 1])
            >>> sides = np.array(['OVER', 'OVER', 'UNDER', 'UNDER'])
            >>> metrics = cal.fit(probs, outcomes, sides)
        """
        # Validate inputs
        probs_raw = np.asarray(probs_raw)
        outcomes = np.asarray(outcomes).astype(int)
        sides = np.asarray(sides)

        if len(probs_raw) != len(outcomes) or len(probs_raw) != len(sides):
            raise ValueError("probs_raw, outcomes, and sides must have same length")

        if not all((outcomes == 0) | (outcomes == 1)):
            raise ValueError("outcomes must be binary (0 or 1)")

        # Clip probabilities to avoid numerical issues
        probs_raw = np.clip(probs_raw, 1e-6, 1 - 1e-6)

        # Split by side
        over_mask = sides == "OVER"
        under_mask = sides == "UNDER"

        probs_over = probs_raw[over_mask]
        outcomes_over = outcomes[over_mask]

        probs_under = probs_raw[under_mask]
        outcomes_under = outcomes[under_mask]

        self.metadata["n_over"] = len(probs_over)
        self.metadata["n_under"] = len(probs_under)

        logger.info(
            f"Fitting calibrators: {
                self.metadata['n_over']} OVER, {
                self.metadata['n_under']} UNDER"
        )

        # Fit OVER calibrator
        if len(probs_over) < 10:
            raise ValueError(
                f"Insufficient OVER samples: {
                    len(probs_over)} < 10"
            )

        use_beta_over = (self.calibrator_type == "beta") or (
            self.calibrator_type == "auto"
            and len(probs_over) >= self.min_samples_beta  # noqa: E501
        )

        if use_beta_over:
            logger.info(
                f"Using Beta calibration for OVER ({
                    len(probs_over)} samples)"
            )
            self.calibrator_over = self._fit_beta(probs_over, outcomes_over)
            self.metadata["calibrator_type_over"] = "beta"
        else:
            logger.info(
                f"Using Platt scaling for OVER ({
                    len(probs_over)} samples < {
                    self.min_samples_beta})"
            )
            self.calibrator_over = self._fit_platt(probs_over, outcomes_over)
            self.metadata["calibrator_type_over"] = "platt"

        # Fit UNDER calibrator
        if len(probs_under) < 10:
            raise ValueError(
                f"Insufficient UNDER samples: {
                    len(probs_under)} < 10"
            )

        use_beta_under = (self.calibrator_type == "beta") or (
            self.calibrator_type == "auto"
            and len(probs_under) >= self.min_samples_beta  # noqa: E501
        )

        if use_beta_under:
            logger.info(
                f"Using Beta calibration for UNDER ({
                    len(probs_under)} samples)"
            )
            self.calibrator_under = self._fit_beta(probs_under, outcomes_under)
            self.metadata["calibrator_type_under"] = "beta"
        else:
            logger.info(
                f"Using Platt scaling for UNDER ({
                    len(probs_under)} samples)"
            )
            self.calibrator_under = self._fit_platt(probs_under, outcomes_under)
            self.metadata["calibrator_type_under"] = "platt"

        self.is_fitted = True

        # Validation metrics
        if validate:
            self._compute_validation_metrics(probs_raw, outcomes, sides)

        return self.metadata

    def _fit_beta(self, probs: np.ndarray, outcomes: np.ndarray) -> Dict:
        """
        Fit Beta calibrator using maximum likelihood.

        Beta calibration maps P_raw -> P_cal via:
        P_cal = Beta(a, b).cdf(P_raw)

        Args:
            probs: Raw probabilities
            outcomes: Binary outcomes

        Returns:
            Dict with 'type': 'beta', 'a': float, 'b': float
        """
        # Clip probabilities
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        # Negative log-likelihood function
        def neg_log_likelihood(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10  # Invalid parameters

            # P_cal = Beta(a, b).cdf(P_raw)
            p_cal = beta_dist.cdf(probs, a, b)
            p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)

            # Binary cross-entropy
            loss = -np.sum(outcomes * np.log(p_cal) + (1 - outcomes) * np.log(1 - p_cal))
            return loss

        # Optimize
        # Initial guess: a=1, b=1 (uniform distribution)
        result = minimize(
            neg_log_likelihood, x0=[1.0, 1.0], method="Nelder-Mead", options={"maxiter": 1000}
        )

        if not result.success:
            logger.warning(
                "Beta calibration optimization did not converge, using default params"
            )  # noqa: E501
            a, b = 1.0, 1.0
        else:
            a, b = result.x

        logger.debug(f"Beta calibration: a={a:.3f}, b={b:.3f}")

        return {"type": "beta", "a": a, "b": b}

    def _fit_platt(self, probs: np.ndarray, outcomes: np.ndarray) -> Dict:
        """
        Fit Platt scaling (logistic regression on logit-transformed probabilities).  # noqa: E501

        Platt scaling maps P_raw -> P_cal via:
        P_cal = 1 / (1 + exp(a + b * logit(P_raw)))

        Args:
            probs: Raw probabilities
            outcomes: Binary outcomes

        Returns:
            Dict with 'type': 'platt', 'a': float, 'b': float
        """
        # Clip probabilities
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        # Transform to logit space
        logit_probs = logit(probs).reshape(-1, 1)

        # Fit logistic regression
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit_probs, outcomes)

        a = lr.intercept_[0]
        b = lr.coef_[0][0]

        logger.debug(f"Platt scaling: a={a:.3f}, b={b:.3f}")

        return {"type": "platt", "a": a, "b": b, "model": lr}

    def predict(
        self, probs_raw: np.ndarray, sides: np.ndarray, clip_for_betting: bool = False
    ) -> np.ndarray:
        """
        Apply calibration to raw probabilities.

        IMPORTANT: Does NOT clip by default (for metrics).
        Set clip_for_betting=True only in production pipeline.

        Args:
            probs_raw: Raw predicted probabilities
            sides: Array of 'OVER' or 'UNDER' labels
            clip_for_betting: If True, clips to [0.5, 0.999] for betting

        Returns:
            Calibrated probabilities (same length and order as inputs)

        Raises:
            ValueError: If not fitted

        Example:
            >>> # For validation metrics (no clipping)
            >>> probs_cal = cal.predict(probs_raw, sides, clip_for_betting=False)  # noqa: E501
            >>> ece = compute_ece(probs_cal, outcomes)
            >>>
            >>> # For production betting (with clipping)
            >>> probs_cal = cal.predict(probs_raw, sides, clip_for_betting=True)  # noqa: E501
            >>> bet_if_prob >= 0.55
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        # Validate inputs
        probs_raw = np.asarray(probs_raw)
        sides = np.asarray(sides)

        if len(probs_raw) != len(sides):
            raise ValueError("probs_raw and sides must have same length")

        # Clip to avoid numerical issues
        probs_raw = np.clip(probs_raw, 1e-6, 1 - 1e-6)

        # Initialize output (preserve order)
        probs_cal = np.zeros_like(probs_raw)

        # Apply OVER calibrator
        over_mask = sides == "OVER"
        if over_mask.sum() > 0:
            probs_cal[over_mask] = self._apply_calibrator(
                probs_raw[over_mask], self.calibrator_over
            )

        # Apply UNDER calibrator
        under_mask = sides == "UNDER"
        if under_mask.sum() > 0:
            probs_cal[under_mask] = self._apply_calibrator(
                probs_raw[under_mask], self.calibrator_under
            )

        # Clip for betting (only if requested)
        if clip_for_betting:
            probs_cal = np.clip(probs_cal, 0.5, 0.999)
            logger.debug("Applied betting clip: [0.5, 0.999]")

        return probs_cal

    def _apply_calibrator(self, probs: np.ndarray, calibrator: Dict) -> np.ndarray:
        """
        Apply fitted calibrator to probabilities.

        Args:
            probs: Raw probabilities
            calibrator: Fitted calibrator dict

        Returns:
            Calibrated probabilities
        """
        probs = np.clip(probs, 1e-6, 1 - 1e-6)

        if calibrator["type"] == "beta":
            # P_cal = Beta(a, b).cdf(P_raw)
            a, b = calibrator["a"], calibrator["b"]
            return beta_dist.cdf(probs, a, b)

        elif calibrator["type"] == "platt":
            # P_cal = 1 / (1 + exp(a + b * logit(P_raw)))
            a, b = calibrator["a"], calibrator["b"]
            logit_probs = logit(probs)
            return expit(-(a + b * logit_probs))

        else:
            raise ValueError(f"Unknown calibrator type: {calibrator['type']}")

    def _compute_validation_metrics(
        self, probs_raw: np.ndarray, outcomes: np.ndarray, sides: np.ndarray
    ):
        """
        Compute validation metrics (ECE, Brier) before and after calibration.

        Updates self.metadata with metrics.
        """
        # Get calibrated probabilities (no clipping for metrics)
        probs_cal = self.predict(probs_raw, sides, clip_for_betting=False)

        # Split by side
        over_mask = sides == "OVER"
        under_mask = sides == "UNDER"

        # OVER metrics
        if over_mask.sum() > 0:
            self.metadata["brier_over_before"] = brier_score_loss(
                outcomes[over_mask], probs_raw[over_mask]
            )
            self.metadata["brier_over_after"] = brier_score_loss(
                outcomes[over_mask], probs_cal[over_mask]
            )
            self.metadata["ece_over_before"] = self._compute_ece(
                probs_raw[over_mask], outcomes[over_mask]
            )
            self.metadata["ece_over_after"] = self._compute_ece(
                probs_cal[over_mask], outcomes[over_mask]
            )

        # UNDER metrics
        if under_mask.sum() > 0:
            self.metadata["brier_under_before"] = brier_score_loss(
                outcomes[under_mask], probs_raw[under_mask]
            )
            self.metadata["brier_under_after"] = brier_score_loss(
                outcomes[under_mask], probs_cal[under_mask]
            )
            self.metadata["ece_under_before"] = self._compute_ece(
                probs_raw[under_mask], outcomes[under_mask]
            )
            self.metadata["ece_under_after"] = self._compute_ece(
                probs_cal[under_mask], outcomes[under_mask]
            )

        # Log results
        logger.info("Validation Metrics:")
        logger.info(
            f"  OVER:  Brier {
                self.metadata['brier_over_before']:.4f} -> {
                self.metadata['brier_over_after']:.4f}, "
            f"ECE {
                self.metadata['ece_over_before']:.4f} -> {
                    self.metadata['ece_over_after']:.4f}"
        )
        logger.info(
            f"  UNDER: Brier {
                self.metadata['brier_under_before']:.4f} -> {
                self.metadata['brier_under_after']:.4f}, "
            f"ECE {
                self.metadata['ece_under_before']:.4f} -> {
                    self.metadata['ece_under_after']:.4f}"
        )

    def _compute_ece(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            probs: Predicted probabilities
            outcomes: Binary outcomes
            n_bins: Number of bins

        Returns:
            ECE value (0.0 - 1.0)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        total_samples = len(probs)

        for i in range(n_bins):
            in_bin = bin_indices == i
            n_in_bin = in_bin.sum()

            if n_in_bin == 0:
                continue

            avg_pred_prob = probs[in_bin].mean()
            obs_freq = outcomes[in_bin].mean()
            bin_weight = n_in_bin / total_samples

            ece += np.abs(avg_pred_prob - obs_freq) * bin_weight

        return ece

    def save(self, path: str):
        """
        Save calibrator to disk.

        Args:
            path: File path to save to (e.g., 'models/beta_calibrator.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "calibrator_over": self.calibrator_over,
                    "calibrator_under": self.calibrator_under,
                    "metadata": self.metadata,
                    "min_samples_beta": self.min_samples_beta,
                    "calibrator_type": self.calibrator_type,
                },
                f,
            )

        logger.info(f"✅ Calibrator saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BetaCalibrator":
        """
        Load calibrator from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded BetaCalibrator instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        calibrator = cls(
            min_samples_beta=data.get("min_samples_beta", 30),
            calibrator_type=data.get("calibrator_type", "auto"),
        )
        calibrator.calibrator_over = data["calibrator_over"]
        calibrator.calibrator_under = data["calibrator_under"]
        calibrator.metadata = data["metadata"]
        calibrator.is_fitted = True

        logger.info(f"✅ Calibrator loaded from {path}")
        return calibrator

    def print_summary(self):
        """Print summary of calibration results."""
        if not self.is_fitted:
            print("Calibrator not fitted yet")
            return

        print("=" * 80)
        print("BETA CALIBRATOR SUMMARY")
        print("=" * 80)

        print("\n📊 Sample Sizes:")
        print(
            f"   OVER:  {
                self.metadata['n_over']:,    } samples ({
                self.metadata['calibrator_type_over']})"
        )
        print(
            f"   UNDER: {
                self.metadata['n_under']:,    } samples ({
                self.metadata['calibrator_type_under']})"
        )

        print("\n📈 OVER Calibration:")
        print(
            f"   Brier: {
                self.metadata['brier_over_before']:.4f} -> {
                self.metadata['brier_over_after']:.4f} "
            f"(Δ {
                self.metadata['brier_over_after'] -
                self.metadata['brier_over_before']:+.4f})"
        )
        print(
            f"   ECE:   {
                self.metadata['ece_over_before']:.4f} -> {
                self.metadata['ece_over_after']:.4f} "
            f"(Δ {
                self.metadata['ece_over_after'] -
                self.metadata['ece_over_before']:+.4f})"
        )

        print("\n📈 UNDER Calibration:")
        print(
            f"   Brier: {
                self.metadata['brier_under_before']:.4f} -> {
                self.metadata['brier_under_after']:.4f} "
            f"(Δ {
                self.metadata['brier_under_after'] -
                self.metadata['brier_under_before']:+.4f})"
        )
        print(
            f"   ECE:   {
                self.metadata['ece_under_before']:.4f} -> {
                self.metadata['ece_under_after']:.4f} "
            f"(Δ {
                self.metadata['ece_under_after'] -
                self.metadata['ece_under_before']:+.4f})"
        )

        print("\n" + "=" * 80)


# Example usage and testing
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("BETA CALIBRATOR WITH PLATT FALLBACK")
    print("=" * 80)

    # Simulate data
    np.random.seed(42)

    # OVER: overconfident predictions (n=100, use Beta)
    n_over = 100
    probs_over_raw = np.random.uniform(0.55, 0.75, n_over)
    # Actual win rate is 5% lower than predicted (overconfidence)
    outcomes_over = (np.random.uniform(0, 1, n_over) < (probs_over_raw - 0.05)).astype(int)

    # UNDER: small sample (n=25, use Platt)
    n_under = 25
    probs_under_raw = np.random.uniform(0.55, 0.70, n_under)
    # Actual win rate is 3% lower
    outcomes_under = (np.random.uniform(0, 1, n_under) < (probs_under_raw - 0.03)).astype(int)

    # Combine
    probs_raw = np.concatenate([probs_over_raw, probs_under_raw])
    outcomes = np.concatenate([outcomes_over, outcomes_under])
    sides = np.array(["OVER"] * n_over + ["UNDER"] * n_under)

    print("\n📊 Simulated Data:")
    print(
        f"   OVER:  {n_over} samples, avg prob={
            probs_over_raw.mean():.2%}, actual WR={
            outcomes_over.mean():.2%}"
    )
    print(
        f"   UNDER: {n_under} samples, avg prob={
            probs_under_raw.mean():.2%}, actual WR={
            outcomes_under.mean():.2%}"
    )

    # Fit calibrator (auto-selects Beta/Platt based on sample size)
    print("\n" + "=" * 80)
    print("FITTING CALIBRATOR (AUTO MODE)")
    print("=" * 80)

    cal = BetaCalibrator(min_samples_beta=30, calibrator_type="auto")
    metrics = cal.fit(probs_raw, outcomes, sides, validate=True)

    # Print summary
    cal.print_summary()

    # Test prediction (no clipping for metrics)
    print("\n" + "=" * 80)
    print("PREDICTION TEST (NO CLIPPING)")
    print("=" * 80)

    probs_cal = cal.predict(probs_raw, sides, clip_for_betting=False)

    print("\nOVER predictions:")
    print(
        f"   Raw:        min={
            probs_over_raw.min():.3f}, max={
            probs_over_raw.max():.3f}, mean={
                probs_over_raw.mean():.3f}"
    )
    print(
        f"   Calibrated: min={probs_cal[:n_over].min():.3f}, max={probs_cal[:n_over].max():.3f}, mean={probs_cal[:n_over].mean():.3f}"  # noqa: E501
    )

    print("\nUNDER predictions:")
    print(
        f"   Raw:        min={
            probs_under_raw.min():.3f}, max={
            probs_under_raw.max():.3f}, mean={
                probs_under_raw.mean():.3f}"
    )
    print(
        f"   Calibrated: min={probs_cal[n_over:].min():.3f}, max={probs_cal[n_over:].max():.3f}, mean={probs_cal[n_over:].mean():.3f}"  # noqa: E501
    )

    # Test prediction (with clipping for betting)
    print("\n" + "=" * 80)
    print("PREDICTION TEST (WITH BETTING CLIP)")
    print("=" * 80)

    probs_cal_clipped = cal.predict(probs_raw, sides, clip_for_betting=True)

    print("\nCalibrated with clip [0.5, 0.999]:")
    print(
        f"   OVER:  min={probs_cal_clipped[:n_over].min():.3f}, max={probs_cal_clipped[:n_over].max():.3f}"  # noqa: E501
    )
    print(
        f"   UNDER: min={probs_cal_clipped[n_over:].min():.3f}, max={probs_cal_clipped[n_over:].max():.3f}"  # noqa: E501
    )

    # Save and load test
    print("\n" + "=" * 80)
    print("SAVE/LOAD TEST")
    print("=" * 80)

    test_path = "/tmp/beta_calibrator_test.pkl"
    cal.save(test_path)

    cal_loaded = BetaCalibrator.load(test_path)
    probs_cal_loaded = cal_loaded.predict(probs_raw, sides, clip_for_betting=False)

    # Verify identical
    assert np.allclose(
        probs_cal, probs_cal_loaded
    ), "Loaded calibrator produces different results!"  # noqa: E501
    print("✅ Save/load verified: predictions match")

    print("\n" + "=" * 80)
    print("✅ BETA CALIBRATOR TEST COMPLETE")
    print("=" * 80)
