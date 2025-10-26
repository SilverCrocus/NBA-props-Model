"""
Player-Specific Variance Calculator

Calculates σ_player for each player using historical prediction errors
with empirical Bayes shrinkage toward global mean.

This addresses the critical issue where all players were treated with the same
prediction uncertainty, ignoring that consistent players (e.g., Giannis) have
lower variance than volatile players (e.g., Jordan Poole).
"""

import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


class PlayerVarianceCalculator:
    """
    Calculate and cache player-specific prediction variance

    Uses historical walk-forward results to compute per-player σ with shrinkage.  # noqa: E501
    """

    def __init__(
        self,
        historical_data_path: str = "data/results/walk_forward_leak_free_2024_25.csv",  # noqa: E501
        global_std: float = 7.97,
        shrinkage_lambda: float = 10.0,
        min_games: int = 3,
        min_sigma: float = 3.5,
        max_sigma: float = 14.5,
        heavy_shrink_threshold: int = 10,
    ):
        """
        Initialize player variance calculator with guardrails

        Args:
            historical_data_path: Path to walk-forward OOS results with errors
            global_std: Global model standard deviation (default from validation)  # noqa: E501
            shrinkage_lambda: Shrinkage parameter (higher = more shrinkage)
            min_games: Minimum games required for player-specific variance
            min_sigma: Lower bound for σ (prevent numerical extremes)
            max_sigma: Upper bound for σ (prevent numerical extremes)
            heavy_shrink_threshold: Games below which to use heavier shrinkage
        """
        self.historical_data_path = historical_data_path
        self.global_std = global_std
        self.shrinkage_lambda = shrinkage_lambda
        self.min_games = min_games
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.heavy_shrink_threshold = heavy_shrink_threshold

        # Load historical data
        self._load_historical_data()

        # Verify OOS data
        self._verify_oos_data()

        # Calculate player variances
        self.player_variances = self._calculate_all_player_variances()

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        Normalize player name to handle Jr./Sr./III variations and special characters  # noqa: E501

        Examples:
            "Jimmy Butler III" → "Jimmy Butler"
            "Luka Dončić" → "Luka Doncic"
            "P.J. Washington" → "PJ Washington"
        """
        # Convert special characters to ASCII equivalents (č → c, ñ → n, etc.)
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

        # Remove Jr., Jr, Sr., III, etc.
        name = re.sub(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "", name, flags=re.IGNORECASE)

        # Remove periods from initials (P.J. → PJ)
        name = name.replace(".", "")

        # Remove extra whitespace
        name = " ".join(name.split())

        return name.strip()

    def _load_historical_data(self):
        """Load historical walk-forward results"""
        if not Path(self.historical_data_path).exists():
            print(
                f"⚠️  Historical data not found: {
                    self.historical_data_path}"
            )
            print("   Using global variance for all players")
            self.historical_df = pd.DataFrame()
            return

        self.historical_df = pd.read_csv(self.historical_data_path)

        # Normalize player names to handle Jr./Sr./III variations
        self.historical_df["PLAYER_NAME_NORMALIZED"] = self.historical_df[
            "PLAYER_NAME"
        ].apply(  # noqa: E501
            self._normalize_name
        )

        print(
            f"✅ Loaded {len(self.historical_df):,} historical predictions "
            "for variance calculation"
        )
        print(f"   Players: {self.historical_df['PLAYER_NAME'].nunique()}")
        date_min = self.historical_df["GAME_DATE"].min()
        date_max = self.historical_df["GAME_DATE"].max()
        print(f"   Date range: {date_min} to {date_max}")

    def _verify_oos_data(self):
        """
        Verify that historical data comes from out-of-sample walk-forward predictions  # noqa: E501

        This is CRITICAL: if residuals come from in-sample fits, σ will be too small  # noqa: E501
        and model will be overconfident.
        """
        if len(self.historical_df) == 0:
            return

        # Check that filename suggests OOS data
        if "walk_forward" not in self.historical_data_path.lower():
            print(
                "⚠️  WARNING: Historical data may not be from walk-forward validation"
            )  # noqa: E501
            print("   Ensure residuals are from out-of-sample predictions!")

        # Check for required columns
        required_cols = ["PLAYER_NAME", "error", "GAME_DATE"]
        missing_cols = [
            col for col in required_cols if col not in self.historical_df.columns
        ]  # noqa: E501
        if missing_cols:
            print(f"⚠️  WARNING: Missing required columns: {missing_cols}")
            return

        # Basic sanity checks
        print("✅ OOS Data Verification:")
        has_walk_forward = "walk_forward" in self.historical_data_path.lower()
        print(f"   Filename contains 'walk_forward': {has_walk_forward}")
        print(
            f"   Error column present: {
                'error' in self.historical_df.columns}"
        )
        print(
            f"   Mean error (should be ~0): {
                self.historical_df['error'].mean():.3f}"
        )
        err_std = self.historical_df["error"].std()
        print(
            f"   Std error (should match global): {
                err_std:.2f} "
            f"vs {
                self.global_std:.2f}"
        )

    def _calculate_player_variance(self, player_errors: np.ndarray) -> float:
        """
        Calculate player-specific variance with shrinkage and guardrails

        Uses robust MAD (Median Absolute Deviation) and empirical Bayes shrinkage.  # noqa: E501
        Includes guardrails:
        - Heavy shrinkage for small samples (n < 10)
        - σ bounds [min_sigma, max_sigma] to prevent numerical extremes
        - Robust estimators to handle outliers

        Formula: σ_i² = (n/(n+λ)) × s_i² + (λ/(n+λ)) × σ_global²
        where s_i = 1.4826 × MAD

        Args:
            player_errors: Array of prediction errors for this player (OOS)

        Returns:
            Player-specific standard deviation (clamped to bounds)
        """
        n = len(player_errors)

        if n < self.min_games:
            # Not enough history: use global
            return self.global_std

        # Robust MAD estimator (less sensitive to outliers than std)
        median_error = np.median(player_errors)
        mad = np.median(np.abs(player_errors - median_error))

        # Handle edge case: all errors identical (mad = 0)
        if mad < 0.01:
            return self.global_std

        s_i = 1.4826 * mad  # Convert MAD to σ for normal distribution

        # Heavy shrinkage for small samples (< 10 games)
        if n < self.heavy_shrink_threshold:
            # Use 2x lambda for heavier shrinkage
            effective_lambda = self.shrinkage_lambda * 2
        else:
            effective_lambda = self.shrinkage_lambda

        # Empirical Bayes shrinkage
        weight = n / (n + effective_lambda)

        # Weighted combination
        sigma_squared = weight * (s_i**2) + (1 - weight) * (self.global_std**2)
        sigma = np.sqrt(sigma_squared)

        # Clamp to bounds to prevent numerical extremes
        sigma = np.clip(sigma, self.min_sigma, self.max_sigma)

        return sigma

    def _calculate_all_player_variances(self) -> Dict[str, Tuple[float, int]]:
        """
        Calculate variance for all players in historical data

        Creates two lookups:
        1. Original name → (sigma, n_games)
        2. Normalized name → (sigma, n_games)

        Returns:
            Dict mapping player_name → (sigma, n_games)
        """
        if len(self.historical_df) == 0:
            return {}

        player_variances = {}

        for player_name in self.historical_df["PLAYER_NAME"].unique():
            player_data = self.historical_df[
                self.historical_df["PLAYER_NAME"] == player_name
            ]  # noqa: E501
            player_errors = player_data["error"].values

            sigma = self._calculate_player_variance(player_errors)
            n_games = len(player_data)

            # Store both original and normalized name mappings
            player_variances[player_name] = (sigma, n_games)

            # Also store normalized version for fuzzy matching
            normalized_name = self._normalize_name(player_name)
            if normalized_name != player_name:
                player_variances[normalized_name] = (sigma, n_games)

        return player_variances

    def get_player_variance(self, player_name: str, predicted_pra: float = None) -> float:
        """
        Get variance for a specific player with automatic name normalization

        Args:
            player_name: Player's name (will try both original and normalized)
            predicted_pra: Current prediction (optional, for extreme prediction adjustment)  # noqa: E501

        Returns:
            Player-specific standard deviation (or global if player unknown)
        """
        # Try original name first
        if player_name in self.player_variances:
            sigma, n_games = self.player_variances[player_name]
        else:
            # Try normalized name (handles Jr./Sr./III variations)
            normalized_name = self._normalize_name(player_name)
            if normalized_name in self.player_variances:
                sigma, n_games = self.player_variances[normalized_name]
            else:
                # Unknown player: use global
                return self.global_std

        # Optional: Add uncertainty for extreme predictions
        if predicted_pra is not None and len(self.historical_df) > 0:
            # Try both original and normalized name for player history lookup
            player_data = self.historical_df[
                self.historical_df["PLAYER_NAME"] == player_name
            ]  # noqa: E501
            if len(player_data) == 0:
                # Try normalized name
                normalized_name = self._normalize_name(player_name)
                player_data = self.historical_df[
                    self.historical_df["PLAYER_NAME_NORMALIZED"] == normalized_name  # noqa: E501
                ]

            if len(player_data) > 0:
                player_avg = player_data["PRA"].mean()
                deviation_from_avg = abs(predicted_pra - player_avg)
                # +5% per 1 PRA deviation
                uncertainty_factor = 1 + (deviation_from_avg / 20)
                sigma = sigma * uncertainty_factor

                # CRITICAL FIX: Re-clamp after uncertainty adjustment
                # Bug: uncertainty_factor can push sigma above max_sigma
                # Impact: 62% of bets had σ > 14.5 (e.g., Tyrese Maxey σ=19.1)
                sigma = np.clip(sigma, self.min_sigma, self.max_sigma)

        return sigma

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of player variances

        Returns:
            DataFrame with player variance statistics
        """
        if len(self.player_variances) == 0:
            return pd.DataFrame()

        summary = []
        for player_name, (sigma, n_games) in self.player_variances.items():
            summary.append(
                {
                    "player_name": player_name,
                    "sigma": sigma,
                    "n_games": n_games,
                    "shrinkage_weight": n_games / (n_games + self.shrinkage_lambda),  # noqa: E501
                }
            )

        summary_df = pd.DataFrame(summary).sort_values("sigma")

        print("\n📊 Player Variance Summary:")
        print(f"   Total players: {len(summary_df)}")
        print(f"   Mean σ: {summary_df['sigma'].mean():.2f}")
        print(f"   Median σ: {summary_df['sigma'].median():.2f}")
        print(
            f"   Min σ: {
                summary_df['sigma'].min():.2f} ({
                summary_df.iloc[0]['player_name']})"
        )
        print(
            f"   Max σ: {summary_df['sigma'].max():.2f} ({summary_df.iloc[-1]['player_name']})"
        )  # noqa: E501
        print(f"   Global σ: {self.global_std:.2f}")

        return summary_df


# Singleton instance for production use
_player_variance_calculator = None


def get_player_variance_calculator() -> PlayerVarianceCalculator:
    """Get or create singleton PlayerVarianceCalculator"""
    global _player_variance_calculator

    if _player_variance_calculator is None:
        _player_variance_calculator = PlayerVarianceCalculator()

    return _player_variance_calculator


if __name__ == "__main__":
    # Test the calculator
    print("=" * 80)
    print("TESTING PLAYER VARIANCE CALCULATOR")
    print("=" * 80)

    calc = PlayerVarianceCalculator()

    # Get summary stats
    summary = calc.get_summary_stats()

    print("\n📈 Most Consistent Players (Low σ):")
    print(summary.head(10)[["player_name", "sigma", "n_games"]].to_string(index=False))

    print("\n📉 Most Volatile Players (High σ):")
    print(summary.tail(10)[["player_name", "sigma", "n_games"]].to_string(index=False))

    # Test specific players
    print("\n🔍 Example Player Variances:")
    test_players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Anthony Davis"]
    for player in test_players:
        sigma = calc.get_player_variance(player)
        if player in calc.player_variances:
            _, n_games = calc.player_variances[player]
            print(f"   {player}: σ = {sigma:.2f} ({n_games} games)")
        else:
            print(
                f"   {player}: σ = {
                    sigma:.2f} (using global - not in historical data)"
            )
