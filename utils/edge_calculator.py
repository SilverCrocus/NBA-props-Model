"""
Improved Edge Calculation for NBA Props Betting

Fixes the broken edge calculation that was causing inverted performance.

OLD (Broken):
    edge = predicted_PRA - betting_line
    Problem: Doesn't account for prediction uncertainty, odds, or vig

NEW (Fixed):
    - Calculates probability-based expected value
    - Accounts for prediction uncertainty (standard deviation)
    - Incorporates market odds and vig
    - Returns only positive EV bets with confidence scoring
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal odds

    Examples:
        +150 → 2.50 (win $150 on $100 bet)
        -110 → 1.909 (risk $110 to win $100)
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_to_probability(american_odds: float) -> float:
    """
    Convert American odds to implied probability

    Examples:
        -110 → 0.5238 (52.38%)
        +150 → 0.4000 (40.00%)
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def remove_vig(over_implied_prob: float, under_implied_prob: float) -> Tuple[float, float]:
    """
    Remove sportsbook vig (juice) from two-way market to get true probabilities

    Sportsbooks set odds so both sides sum to >100% (e.g., 52.4% + 52.4% = 104.8%).  # noqa: E501
    This function removes the vig to find the "fair" market probability.

    Example:
        Over: -110 → 52.38% implied
        Under: -110 → 52.38% implied
        Total: 104.76% (4.76% vig)

        No-vig: 52.38% / 104.76% = 50.00% each side

    Args:
        over_implied_prob: Market implied probability for OVER
        under_implied_prob: Market implied probability for UNDER

    Returns:
        (no_vig_over_prob, no_vig_under_prob): Fair probabilities without vig
    """
    total_implied = over_implied_prob + under_implied_prob

    if total_implied <= 0:
        # Degenerate case: return 50 / 50
        return 0.5, 0.5

    no_vig_over = over_implied_prob / total_implied
    no_vig_under = under_implied_prob / total_implied

    return no_vig_over, no_vig_under


def calculate_prediction_std(
    player_history: pd.DataFrame,
    predicted_pra: float,
    global_std: float = 7.97,
    shrinkage_lambda: float = 10.0,
) -> float:
    """
    Estimate prediction uncertainty (standard deviation) with empirical Bayes shrinkage

    Uses robust MAD (Median Absolute Deviation) and shrinks toward global mean
    when sample size is small. This prevents overfitting to small samples.

    Formula: σ_i² = (n/(n+λ)) × s_i² + (λ/(n+λ)) × σ_global²
    where s_i = 1.4826 × MAD (robust estimator)

    Args:
        player_history: Historical predictions + actuals for this player
        predicted_pra: Current prediction
        global_std: Overall model standard deviation (default from walk-forward)  # noqa: E501
        shrinkage_lambda: Shrinkage parameter (default 10 games)

    Returns:
        Estimated standard deviation of prediction with shrinkage
    """
    if len(player_history) >= 3:  # Need at least 3 games for MAD
        # Use robust MAD instead of MAE (less sensitive to outliers)
        residuals = player_history["error"].values

        # MAD = median absolute deviation from median
        # 1.4826 factor converts MAD to σ for normal distribution
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        s_i = 1.4826 * mad

        # Empirical Bayes shrinkage
        n = len(player_history)
        weight = n / (n + shrinkage_lambda)

        # Weighted combination of player-specific and global variance
        sigma_squared = weight * (s_i**2) + (1 - weight) * (global_std**2)
        player_std = np.sqrt(sigma_squared)

        # Add uncertainty for extreme predictions (model less reliable far from
        # average)
        player_avg = player_history["actual_PRA"].mean()
        deviation_from_avg = abs(predicted_pra - player_avg)
        # +5% per 1 PRA deviation
        uncertainty_factor = 1 + (deviation_from_avg / 20)

        return player_std * uncertainty_factor
    else:
        # Not enough history: use global std (full shrinkage)
        return global_std


def calculate_true_edge(
    predicted_pra: float,
    prediction_std: float,
    betting_line: float,
    over_odds: float,
    under_odds: float,
    min_edge_threshold: float = 0.05,
) -> Tuple[float, str, Dict]:
    """
    Calculate true expected value accounting for uncertainty and market odds

    Args:
        predicted_pra: Model's PRA prediction
        prediction_std: Uncertainty (standard deviation) of prediction
        betting_line: Bookmaker's over/under line
        over_odds: American odds for OVER bet
        under_odds: American odds for UNDER bet
        min_edge_threshold: Minimum EV required to bet (default 5%)

    Returns:
        (edge, bet_side, details)

        edge: Expected value as decimal (0.08 = 8% EV)
        bet_side: 'OVER', 'UNDER', or 'SKIP'
        details: Dict with diagnostic information
    """

    # Convert odds to decimal and probability
    over_decimal = american_to_decimal(over_odds)
    under_decimal = american_to_decimal(under_odds)

    over_prob_implied = american_to_probability(over_odds)
    under_prob_implied = american_to_probability(under_odds)

    # Remove vig to get fair market probability
    # This is critical: sportsbooks build in 4 - 5% vig, so both sides sum to >100%  # noqa: E501
    # We need to compare against the "true" market price, not the vigged price
    no_vig_over_prob, no_vig_under_prob = remove_vig(over_prob_implied, under_prob_implied)

    # Calculate our probability using normal distribution
    # P(actual > line) = 1 - CDF(line | μ=pred, σ=std)
    our_over_prob = 1 - norm.cdf(betting_line, loc=predicted_pra, scale=prediction_std)
    our_under_prob = norm.cdf(betting_line, loc=predicted_pra, scale=prediction_std)

    # Expected value = (our_prob × payout) - 1
    # Payout includes original stake: betting $1 returns $X if win
    # Use actual decimal odds (not no-vig) for payout calculation
    over_ev = (our_over_prob * over_decimal) - 1
    under_ev = (our_under_prob * under_decimal) - 1

    # Calculate edge vs NO-VIG market (fair comparison)
    # This shows if we have an edge after removing the sportsbook's margin
    over_edge_prob = our_over_prob - no_vig_over_prob
    under_edge_prob = our_under_prob - no_vig_under_prob

    # Confidence based on prediction certainty
    # Narrow std = high confidence, wide std = low confidence
    confidence_score = 1 / (1 + (prediction_std / 5))  # Normalized to 0 - 1

    # Betting decision
    details = {
        "predicted_pra": predicted_pra,
        "prediction_std": prediction_std,
        "betting_line": betting_line,
        "our_over_prob": our_over_prob,
        "our_under_prob": our_under_prob,
        "market_over_prob": over_prob_implied,
        "market_under_prob": under_prob_implied,
        "no_vig_over_prob": no_vig_over_prob,
        "no_vig_under_prob": no_vig_under_prob,
        "over_ev": over_ev,
        "under_ev": under_ev,
        "over_edge_prob": over_edge_prob,
        "under_edge_prob": under_edge_prob,
        "confidence_score": confidence_score,
    }

    # Select best bet (highest positive EV above threshold)
    if over_ev > under_ev and over_ev > min_edge_threshold:
        return over_ev, "OVER", details
    elif under_ev > over_ev and under_ev > min_edge_threshold:
        return under_ev, "UNDER", details
    else:
        return 0.0, "SKIP", details


def calculate_edge_quality(edge: float, prediction_std: float) -> str:
    """
    Classify bet quality based on edge and confidence

    Args:
        edge: Expected value (decimal)
        prediction_std: Prediction uncertainty

    Returns:
        Quality tier: 'ELITE', 'EXCELLENT', 'GOOD', 'MARGINAL', 'SKIP'
    """
    # Adjust edge for uncertainty
    # Wide std = reduce effective edge (less confident)
    adjusted_edge = edge * (1 / (1 + (prediction_std / 7)))

    if adjusted_edge > 0.12:
        return "ELITE"  # >12% EV with confidence
    elif adjusted_edge > 0.08:
        return "EXCELLENT"  # 8 - 12% EV
    elif adjusted_edge > 0.05:
        return "GOOD"  # 5 - 8% EV
    elif adjusted_edge > 0.03:
        return "MARGINAL"  # 3 - 5% EV
    else:
        return "SKIP"  # <3% EV


def calculate_kelly_fraction(
    ev: float,
    decimal_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 0.02,
    min_bet_dollars: float = None,
    ece: float = None,
) -> float:
    """
    Calculate optimal bet size using fractional Kelly criterion

    Research shows Full Kelly leads to bankruptcy. Quarter Kelly is optimal.

    TIER 1 FIX APPLIED: Added ECE penalty to reduce Kelly sizing when
    model is miscalibrated (ECE > 0.05). This prevents betting heavy
    on overconfident predictions that tend to lose.

    Args:
        ev: Expected value (decimal, e.g., 0.08 for 8%)
        decimal_odds: Decimal odds of the bet (e.g., 1.91 for -110)
        bankroll: Total bankroll
        kelly_fraction: Conservative fraction (0.25 = Quarter Kelly recommended)  # noqa: E501
        max_bet_pct: Maximum bet as % of bankroll (default 2%)
        min_bet_dollars: Minimum bet size in dollars
            (None = auto-calculate as 0.5% of bankroll, floor $1)
        ece: Expected Calibration Error (optional). If provided, Kelly is
            penalized when ECE > 0.05. At ECE=0.208, kelly_mult=0 (blocks bets).  # noqa: E501

    Returns:
        Bet size in dollars (0.0 if below minimum threshold or blocked by ECE)
    """
    if ev <= 0 or decimal_odds <= 1.0:
        return 0.0

    # Full Kelly formula: f* = EV / (decimal_odds - 1)
    # This is mathematically equivalent to: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = win prob, q = 1 - p
    full_kelly = ev / (decimal_odds - 1)

    # Apply fractional Kelly for safety (quarter Kelly recommended)
    fractional_kelly = full_kelly * kelly_fraction

    # TIER 1 FIX: Apply ECE penalty if calibration is poor
    if ece is not None and ece > 0.05:
        # Penalty increases linearly from 0% at ECE=0.05 to 100% at ECE=0.10
        # At ECE > 0.10, penalty > 100% → kelly_mult = 0 (blocks all bets)
        # 0 at ECE=0.05, 1 at ECE=0.10, >1 beyond
        ece_penalty = (ece - 0.05) / 0.05
        kelly_mult = max(0.0, min(1.0, 1 - ece_penalty))  # Clamp to [0, 1]

        fractional_kelly = fractional_kelly * kelly_mult

        # Log penalty application (only when significant)
        if kelly_mult < 0.5:
            import warnings

            warnings.warn(
                f"ECE penalty applied: ECE={ece:.3f}, kelly_mult={kelly_mult:.3f}. "  # noqa: E501
                f"Bet size reduced by {(1 - kelly_mult) * 100:.0f}%.",
                UserWarning,
            )

    # Cap at max_bet_pct of bankroll per bet (risk management)
    bet_fraction = min(fractional_kelly, max_bet_pct)

    # Convert to dollar amount
    bet_size = bet_fraction * bankroll

    # Determine minimum bet threshold
    # Auto-calculate: 0.5% of bankroll with $1 floor (scales with bankroll size)  # noqa: E501
    # Examples: $50 → $1, $1000 → $5, $10k → $50
    if min_bet_dollars is None:
        min_bet = max(1.0, bankroll * 0.005)
    else:
        min_bet = min_bet_dollars

    # Filter bets below minimum (avoid dust bets not worth placing)
    if bet_size < min_bet:
        return 0.0

    return max(bet_size, 0.0)


# Example usage
if __name__ == "__main__":
    # Test case 1: Strong OVER edge
    print("=" * 80)
    print("TEST CASE 1: Strong OVER edge")
    print("=" * 80)

    edge, side, details = calculate_true_edge(
        predicted_pra=35.5,
        prediction_std=6.0,
        betting_line=30.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05,
    )

    print(
        f"Prediction: {
            details['predicted_pra']:.1f} ± {
            details['prediction_std']:.1f}"
    )
    print(f"Line: {details['betting_line']:.1f}")
    print(
        f"Our Over Prob: {
            details['our_over_prob']:.3f} (Market: {
            details['market_over_prob']:.3f})"
    )
    print(f"Edge: {edge:.3f} ({edge * 100:.1f}%)")
    print(f"Bet: {side}")
    print(
        f"Quality: {
            calculate_edge_quality(
                edge,
                details['prediction_std'])}"
    )
    print()

    # Test case 2: Weak edge (should skip)
    print("=" * 80)
    print("TEST CASE 2: Weak edge (should skip)")
    print("=" * 80)

    edge, side, details = calculate_true_edge(
        predicted_pra=25.3,
        prediction_std=8.0,
        betting_line=25.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05,
    )

    print(
        f"Prediction: {
            details['predicted_pra']:.1f} ± {
            details['prediction_std']:.1f}"
    )
    print(f"Line: {details['betting_line']:.1f}")
    print(f"Edge: {edge:.3f} ({edge * 100:.1f}%)")
    print(f"Bet: {side}")
    print()

    # Test case 3: Strong UNDER edge
    print("=" * 80)
    print("TEST CASE 3: Strong UNDER edge")
    print("=" * 80)

    edge, side, details = calculate_true_edge(
        predicted_pra=18.2,
        prediction_std=5.5,
        betting_line=24.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05,
    )

    print(
        f"Prediction: {
            details['predicted_pra']:.1f} ± {
            details['prediction_std']:.1f}"
    )
    print(f"Line: {details['betting_line']:.1f}")
    print(
        f"Our Under Prob: {details['our_under_prob']:.3f} "
        f"(Market: {details['market_under_prob']:.3f})"
    )
    print(f"Edge: {edge:.3f} ({edge * 100:.1f}%)")
    print(f"Bet: {side}")
    print(
        f"Quality: {
            calculate_edge_quality(
                edge,
                details['prediction_std'])}"
    )
    print()

    # Test Kelly bet sizing
    print("=" * 80)
    print("KELLY BET SIZING EXAMPLE")
    print("=" * 80)

    bankroll = 10000
    bet_size = calculate_kelly_fraction(
        edge=0.08,  # 8% EV
        win_prob=0.57,  # 57% win probability
        bankroll=bankroll,
        kelly_fraction=0.25,  # Quarter Kelly
    )

    print(f"Bankroll: ${bankroll:,.0f}")
    print("Edge: 8%")
    print("Win Prob: 57%")
    bet_pct = bet_size / bankroll * 100
    print(
        f"Recommended Bet Size: ${
            bet_size:.2f} ({
            bet_pct:.2f}% of bankroll)"
    )
