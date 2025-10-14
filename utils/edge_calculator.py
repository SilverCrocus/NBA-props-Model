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

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict
import pandas as pd


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


def calculate_prediction_std(player_history: pd.DataFrame,
                            predicted_pra: float,
                            global_mae: float = 7.97) -> float:
    """
    Estimate prediction uncertainty (standard deviation)

    Uses player-specific error history when available, falls back to global MAE

    Args:
        player_history: Historical predictions + actuals for this player
        predicted_pra: Current prediction
        global_mae: Overall model MAE (default from walk-forward validation)

    Returns:
        Estimated standard deviation of prediction
    """
    if len(player_history) >= 5:
        # Player-specific error
        player_errors = player_history['error'].abs()
        player_mae = player_errors.mean()

        # MAE to std dev conversion (for normal distribution: std ≈ MAE × 1.25)
        player_std = player_mae * 1.25

        # Add uncertainty for extreme predictions
        # (model is less reliable far from player's average)
        player_avg = player_history['actual_PRA'].mean()
        deviation_from_avg = abs(predicted_pra - player_avg)
        uncertainty_factor = 1 + (deviation_from_avg / 20)  # +5% per 1 PRA deviation

        return player_std * uncertainty_factor
    else:
        # Use global MAE as baseline
        return global_mae * 1.25


def calculate_true_edge(
    predicted_pra: float,
    prediction_std: float,
    betting_line: float,
    over_odds: float,
    under_odds: float,
    min_edge_threshold: float = 0.05
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

    # Calculate our probability using normal distribution
    # P(actual > line) = 1 - CDF(line | μ=pred, σ=std)
    our_over_prob = 1 - norm.cdf(betting_line, loc=predicted_pra, scale=prediction_std)
    our_under_prob = norm.cdf(betting_line, loc=predicted_pra, scale=prediction_std)

    # Expected value = (our_prob × payout) - 1
    # Payout includes original stake: betting $1 returns $X if win
    over_ev = (our_over_prob * over_decimal) - 1
    under_ev = (our_under_prob * under_decimal) - 1

    # Calculate edge as difference between our prob and market prob
    over_edge_prob = our_over_prob - over_prob_implied
    under_edge_prob = our_under_prob - under_prob_implied

    # Confidence based on prediction certainty
    # Narrow std = high confidence, wide std = low confidence
    confidence_score = 1 / (1 + (prediction_std / 5))  # Normalized to 0-1

    # Betting decision
    details = {
        'predicted_pra': predicted_pra,
        'prediction_std': prediction_std,
        'betting_line': betting_line,
        'our_over_prob': our_over_prob,
        'our_under_prob': our_under_prob,
        'market_over_prob': over_prob_implied,
        'market_under_prob': under_prob_implied,
        'over_ev': over_ev,
        'under_ev': under_ev,
        'over_edge_prob': over_edge_prob,
        'under_edge_prob': under_edge_prob,
        'confidence_score': confidence_score
    }

    # Select best bet (highest positive EV above threshold)
    best_ev = max(over_ev, under_ev)

    if over_ev > under_ev and over_ev > min_edge_threshold:
        return over_ev, 'OVER', details
    elif under_ev > over_ev and under_ev > min_edge_threshold:
        return under_ev, 'UNDER', details
    else:
        return 0.0, 'SKIP', details


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
        return 'ELITE'      # >12% EV with confidence
    elif adjusted_edge > 0.08:
        return 'EXCELLENT'  # 8-12% EV
    elif adjusted_edge > 0.05:
        return 'GOOD'       # 5-8% EV
    elif adjusted_edge > 0.03:
        return 'MARGINAL'   # 3-5% EV
    else:
        return 'SKIP'       # <3% EV


def calculate_kelly_fraction(edge: float, win_prob: float,
                            bankroll: float,
                            kelly_fraction: float = 0.25) -> float:
    """
    Calculate optimal bet size using fractional Kelly criterion

    Research shows Full Kelly leads to bankruptcy. Quarter Kelly is optimal.

    Args:
        edge: Expected value (decimal, e.g., 0.08 for 8%)
        win_prob: Our estimated probability of winning
        bankroll: Total bankroll
        kelly_fraction: Conservative fraction (0.25 = Quarter Kelly recommended)

    Returns:
        Bet size in dollars
    """
    if edge <= 0 or win_prob <= 0.5:
        return 0.0

    # Full Kelly formula: f = (edge × p - (1 - p)) / edge
    # Simplified: f = p - (1 - p) / odds
    decimal_odds = 1 + edge  # Convert EV to decimal odds
    full_kelly = (win_prob * decimal_odds - 1) / (decimal_odds - 1)

    # Apply fractional Kelly for safety
    fractional_kelly = full_kelly * kelly_fraction

    # Cap at 2% of bankroll per bet (risk management)
    bet_size = min(fractional_kelly * bankroll, 0.02 * bankroll)

    # Minimum bet size (avoid tiny bets)
    if bet_size < 10:
        return 0.0

    return max(bet_size, 0.0)


# Example usage
if __name__ == "__main__":
    # Test case 1: Strong OVER edge
    print("="*80)
    print("TEST CASE 1: Strong OVER edge")
    print("="*80)

    edge, side, details = calculate_true_edge(
        predicted_pra=35.5,
        prediction_std=6.0,
        betting_line=30.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05
    )

    print(f"Prediction: {details['predicted_pra']:.1f} ± {details['prediction_std']:.1f}")
    print(f"Line: {details['betting_line']:.1f}")
    print(f"Our Over Prob: {details['our_over_prob']:.3f} (Market: {details['market_over_prob']:.3f})")
    print(f"Edge: {edge:.3f} ({edge*100:.1f}%)")
    print(f"Bet: {side}")
    print(f"Quality: {calculate_edge_quality(edge, details['prediction_std'])}")
    print()

    # Test case 2: Weak edge (should skip)
    print("="*80)
    print("TEST CASE 2: Weak edge (should skip)")
    print("="*80)

    edge, side, details = calculate_true_edge(
        predicted_pra=25.3,
        prediction_std=8.0,
        betting_line=25.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05
    )

    print(f"Prediction: {details['predicted_pra']:.1f} ± {details['prediction_std']:.1f}")
    print(f"Line: {details['betting_line']:.1f}")
    print(f"Edge: {edge:.3f} ({edge*100:.1f}%)")
    print(f"Bet: {side}")
    print()

    # Test case 3: Strong UNDER edge
    print("="*80)
    print("TEST CASE 3: Strong UNDER edge")
    print("="*80)

    edge, side, details = calculate_true_edge(
        predicted_pra=18.2,
        prediction_std=5.5,
        betting_line=24.5,
        over_odds=-110,
        under_odds=-110,
        min_edge_threshold=0.05
    )

    print(f"Prediction: {details['predicted_pra']:.1f} ± {details['prediction_std']:.1f}")
    print(f"Line: {details['betting_line']:.1f}")
    print(f"Our Under Prob: {details['our_under_prob']:.3f} (Market: {details['market_under_prob']:.3f})")
    print(f"Edge: {edge:.3f} ({edge*100:.1f}%)")
    print(f"Bet: {side}")
    print(f"Quality: {calculate_edge_quality(edge, details['prediction_std'])}")
    print()

    # Test Kelly bet sizing
    print("="*80)
    print("KELLY BET SIZING EXAMPLE")
    print("="*80)

    bankroll = 10000
    bet_size = calculate_kelly_fraction(
        edge=0.08,  # 8% EV
        win_prob=0.57,  # 57% win probability
        bankroll=bankroll,
        kelly_fraction=0.25  # Quarter Kelly
    )

    print(f"Bankroll: ${bankroll:,.0f}")
    print(f"Edge: 8%")
    print(f"Win Prob: 57%")
    print(f"Recommended Bet Size: ${bet_size:.2f} ({bet_size/bankroll*100:.2f}% of bankroll)")
