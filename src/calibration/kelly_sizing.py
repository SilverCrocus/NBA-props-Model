"""
Kelly Sizing with Side-Specific Rolling ECE Penalty

Corrected implementation that:
- Computes rolling ECE (last 50 bets) by side
- Clamps multiplier to [0, 1] (prevents negative stakes)
- Applies different penalties for OVER vs UNDER
- Gracefully handles insufficient data (<10 bets)

Mathematical Background:
- Full Kelly: f* = (bp - q) / b = EV / (decimal_odds - 1)
- Fractional Kelly: f = f* × fraction (0.1 - 0.25 recommended)
- ECE Penalty: multiplier = max(0, 1 - ECE/ece_threshold)
- Final Stake: min(f × multiplier × bankroll, max_bet_pct × bankroll)

Author: NBA Props Model
Date: October 25, 2025
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_rolling_ece(
    recent_bets: pd.DataFrame, side: str, window: int = 50, n_bins: int = 10, min_bets: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) on last N bets for given side.

    ECE measures the difference between predicted probabilities and observed frequencies.  # noqa: E501
    Lower ECE = better calibration (predictions match reality).

    Formula:
        ECE = Σ (|P(win | bin) - freq(win | bin)| × P(bin))

    Args:
        recent_bets: DataFrame with recent bets (must have columns:
                    'side', 'prob_cal', 'won')
        side: 'OVER' or 'UNDER'
        window: Rolling window size (default 50 bets)
        n_bins: Number of probability bins (default 10)
        min_bets: Minimum bets required (default 10)

    Returns:
        ECE value (0.0 - 1.0), or np.nan if insufficient data

    Edge Cases:
        - Insufficient data (<min_bets): returns np.nan
        - Missing prob_cal or won: filters out invalid rows
        - All wins or all losses: returns 0.0 (perfect calibration in limit)
        - Empty bin: excluded from ECE calculation

    Example:
        >>> bets = pd.DataFrame({
        ...     'side': ['OVER', 'OVER', 'OVER'],
        ...     'prob_cal': [0.6, 0.7, 0.8],
        ...     'won': [1, 0, 1]
        ... })
        >>> ece = compute_rolling_ece(bets, 'OVER', window=50)
        >>> print(f"ECE: {ece:.3f}")
    """
    # Filter to side and last N bets
    side_bets = recent_bets[recent_bets["side"] == side].tail(window).copy()

    # Validate required columns
    required_cols = ["prob_cal", "won"]
    if not all(col in side_bets.columns for col in required_cols):
        logger.warning(f"Missing required columns for ECE: {required_cols}")
        return np.nan

    # Drop rows with missing values
    side_bets = side_bets.dropna(subset=required_cols)

    # Check minimum data requirement
    if len(side_bets) < min_bets:
        logger.debug(
            f"Insufficient data for {side} ECE: {
                len(side_bets)} < {min_bets}"
        )
        return np.nan

    # Extract probabilities and outcomes
    probs = side_bets["prob_cal"].values
    outcomes = side_bets["won"].astype(int).values

    # Edge case: all wins or all losses (perfect calibration in limit)
    if outcomes.sum() == 0 or outcomes.sum() == len(outcomes):
        logger.debug(f"{side} has all wins or all losses - returning 0.0 ECE")
        return 0.0

    # Create probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate ECE
    ece = 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        # Get samples in this bin
        in_bin = bin_indices == i
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            continue

        # Average predicted probability in bin
        avg_pred_prob = probs[in_bin].mean()

        # Observed frequency in bin
        obs_freq = outcomes[in_bin].mean()

        # Bin weight (proportion of total samples)
        bin_weight = n_in_bin / total_samples

        # Contribution to ECE
        ece += np.abs(avg_pred_prob - obs_freq) * bin_weight

    return ece


def calculate_kelly_side_specific(
    ev: float,
    odds: float,
    bankroll: float,
    side: str,
    recent_bets: pd.DataFrame,
    kelly_fraction: float = 0.1,
    max_bet_pct: float = 0.05,
    ece_threshold_over: float = 0.05,
    ece_threshold_under: float = 0.06,
    min_stake: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Kelly sizing with side-specific rolling ECE penalty.

    Corrected Implementation:
    1. Computes rolling ECE separately for OVER and UNDER
    2. Clamps ECE penalty multiplier to [0, 1]
    3. Applies stricter penalty for UNDER (higher ECE threshold)
    4. Returns 0 if ECE penalty blocks bet

    Mathematical Flow:
        1. Full Kelly: f* = EV / (decimal_odds - 1)
        2. Fractional Kelly: f = f* × kelly_fraction
        3. ECE Penalty:
           - If ECE <= threshold: multiplier = 1.0
           - If ECE > threshold: multiplier = max(0, 1 - (ECE - threshold) / threshold)
        4. Final Stake: min(f × multiplier × bankroll, max_bet_pct × bankroll)

    Args:
        ev: Expected value (as decimal, e.g., 0.05 = 5%)
        odds: Decimal odds (e.g., 1.91)
        bankroll: Current bankroll
        side: 'OVER' or 'UNDER'
        recent_bets: DataFrame with recent bets for ECE calculation
                    Must have columns: ['side', 'prob_cal', 'won']
        kelly_fraction: Fraction of Kelly to bet (0.1 = 10%, conservative)
        max_bet_pct: Maximum bet as % of bankroll (default 5%)
        ece_threshold_over: ECE threshold for OVER bets (default 5%)
        ece_threshold_under: ECE threshold for UNDER bets (default 6% - stricter)  # noqa: E501
        min_stake: Minimum stake to return (default 0.0)

    Returns:
        Tuple of (stake_amount, diagnostics_dict)

        stake_amount: Final bet size in dollars (0 if blocked by ECE or negative EV)  # noqa: E501
        diagnostics_dict: {
            'kelly_base': Full Kelly fraction,
            'kelly_frac': Fractional Kelly (after fraction applied),
            'ece_rolling': Rolling ECE for this side,
            'ece_threshold': ECE threshold used,
            'kelly_mult': ECE penalty multiplier [0, 1],
            'stake_before_cap': Stake before max_bet_pct cap,
            'stake_final': Final stake after all adjustments,
            'blocked_by_ece': True if ECE penalty blocked bet
        }

    Example:
        >>> recent_bets = pd.DataFrame({
        ...     'side': ['OVER'] * 50,
        ...     'prob_cal': np.random.uniform(0.5, 0.7, 50),
        ...     'won': np.random.binomial(1, 0.55, 50)
        ... })
        >>> stake, diag = calculate_kelly_side_specific(
        ...     ev=0.08,
        ...     odds=1.91,
        ...     bankroll=10000,
        ...     side='OVER',
        ...     recent_bets=recent_bets,
        ...     kelly_fraction=0.1
        ... )
        >>> print(f"Stake: ${stake:.2f}, ECE: {diag['ece_rolling']:.3f}")
    """
    # Initialize diagnostics
    diagnostics = {
        "kelly_base": 0.0,
        "kelly_frac": 0.0,
        "ece_rolling": np.nan,
        "ece_threshold": 0.0,
        "kelly_mult": 0.0,
        "stake_before_cap": 0.0,
        "stake_final": 0.0,
        "blocked_by_ece": False,
    }

    # Validate inputs
    if ev <= 0:
        logger.debug(f"Negative or zero EV: {ev:.4f}")
        return 0.0, diagnostics

    if odds <= 1.0:
        logger.warning(f"Invalid odds: {odds:.2f} (must be > 1.0)")
        return 0.0, diagnostics

    if bankroll <= 0:
        logger.warning(f"Invalid bankroll: {bankroll:.2f}")
        return 0.0, diagnostics

    # Step 1: Calculate Full Kelly
    kelly_base = ev / (odds - 1)
    diagnostics["kelly_base"] = kelly_base

    # Step 2: Apply Kelly Fraction
    kelly_frac = kelly_base * kelly_fraction
    diagnostics["kelly_frac"] = kelly_frac

    # Step 3: Compute Rolling ECE for this side
    ece_rolling = compute_rolling_ece(
        recent_bets=recent_bets, side=side, window=50, n_bins=10, min_bets=10
    )
    diagnostics["ece_rolling"] = ece_rolling

    # Step 4: Determine ECE threshold (side-specific)
    ece_threshold = ece_threshold_under if side == "UNDER" else ece_threshold_over  # noqa: E501
    diagnostics["ece_threshold"] = ece_threshold

    # Step 5: Calculate ECE Penalty Multiplier
    if np.isnan(ece_rolling):
        # Insufficient data: use conservative multiplier
        kelly_mult = 0.5  # 50% penalty for uncertainty
        logger.debug(
            f"{side}: Insufficient ECE data, using conservative 0.5 multiplier"
        )  # noqa: E501
    elif ece_rolling <= ece_threshold:
        # Good calibration: no penalty
        kelly_mult = 1.0
    else:
        # Poor calibration: apply penalty
        # multiplier = max(0, 1 - (ece - threshold) / threshold)
        excess_ece = ece_rolling - ece_threshold
        penalty = excess_ece / ece_threshold
        kelly_mult = max(0.0, 1.0 - penalty)

        if kelly_mult == 0.0:
            diagnostics["blocked_by_ece"] = True
            logger.info(
                f"{side} bet BLOCKED by ECE: {
                    ece_rolling:.3f} > {
                    ece_threshold:.3f}"
            )

    diagnostics["kelly_mult"] = kelly_mult

    # Step 6: Calculate Stake (before cap)
    stake_before_cap = kelly_frac * kelly_mult * bankroll
    diagnostics["stake_before_cap"] = stake_before_cap

    # Step 7: Apply Maximum Bet % Cap
    max_stake = max_bet_pct * bankroll
    stake_final = min(stake_before_cap, max_stake)

    # Step 8: Apply Minimum Stake Filter
    if stake_final < min_stake:
        stake_final = 0.0

    diagnostics["stake_final"] = stake_final

    # Logging
    if stake_final > 0:
        logger.info(
            f"{side} Kelly: EV={ev:.2%}, Odds={odds:.2f}, ECE={ece_rolling:.3f}, "  # noqa: E501
            f"Mult={kelly_mult:.2f}, Stake=${stake_final:.2f} "
            f"({stake_final / bankroll * 100:.2f}% of bankroll)"
        )

    return stake_final, diagnostics


def compute_ece_by_side(
    ledger: pd.DataFrame, window: int = 50, n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute rolling ECE separately for OVER and UNDER sides.

    Convenience function for reporting/monitoring.

    Args:
        ledger: Bet ledger with columns ['side', 'prob_cal', 'won']
        window: Rolling window size (default 50)
        n_bins: Number of probability bins (default 10)

    Returns:
        Dictionary with keys 'OVER' and 'UNDER' mapping to ECE values

    Example:
        >>> ledger = pd.DataFrame({
        ...     'side': ['OVER', 'UNDER', 'OVER', 'UNDER'] * 25,
        ...     'prob_cal': np.random.uniform(0.5, 0.7, 100),
        ...     'won': np.random.binomial(1, 0.55, 100)
        ... })
        >>> ece_by_side = compute_ece_by_side(ledger, window=50)
        >>> print(f"OVER ECE: {ece_by_side['OVER']:.3f}")
        >>> print(f"UNDER ECE: {ece_by_side['UNDER']:.3f}")
    """
    results = {}

    for side in ["OVER", "UNDER"]:
        ece = compute_rolling_ece(
            recent_bets=ledger, side=side, window=window, n_bins=n_bins, min_bets=10
        )
        results[side] = ece

    return results


def validate_kelly_inputs(ev: float, odds: float, bankroll: float, side: str) -> Tuple[bool, str]:
    """
    Validate inputs for Kelly sizing.

    Args:
        ev: Expected value
        odds: Decimal odds
        bankroll: Current bankroll
        side: 'OVER' or 'UNDER'

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> valid, msg = validate_kelly_inputs(0.05, 1.91, 10000, 'OVER')
        >>> assert valid == True
        >>>
        >>> valid, msg = validate_kelly_inputs(-0.05, 1.91, 10000, 'OVER')
        >>> assert valid == False
        >>> assert 'EV must be positive' in msg
    """
    if ev <= 0:
        return False, f"EV must be positive, got {ev:.4f}"

    if odds <= 1.0:
        return False, f"Decimal odds must be > 1.0, got {odds:.2f}"

    if bankroll <= 0:
        return False, f"Bankroll must be positive, got {bankroll:.2f}"

    if side not in ["OVER", "UNDER"]:
        return False, f"Side must be 'OVER' or 'UNDER', got '{side}'"

    return True, ""


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("KELLY SIZING WITH SIDE-SPECIFIC ROLLING ECE")
    print("=" * 80)

    # Simulate bet history
    np.random.seed(42)

    # OVER bets: good calibration (ECE ~0.03)
    over_bets = pd.DataFrame(
        {
            "side": ["OVER"] * 60,
            "prob_cal": np.random.uniform(0.52, 0.65, 60),
            # Slightly better than predicted
            "won": np.random.binomial(1, 0.57, 60),
        }
    )

    # UNDER bets: poor calibration (ECE ~0.08)
    under_probs = np.random.uniform(0.52, 0.65, 60)
    # Simulate overconfidence: actual win rate is lower than predicted
    under_outcomes = (np.random.uniform(0, 1, 60) < (under_probs - 0.10)).astype(int)
    under_bets = pd.DataFrame(
        {"side": ["UNDER"] * 60, "prob_cal": under_probs, "won": under_outcomes}  # noqa: E501
    )

    # Combine
    all_bets = pd.concat([over_bets, under_bets], ignore_index=True)

    print("\n📊 Simulated Bet History:")
    print(
        f"   OVER bets: {
            len(over_bets)}, Win rate: {
            over_bets['won'].mean():.1%}"
    )
    print(
        f"   UNDER bets: {
            len(under_bets)}, Win rate: {
            under_bets['won'].mean():.1%}"
    )

    # Calculate ECE by side
    ece_by_side = compute_ece_by_side(all_bets, window=50)
    print("\n📈 Rolling ECE (last 50 bets):")
    print(f"   OVER:  {ece_by_side['OVER']:.3f}")
    print(f"   UNDER: {ece_by_side['UNDER']:.3f}")

    # Test Kelly sizing for OVER bet
    print("\n" + "=" * 80)
    print("TEST CASE 1: OVER Bet (Good Calibration)")
    print("=" * 80)

    stake_over, diag_over = calculate_kelly_side_specific(
        ev=0.08,  # 8% EV
        odds=1.91,  # -110 odds
        bankroll=10000,  # $10k bankroll
        side="OVER",
        recent_bets=all_bets,
        kelly_fraction=0.1,  # 10% of Kelly (Tier 1 / 2)
        max_bet_pct=0.05,  # Cap at 5% of bankroll
    )

    print("\nInputs:")
    print(f"   EV: {0.08:.1%}")
    print("   Odds: 1.91 (-110)")
    print("   Bankroll: $10,000")
    print("   Kelly Fraction: 10% (conservative)")

    print("\nCalculations:")
    print(
        f"   Full Kelly: {
            diag_over['kelly_base']:.4f} ({
            diag_over['kelly_base'] *
            100:.2f}% of bankroll)"
    )
    print(f"   Fractional Kelly: {diag_over['kelly_frac']:.4f}")
    print(f"   Rolling ECE: {diag_over['ece_rolling']:.3f}")
    print(f"   ECE Threshold: {diag_over['ece_threshold']:.3f}")
    print(f"   ECE Multiplier: {diag_over['kelly_mult']:.2f}")

    print("\nResult:")
    print(f"   Stake (before cap): ${diag_over['stake_before_cap']:.2f}")
    print(
        f"   Stake (final): ${
            stake_over:.2f} ({
            stake_over /
            10000 *
            100:.2f}% of bankroll)"
    )
    print(f"   Blocked by ECE: {diag_over['blocked_by_ece']}")

    # Test Kelly sizing for UNDER bet
    print("\n" + "=" * 80)
    print("TEST CASE 2: UNDER Bet (Poor Calibration)")
    print("=" * 80)

    stake_under, diag_under = calculate_kelly_side_specific(
        ev=0.08,  # 8% EV
        odds=1.91,  # -110 odds
        bankroll=10000,  # $10k bankroll
        side="UNDER",
        recent_bets=all_bets,
        kelly_fraction=0.1,
        max_bet_pct=0.05,
        ece_threshold_under=0.06,  # Stricter threshold for UNDER
    )

    print("\nInputs:")
    print(f"   EV: {0.08:.1%}")
    print("   Odds: 1.91 (-110)")
    print("   Bankroll: $10,000")
    print("   Kelly Fraction: 10%")

    print("\nCalculations:")
    print(f"   Full Kelly: {diag_under['kelly_base']:.4f}")
    print(f"   Fractional Kelly: {diag_under['kelly_frac']:.4f}")
    print(f"   Rolling ECE: {diag_under['ece_rolling']:.3f}")
    print(
        f"   ECE Threshold: {
            diag_under['ece_threshold']:.3f} (stricter for UNDER)"
    )
    print(f"   ECE Multiplier: {diag_under['kelly_mult']:.2f}")

    print("\nResult:")
    print(f"   Stake (before cap): ${diag_under['stake_before_cap']:.2f}")
    print(
        f"   Stake (final): ${
            stake_under:.2f} ({
            stake_under /
            10000 *
            100:.2f}% of bankroll)"
    )
    print(f"   Blocked by ECE: {diag_under['blocked_by_ece']}")

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nSame EV (8%), same odds (-110), same bankroll ($10k):")
    print(
        f"   OVER stake:  ${
            stake_over:7.2f} (ECE={
            diag_over['ece_rolling']:.3f}, good)"
    )
    print(
        f"   UNDER stake: ${
            stake_under:7.2f} (ECE={
            diag_under['ece_rolling']:.3f}, poor)"
    )
    print(f"\n   Difference: ${stake_over - stake_under:+.2f}")
    print(
        f"   UNDER penalty: {(1 -
                                stake_under /
                                stake_over) *
                               100:.1f}% stake reduction"
    )

    print("\n" + "=" * 80)
    print("✅ KELLY SIZING TEST COMPLETE")
    print("=" * 80)
