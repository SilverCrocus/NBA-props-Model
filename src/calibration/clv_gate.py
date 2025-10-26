"""
CLV Gate Function for Side-Specific Validation

Validates that each side (OVER/UNDER) has proven positive CLV before allowing bets.  # noqa: E501

CLV (Closing Line Value) is the gold standard for model validation:
- Provides statistical significance in 50 bets vs 2000+ needed for ROI
- Research shows: +2% avg CLV → +4% ROI (correlation 0.86)
- Sharp bettors: 55 - 60% beat closing rate, +2 - 3% avg CLV

Gate Logic:
- OVER: Always enabled (baseline, higher volume)
- UNDER: Requires CLV validation before enabling
  * % positive CLV > 55% (last 50 bets)
  * Mean CLV > 0 (last 50 bets)
  * Reason: UNDERs have OT/tail risk, need higher confidence

EV Thresholds:
- OVER: EV >= 4% during stabilization (Tier 1 / 2)
- UNDER: EV >= 6% initially (higher due to risk)
- After CLV validation: UNDER EV can drop to 4%

Author: NBA Props Model
Date: October 25, 2025
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_clv_gate(
    side: str,
    clv_ledger: pd.DataFrame,
    min_bets: int = 50,
    min_pct_positive: float = 0.55,
    min_avg_clv: float = 0.0,
) -> Tuple[bool, Dict[str, float]]:
    """
    Check if side has proven positive CLV.

    Args:
        side: 'OVER' or 'UNDER'
        clv_ledger: DataFrame with CLV data (must have columns:
                    'side', 'clv_pct', 'beat_closing')
        min_bets: Minimum bets required for validation (default 50)
        min_pct_positive: Minimum % of bets that beat closing (default 55%)
        min_avg_clv: Minimum average CLV (default 0%)

    Returns:
        Tuple of (passes_gate, diagnostics_dict)

        passes_gate: True if side passes CLV gate (safe to bet)
        diagnostics_dict: {
            'n_bets': Number of bets with CLV data,
            'pct_beat_closing': % of bets that beat closing,
            'avg_clv': Average CLV,
            'median_clv': Median CLV,
            'std_clv': Standard deviation of CLV,
            'recent_trend': CLV trend (last 20 vs previous 30),
            'passes_pct_threshold': True if pct beats threshold,
            'passes_avg_threshold': True if avg beats threshold,
            'passes_min_bets': True if enough bets
        }

    Example:
        >>> ledger = pd.DataFrame({
        ...     'side': ['UNDER'] * 60,
        ...     'clv_pct': np.random.normal(0.02, 0.03, 60),
        ...     'beat_closing': np.random.binomial(1, 0.58, 60)
        ... })
        >>> passes, diag = check_clv_gate('UNDER', ledger, min_bets=50)
        >>> if passes:
        ...     print(f"UNDER validated: {diag['pct_beat_closing']:.1%} beat closing")  # noqa: E501
        ... else:
        ...     print(f"UNDER blocked: {diag['n_bets']} < {min_bets} bets")
    """
    # OVER always passes (baseline, no CLV gate)
    if side == "OVER":
        return True, {
            "n_bets": 0,
            "pct_beat_closing": np.nan,
            "avg_clv": np.nan,
            "median_clv": np.nan,
            "std_clv": np.nan,
            "recent_trend": np.nan,
            "passes_pct_threshold": True,
            "passes_avg_threshold": True,
            "passes_min_bets": True,
            "reason": "OVER always enabled",
        }

    # Validate ledger columns
    required_cols = ["side", "clv_pct", "beat_closing"]
    if not all(col in clv_ledger.columns for col in required_cols):
        logger.warning(f"CLV ledger missing required columns: {required_cols}")
        return False, {
            "n_bets": 0,
            "pct_beat_closing": np.nan,
            "avg_clv": np.nan,
            "median_clv": np.nan,
            "std_clv": np.nan,
            "recent_trend": np.nan,
            "passes_pct_threshold": False,
            "passes_avg_threshold": False,
            "passes_min_bets": False,
            "reason": "Missing CLV data",
        }

    # Filter to side and valid CLV data
    side_ledger = clv_ledger[
        (clv_ledger["side"] == side)
        & (~clv_ledger["clv_pct"].isna())
        & (~clv_ledger["beat_closing"].isna())
    ].copy()

    # Sort by date (most recent last)
    if "date" in side_ledger.columns:
        side_ledger = side_ledger.sort_values("date")

    # Take last N bets
    recent_bets = side_ledger.tail(min_bets)

    n_bets = len(recent_bets)

    # Check minimum bets
    passes_min_bets = n_bets >= min_bets

    if not passes_min_bets:
        logger.info(f"{side} CLV gate: Insufficient data ({n_bets} < {min_bets} bets)")
        return False, {
            "n_bets": n_bets,
            "pct_beat_closing": np.nan,
            "avg_clv": np.nan,
            "median_clv": np.nan,
            "std_clv": np.nan,
            "recent_trend": np.nan,
            "passes_pct_threshold": False,
            "passes_avg_threshold": False,
            "passes_min_bets": False,
            "reason": f"Need {min_bets - n_bets} more bets",
        }

    # Calculate CLV statistics
    pct_beat_closing = recent_bets["beat_closing"].mean()
    avg_clv = recent_bets["clv_pct"].mean()
    median_clv = recent_bets["clv_pct"].median()
    std_clv = recent_bets["clv_pct"].std()

    # Calculate trend (last 20 vs previous 30)
    if n_bets >= min_bets:
        recent_20 = recent_bets.tail(20)["clv_pct"].mean()
        previous_30 = recent_bets.head(30)["clv_pct"].mean()
        recent_trend = recent_20 - previous_30
    else:
        recent_trend = np.nan

    # Check thresholds
    passes_pct_threshold = pct_beat_closing >= min_pct_positive
    passes_avg_threshold = avg_clv >= min_avg_clv

    # Overall gate decision
    passes_gate = passes_min_bets and passes_pct_threshold and passes_avg_threshold  # noqa: E501

    # Diagnostics
    diagnostics = {
        "n_bets": n_bets,
        "pct_beat_closing": pct_beat_closing,
        "avg_clv": avg_clv,
        "median_clv": median_clv,
        "std_clv": std_clv,
        "recent_trend": recent_trend,
        "passes_pct_threshold": passes_pct_threshold,
        "passes_avg_threshold": passes_avg_threshold,
        "passes_min_bets": passes_min_bets,
        "reason": "",
    }

    if passes_gate:
        diagnostics["reason"] = "CLV validated ✅"
        logger.info(
            f"{side} CLV gate: PASS ({
                pct_beat_closing:.1%} beat closing, {
                avg_clv:+.2%} avg CLV)"
        )
    else:
        if not passes_pct_threshold:
            diagnostics["reason"] = (
                f"% beat closing too low: {
                    pct_beat_closing:.1%} < {
                    min_pct_positive:.0%}"
            )
        elif not passes_avg_threshold:
            diagnostics["reason"] = (
                f"Avg CLV too low: {
                avg_clv:+.2%} < {
                min_avg_clv:+.0%}"
            )
        else:
            diagnostics["reason"] = "Unknown failure"

        logger.warning(f"{side} CLV gate: FAIL - {diagnostics['reason']}")

    return passes_gate, diagnostics


def get_ev_threshold(
    side: str, clv_ledger: pd.DataFrame, tier: int = 1, min_bets: int = 50
) -> Tuple[float, str]:
    """
    Get EV threshold for bet selection based on side and CLV validation status.

    Tiered EV thresholds:
    - Tier 1 (Stabilization): Conservative, proving model
    - Tier 2 (Validation): Moderate, expanding volume
    - Tier 3 (Production): Aggressive, full deployment

    EV Gates:
    - OVER: 4% (all tiers after stabilization)
    - UNDER (pre-CLV): 6% (higher due to OT/tail risk)
    - UNDER (post-CLV): 4% (validated, same as OVER)

    Args:
        side: 'OVER' or 'UNDER'
        clv_ledger: DataFrame with CLV data
        tier: Strategy tier (1, 2, or 3)
        min_bets: Minimum bets for CLV validation

    Returns:
        Tuple of (ev_threshold, reason)

    Example:
        >>> # OVER bet (always 4%)
        >>> ev_threshold, reason = get_ev_threshold('OVER', ledger, tier=1)
        >>> assert ev_threshold == 0.04
        >>>
        >>> # UNDER bet (pre-validation: 6%)
        >>> ev_threshold, reason = get_ev_threshold('UNDER', empty_ledger, tier=1)  # noqa: E501
        >>> assert ev_threshold == 0.06
        >>>
        >>> # UNDER bet (post-validation: 4%)
        >>> ev_threshold, reason = get_ev_threshold('UNDER', validated_ledger, tier=2)
        >>> assert ev_threshold == 0.04
    """
    # OVER: consistent threshold across tiers
    if side == "OVER":
        if tier == 1:
            return 0.04, "OVER Tier 1: Conservative 4% EV"
        elif tier == 2:
            return 0.04, "OVER Tier 2: Standard 4% EV"
        else:  # tier 3
            return 0.03, "OVER Tier 3: Aggressive 3% EV (after CLV validation)"

    # UNDER: depends on CLV validation
    passes_clv, clv_diag = check_clv_gate(side, clv_ledger, min_bets=min_bets)

    if passes_clv:
        # CLV validated: use same threshold as OVER
        if tier == 1:
            return 0.04, "UNDER Tier 1 (CLV validated): Standard 4% EV"
        elif tier == 2:
            return 0.04, "UNDER Tier 2 (CLV validated): Standard 4% EV"
        else:  # tier 3
            return 0.03, "UNDER Tier 3 (CLV validated): Aggressive 3% EV"
    else:
        # CLV not validated: stricter threshold
        reason = clv_diag["reason"]
        if tier == 1:
            return 0.06, f"UNDER Tier 1 (unvalidated): Strict 6% EV ({reason})"
        elif tier == 2:
            return 0.06, f"UNDER Tier 2 (unvalidated): Strict 6% EV ({reason})"
        else:  # tier 3
            # Tier 3 requires CLV validation for UNDER
            return 0.10, f"UNDER Tier 3 (unvalidated): BLOCKED 10% EV ({reason})"  # noqa: E501


def filter_bets_by_clv_gate(
    bets: pd.DataFrame, clv_ledger: pd.DataFrame, tier: int = 1, min_bets: int = 50  # noqa: E501
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter betting opportunities using CLV gate and EV thresholds.

    Applies side-specific EV thresholds based on CLV validation status.

    Args:
        bets: DataFrame with betting opportunities (must have columns:
              'side', 'ev', 'player', 'line')
        clv_ledger: DataFrame with CLV data
        tier: Strategy tier (1, 2, or 3)
        min_bets: Minimum bets for CLV validation

    Returns:
        Tuple of (filtered_bets, filter_stats)

        filtered_bets: DataFrame with bets passing all gates
        filter_stats: {
            'total_opps': Total opportunities,
            'over_opps': OVER opportunities,
            'under_opps': UNDER opportunities,
            'over_passed': OVER bets passing EV threshold,
            'under_passed': UNDER bets passing CLV + EV gates,
            'under_blocked_clv': UNDER bets blocked by CLV gate,
            'under_blocked_ev': UNDER bets blocked by EV threshold,
            'total_passed': Total bets passing all gates
        }

    Example:
        >>> bets = pd.DataFrame({
        ...     'side': ['OVER', 'OVER', 'UNDER', 'UNDER'],
        ...     'ev': [0.05, 0.03, 0.08, 0.05],
        ...     'player': ['A', 'B', 'C', 'D'],
        ...     'line': [25, 30, 35, 40]
        ... })
        >>> filtered, stats = filter_bets_by_clv_gate(bets, clv_ledger, tier=1)
        >>> print(f"Passed: {stats['total_passed']}/{stats['total_opps']}")
    """
    # Initialize stats
    stats = {
        "total_opps": len(bets),
        "over_opps": (bets["side"] == "OVER").sum(),
        "under_opps": (bets["side"] == "UNDER").sum(),
        "over_passed": 0,
        "under_passed": 0,
        "under_blocked_clv": 0,
        "under_blocked_ev": 0,
        "total_passed": 0,
    }

    if len(bets) == 0:
        logger.info("No betting opportunities to filter")
        return bets, stats

    # Validate columns
    required_cols = ["side", "ev"]
    if not all(col in bets.columns for col in required_cols):
        logger.error(f"Bets missing required columns: {required_cols}")
        return bets, stats

    # Initialize pass/fail column
    bets = bets.copy()
    bets["passes_gate"] = False
    bets["gate_reason"] = ""
    bets["ev_threshold"] = np.nan

    # Process OVER bets
    over_mask = bets["side"] == "OVER"
    if over_mask.sum() > 0:
        ev_threshold, reason = get_ev_threshold("OVER", clv_ledger, tier, min_bets)
        bets.loc[over_mask, "ev_threshold"] = ev_threshold

        passes_over = over_mask & (bets["ev"] >= ev_threshold)
        bets.loc[passes_over, "passes_gate"] = True
        bets.loc[passes_over, "gate_reason"] = reason

        fails_over = over_mask & (bets["ev"] < ev_threshold)
        bets.loc[fails_over, "gate_reason"] = (
            f"EV too low: {bets.loc[fails_over, 'ev'].mean():.2%} < {ev_threshold:.0%}"  # noqa: E501
        )

        stats["over_passed"] = passes_over.sum()

    # Process UNDER bets
    under_mask = bets["side"] == "UNDER"
    if under_mask.sum() > 0:
        # Check CLV gate
        passes_clv, clv_diag = check_clv_gate("UNDER", clv_ledger, min_bets)

        if not passes_clv:
            # UNDER blocked by CLV gate
            bets.loc[under_mask, "gate_reason"] = f"CLV gate: {clv_diag['reason']}"
            stats["under_blocked_clv"] = under_mask.sum()
        else:
            # CLV passed, check EV threshold
            ev_threshold, reason = get_ev_threshold("UNDER", clv_ledger, tier, min_bets)
            bets.loc[under_mask, "ev_threshold"] = ev_threshold

            passes_under = under_mask & (bets["ev"] >= ev_threshold)
            bets.loc[passes_under, "passes_gate"] = True
            bets.loc[passes_under, "gate_reason"] = reason

            fails_under = under_mask & (bets["ev"] < ev_threshold)
            bets.loc[fails_under, "gate_reason"] = (
                f"EV too low: {bets.loc[fails_under, 'ev'].mean():.2%} < {ev_threshold:.0%}"  # noqa: E501
            )

            stats["under_passed"] = passes_under.sum()
            stats["under_blocked_ev"] = fails_under.sum()

    # Filter to passing bets
    filtered_bets = bets[bets["passes_gate"]].copy()
    stats["total_passed"] = len(filtered_bets)

    # Log summary
    logger.info(f"CLV Gate Filter (Tier {tier}):")
    logger.info(f"  Total opps: {stats['total_opps']}")
    logger.info(f"  OVER:  {stats['over_passed']}/{stats['over_opps']} passed")
    logger.info(
        f"  UNDER: {stats['under_passed']}/{stats['under_opps']} passed "
        f"({stats['under_blocked_clv']} blocked by CLV, {stats['under_blocked_ev']} blocked by EV)"  # noqa: E501
    )
    logger.info(f"  Total: {stats['total_passed']} bets passing all gates")

    return filtered_bets, stats


# Example usage and testing
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("CLV GATE TESTING")
    print("=" * 80)

    # Simulate CLV ledger
    np.random.seed(42)

    # UNDER bets with good CLV
    under_ledger = pd.DataFrame(
        {
            "date": pd.date_range("2025 - 01 - 01", periods=60, freq="D"),
            "side": ["UNDER"] * 60,
            "clv_pct": np.random.normal(0.025, 0.02, 60),  # 2.5% avg CLV
            # 58% beat closing
            "beat_closing": np.random.binomial(1, 0.58, 60),
        }
    )

    print("\n📊 Simulated UNDER CLV Ledger:")
    print(f"   Bets: {len(under_ledger)}")
    print(f"   % Beat Closing: {under_ledger['beat_closing'].mean():.1%}")
    print(f"   Avg CLV: {under_ledger['clv_pct'].mean():+.2%}")

    # Test CLV gate
    print("\n" + "=" * 80)
    print("TEST 1: CLV Gate Check")
    print("=" * 80)

    passes, diag = check_clv_gate("UNDER", under_ledger, min_bets=50)

    print(f"\nResult: {'✅ PASS' if passes else '❌ FAIL'}")
    print(f"  Bets: {diag['n_bets']}")
    print(f"  % Beat Closing: {diag['pct_beat_closing']:.1%}")
    print(f"  Avg CLV: {diag['avg_clv']:+.2%}")
    print(f"  Median CLV: {diag['median_clv']:+.2%}")
    print(f"  Recent Trend: {diag['recent_trend']:+.2%}")
    print(f"  Reason: {diag['reason']}")

    # Test EV thresholds
    print("\n" + "=" * 80)
    print("TEST 2: EV Threshold Selection")
    print("=" * 80)

    for side in ["OVER", "UNDER"]:
        for tier in [1, 2, 3]:
            ev_threshold, reason = get_ev_threshold(side, under_ledger, tier=tier, min_bets=50)
            print(f"\n{side} Tier {tier}:")
            print(f"  Threshold: {ev_threshold:.0%}")
            print(f"  Reason: {reason}")

    # Test bet filtering
    print("\n" + "=" * 80)
    print("TEST 3: Bet Filtering")
    print("=" * 80)

    # Simulate betting opportunities
    bets = pd.DataFrame(
        {
            "side": ["OVER", "OVER", "OVER", "UNDER", "UNDER", "UNDER"],
            "ev": [0.05, 0.03, 0.08, 0.07, 0.05, 0.09],
            "player": [
                "Player A",
                "Player B",
                "Player C",
                "Player D",
                "Player E",
                "Player F",
            ],  # noqa: E501
            "line": [25, 30, 35, 40, 45, 50],
        }
    )

    print("\nBetting Opportunities:")
    print(bets[["side", "ev", "player", "line"]])

    for tier in [1, 2]:
        print(f"\n--- Tier {tier} Filtering ---")
        filtered, stats = filter_bets_by_clv_gate(bets, under_ledger, tier=tier, min_bets=50)

        print(f"\nPassed ({len(filtered)}/{len(bets)}):")
        if len(filtered) > 0:
            print(filtered[["side", "ev", "player", "ev_threshold", "gate_reason"]])

        print("\nStats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # Test UNDER without CLV validation
    print("\n" + "=" * 80)
    print("TEST 4: UNDER Without CLV Validation")
    print("=" * 80)

    empty_ledger = pd.DataFrame(columns=["side", "clv_pct", "beat_closing"])

    passes_empty, diag_empty = check_clv_gate("UNDER", empty_ledger, min_bets=50)

    print(f"\nResult: {'✅ PASS' if passes_empty else '❌ FAIL'}")
    print(f"  Reason: {diag_empty['reason']}")

    ev_threshold_unvalidated, reason_unvalidated = get_ev_threshold("UNDER", empty_ledger, tier=1)
    print(f"\nEV Threshold (unvalidated): {ev_threshold_unvalidated:.0%}")
    print(f"  Reason: {reason_unvalidated}")

    print("\n" + "=" * 80)
    print("✅ CLV GATE TEST COMPLETE")
    print("=" * 80)
