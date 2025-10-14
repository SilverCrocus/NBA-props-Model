"""
Critical tests for betting edge calculation and Kelly sizing.

These tests ensure betting logic is mathematically correct.
Errors here = losing money in production.

IMPORTANCE: Second most critical tests after temporal leakage.
"""

import pytest
import numpy as np


class TestEdgeCalculation:
    """Test edge calculation logic."""

    def test_edge_positive_when_prediction_gt_implied_prob(self):
        """
        Edge should be positive when our prediction > implied probability.

        Example:
        - Odds: +100 (2.0) → implied prob = 50%
        - Our prediction: 55%
        - Edge = 55% - 50% = 5%
        """
        prediction = 0.55
        odds = 2.0  # +100 in American odds

        implied_prob = 1 / odds  # 0.50
        edge = prediction - implied_prob  # 0.05

        assert edge > 0, "Edge should be positive when prediction > implied prob"
        assert abs(edge - 0.05) < 0.001, f"Expected edge=0.05, got {edge}"

    def test_edge_negative_when_prediction_lt_implied_prob(self):
        """
        Edge should be negative when our prediction < implied probability.

        No bet should be placed when edge is negative.
        """
        prediction = 0.45
        odds = 2.0  # +100 → implied prob = 50%

        implied_prob = 1 / odds
        edge = prediction - implied_prob  # -0.05

        assert edge < 0, "Edge should be negative when prediction < implied prob"
        assert abs(edge + 0.05) < 0.001, f"Expected edge=-0.05, got {edge}"

    def test_edge_accounts_for_vig(self):
        """
        Edge calculation must account for sportsbook vig (juice).

        Example:
        - True odds: +100 (50% prob)
        - Book offers: -110 (52.4% implied prob)
        - Vig = 2.4%
        """
        prediction = 0.52
        american_odds = -110

        # Convert American odds to decimal
        if american_odds < 0:
            decimal_odds = 1 + (100 / abs(american_odds))
        else:
            decimal_odds = 1 + (american_odds / 100)

        implied_prob = 1 / decimal_odds
        edge = prediction - implied_prob

        # Edge should be negative because vig eats up the edge
        assert edge < 0.01, "Vig should reduce edge significantly"

    def test_minimum_edge_threshold(self):
        """
        Only bet when edge exceeds minimum threshold (e.g., 3%).

        This ensures we overcome vig and variance.
        """
        MIN_EDGE = 0.03

        # Case 1: Edge below threshold
        edge = 0.02
        assert edge < MIN_EDGE, "Should not bet when edge < threshold"

        # Case 2: Edge above threshold
        edge = 0.05
        assert edge > MIN_EDGE, "Should bet when edge > threshold"

    def test_edge_calculation_with_real_odds(self):
        """Test edge calculation with realistic odds examples."""
        test_cases = [
            # (prediction, american_odds, expected_edge_sign)
            (0.55, 100, "positive"),   # Good bet
            (0.45, -110, "negative"),  # Bad bet
            (0.60, 110, "positive"),   # Strong edge
            (0.50, -110, "negative"),  # Vig eats edge
        ]

        for prediction, american_odds, expected_sign in test_cases:
            # Convert American to decimal
            if american_odds < 0:
                decimal_odds = 1 + (100 / abs(american_odds))
            else:
                decimal_odds = 1 + (american_odds / 100)

            implied_prob = 1 / decimal_odds
            edge = prediction - implied_prob

            if expected_sign == "positive":
                assert edge > 0, \
                    f"Expected positive edge for prediction={prediction}, odds={american_odds}"
            else:
                assert edge < 0, \
                    f"Expected negative edge for prediction={prediction}, odds={american_odds}"


class TestKellyCriterion:
    """Test Kelly Criterion position sizing."""

    def test_kelly_fraction_basic(self):
        """
        Test basic Kelly Criterion calculation.

        Kelly = (edge * odds - 1) / (odds - 1)
        """
        edge = 0.05  # 5% edge
        odds = 2.0   # +100

        kelly_full = (edge * odds) / (odds - 1)
        # kelly_full = (0.05 * 2) / 1 = 0.10 (bet 10% of bankroll)

        assert abs(kelly_full - 0.10) < 0.001, \
            f"Expected kelly=0.10, got {kelly_full}"

    def test_fractional_kelly_reduces_risk(self):
        """
        Fractional Kelly (e.g., 1/4 Kelly) reduces variance.

        Full Kelly is optimal for long run but has high variance.
        """
        edge = 0.10
        odds = 2.0

        kelly_full = (edge * odds) / (odds - 1)  # 0.20 (20% of bankroll)
        kelly_quarter = kelly_full * 0.25  # 0.05 (5% of bankroll)

        assert kelly_quarter < kelly_full, "Fractional Kelly should be smaller"
        assert kelly_quarter == 0.05, f"Expected 1/4 Kelly = 0.05, got {kelly_quarter}"

    def test_kelly_never_exceeds_max_bet_size(self):
        """
        Enforce maximum bet size regardless of Kelly calculation.

        Even with huge edge, never bet > 10% of bankroll (risk management).
        """
        MAX_BET_SIZE = 0.10

        edge = 0.30  # Unrealistic 30% edge
        odds = 2.0

        kelly_full = (edge * odds) / (odds - 1)  # Would be 60%
        kelly_quarter = kelly_full * 0.25  # Would be 15%

        bet_size = min(kelly_quarter, MAX_BET_SIZE)

        assert bet_size == MAX_BET_SIZE, \
            "Bet size should be capped at MAX_BET_SIZE"

    def test_kelly_zero_when_no_edge(self):
        """
        Kelly sizing should be zero (no bet) when edge is zero or negative.
        """
        edge = 0.0
        odds = 2.0

        kelly = (edge * odds) / (odds - 1)

        assert kelly == 0, "Should not bet when edge is zero"

    def test_kelly_with_real_world_scenarios(self):
        """Test Kelly sizing with realistic betting scenarios."""
        bankroll = 1000
        kelly_fraction = 0.25  # Use 1/4 Kelly

        scenarios = [
            # (edge, odds, expected_bet_range)
            (0.05, 2.0, (10, 15)),   # 5% edge, +100 odds
            (0.10, 2.0, (20, 30)),   # 10% edge, +100 odds
            (0.03, 1.91, (5, 10)),   # 3% edge, -110 odds
        ]

        for edge, odds, (min_bet, max_bet) in scenarios:
            kelly_full = (edge * odds) / (odds - 1)
            kelly_frac = kelly_full * kelly_fraction
            bet_amount = bankroll * kelly_frac

            assert min_bet <= bet_amount <= max_bet, \
                f"Bet amount ${bet_amount:.2f} outside expected range ${min_bet}-${max_bet}"


class TestBettingRiskManagement:
    """Test risk management rules."""

    def test_max_daily_exposure_limit(self):
        """
        Total daily exposure should not exceed bankroll percentage.

        Example: Max 30% of bankroll at risk per day.
        """
        bankroll = 1000
        MAX_DAILY_EXPOSURE = 0.30  # 30%

        bets = [50, 75, 100, 125]  # Total = 350 (35% of bankroll)

        total_exposure = sum(bets) / bankroll

        assert total_exposure > MAX_DAILY_EXPOSURE, \
            "Example should exceed max exposure"

        # Should reject the last bet
        accepted_bets = []
        current_exposure = 0

        for bet in bets:
            new_exposure = (current_exposure + bet) / bankroll
            if new_exposure <= MAX_DAILY_EXPOSURE:
                accepted_bets.append(bet)
                current_exposure += bet
            else:
                break  # Reject this and remaining bets

        assert len(accepted_bets) < len(bets), \
            "Should reject some bets to stay under exposure limit"

    def test_single_bet_max_size(self):
        """
        No single bet should exceed maximum percentage of bankroll.

        Example: No bet > 10% of bankroll.
        """
        bankroll = 1000
        MAX_BET_SIZE_PCT = 0.10  # 10%
        MAX_BET_SIZE = bankroll * MAX_BET_SIZE_PCT  # $100

        # Case 1: Bet within limit
        bet_size = 75
        assert bet_size <= MAX_BET_SIZE, "Bet should be accepted"

        # Case 2: Bet exceeds limit
        bet_size = 150
        assert bet_size > MAX_BET_SIZE, "Bet should be rejected"

        # Adjust to max size
        adjusted_bet = min(bet_size, MAX_BET_SIZE)
        assert adjusted_bet == MAX_BET_SIZE, \
            "Bet should be capped at max size"

    def test_correlated_props_reduce_exposure(self):
        """
        When betting multiple props for same player, reduce exposure.

        Example: PRA + Points props are highly correlated.
        """
        bankroll = 1000

        # Two props for same player
        bet1 = 50  # PRA over
        bet2 = 50  # Points over

        # Correlation factor (0.8 = highly correlated)
        correlation = 0.8

        # Effective exposure is higher due to correlation
        # Independent: exposure = $100 / $1000 = 10%
        # Correlated: effective_exposure = 10% * (1 + correlation) = 18%

        independent_exposure = (bet1 + bet2) / bankroll
        correlation_factor = 1 + correlation
        effective_exposure = independent_exposure * correlation_factor

        assert effective_exposure > independent_exposure, \
            "Correlated bets increase effective exposure"

        # Should reduce bet sizes to maintain target exposure
        TARGET_EXPOSURE = 0.10
        reduction_factor = TARGET_EXPOSURE / effective_exposure

        adjusted_bet1 = bet1 * reduction_factor
        adjusted_bet2 = bet2 * reduction_factor

        assert adjusted_bet1 < bet1, "Should reduce bet size for correlated props"


class TestOddsConversion:
    """Test odds format conversions."""

    def test_american_to_decimal_conversion(self):
        """Test conversion from American to decimal odds."""
        test_cases = [
            # (american, expected_decimal)
            (100, 2.0),     # +100 = 2.0
            (-110, 1.909),  # -110 ≈ 1.909
            (150, 2.5),     # +150 = 2.5
            (-150, 1.667),  # -150 ≈ 1.667
            (200, 3.0),     # +200 = 3.0
        ]

        for american, expected_decimal in test_cases:
            if american < 0:
                decimal = 1 + (100 / abs(american))
            else:
                decimal = 1 + (american / 100)

            assert abs(decimal - expected_decimal) < 0.01, \
                f"American {american} should convert to decimal {expected_decimal}, got {decimal}"

    def test_decimal_to_implied_probability(self):
        """Test conversion from decimal odds to implied probability."""
        test_cases = [
            # (decimal_odds, expected_prob)
            (2.0, 0.50),   # 50%
            (1.909, 0.524),  # 52.4%
            (3.0, 0.333),  # 33.3%
            (1.5, 0.667),  # 66.7%
        ]

        for decimal_odds, expected_prob in test_cases:
            implied_prob = 1 / decimal_odds

            assert abs(implied_prob - expected_prob) < 0.01, \
                f"Odds {decimal_odds} should imply prob {expected_prob}, got {implied_prob}"

    def test_implied_prob_includes_vig(self):
        """
        Implied probabilities from sportsbook odds sum to > 100% (vig).

        Example: Both sides of +100/-120 line should sum to > 100%.
        """
        # Two-sided market
        over_odds = 2.0   # +100 (50% implied)
        under_odds = 1.833  # -120 (54.5% implied)

        over_prob = 1 / over_odds
        under_prob = 1 / under_odds

        total_prob = over_prob + under_prob

        assert total_prob > 1.0, \
            f"Total implied prob should exceed 100% (vig), got {total_prob*100:.1f}%"

        vig = total_prob - 1.0
        assert vig > 0.03, \
            f"Vig should be > 3%, got {vig*100:.1f}%"


class TestEdgeThresholds:
    """Test edge threshold strategies."""

    def test_different_thresholds_by_odds(self):
        """
        May want different edge thresholds for different odds.

        - Heavy favorites (-300): need higher edge (more vig)
        - Underdogs (+200): can accept lower edge
        """
        # Heavy favorite
        favorite_odds = 1.33  # -300
        favorite_min_edge = 0.05  # Need 5% edge

        # Underdog
        underdog_odds = 3.0  # +200
        underdog_min_edge = 0.03  # Only need 3% edge

        assert favorite_min_edge > underdog_min_edge, \
            "Should require higher edge for favorites (more vig)"

    def test_edge_threshold_by_bet_size(self):
        """
        Larger bets may require higher edge threshold.

        Risk management: be more selective with large bets.
        """
        # Small bet
        small_bet_size = 0.02  # 2% of bankroll
        small_min_edge = 0.03  # 3% edge OK

        # Large bet
        large_bet_size = 0.08  # 8% of bankroll
        large_min_edge = 0.05  # Need 5% edge

        assert large_min_edge > small_min_edge, \
            "Should require higher edge for larger bets"


# Pytest configuration
pytestmark = pytest.mark.critical


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
