"""
Multi-Criteria Bet Selection Algorithm for Portfolio Diversification

Mathematical Framework for diversified sports betting portfolio construction.
Addresses correlation risk, sensitivity analysis, and EV-band optimization.

Problem:
    - 20 - 40 qualifying bets per day → need to select ~10 - 15 for placement
    - Naive "top N by EV" creates game concentration (5 bets in Charlotte @ Philly)  # noqa: E501
    - Same-game bets are correlated (blowouts, pace, injuries affect all)
    - Missing good bets with slightly lower EV but better diversification

Solution:
    Multi-stage filtering with tie-breaking hierarchy:
    1. Sensitivity filtering: Remove fragile bets (fail ±0.5 line or +10% sigma stress)
    2. Game concentration limits: Max 2 - 3 bets per game
    3. CLV expectancy ranking: Prefer bets likely to beat closing line
    4. EV-band randomization: Diversify within ±2pp of cutoff

Mathematical Soundness:
    - Preserves Kelly-optimal sizing (no changes to stake calculation)
    - Maintains minimum EV threshold (no compromise on edge)
    - Reproducible randomization (seeded by date for auditability)
    - Graceful degradation (handles edge cases: insufficient bets, all from one game)  # noqa: E501

Author: NBA Props Model
Date: October 26, 2025
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BetPortfolioSelector:
    """
    Multi-criteria bet selection for diversified portfolio construction.

    Implements four-stage filtering hierarchy:
    1. GUARD: Sensitivity stress testing (±0.5 line, +10% sigma)
    2. CONSTRAINT: Game concentration limits (max bets per game)
    3. RANK: CLV expectancy (prefer bets likely to beat closing)
    4. DIVERSIFY: EV-band randomization (within ±2pp of cutoff)

    Attributes:
        max_bets_per_game: Maximum bets allowed per game (default 2)
        ev_band_width: EV range for randomization in pp (default 2.0)
        line_stress: Line movement stress test in points (default 0.5)
        sigma_stress: Sigma inflation stress test as % (default 0.10 = +10%)
        seed_date: Date string for reproducible randomization (YYYY-MM-DD)
    """

    def __init__(
        self,
        max_bets_per_game: int = 2,
        ev_band_width: float = 0.02,  # 2 percentage points
        line_stress: float = 0.5,
        sigma_stress: float = 0.10,
        seed_date: Optional[str] = None,
    ):
        """
        Initialize portfolio selector with diversification parameters.

        Args:
            max_bets_per_game: Max bets per game (2 - 3 recommended)
            ev_band_width: EV range for randomization (0.02 = ±2pp)
            line_stress: Line movement stress in points (0.5 = ±0.5 pts)
            sigma_stress: Sigma inflation stress as fraction (0.10 = +10%)
            seed_date: Date for random seed (None = use today)
        """
        self.max_bets_per_game = max_bets_per_game
        self.ev_band_width = ev_band_width
        self.line_stress = line_stress
        self.sigma_stress = sigma_stress
        self.seed_date = seed_date or datetime.now().strftime("%Y-%m-%d")

        # Set random seed for reproducibility (YYYYMMDD as integer)
        seed_int = int(self.seed_date.replace("-", ""))
        np.random.seed(seed_int)

        logger.info(
            "Portfolio Selector initialized: "
            f"max_bets_per_game={max_bets_per_game}, "
            f"ev_band={ev_band_width:.1%}, "
            f"line_stress=±{line_stress:.1f}, "
            f"sigma_stress=+{sigma_stress:.0%}, "
            f"seed={seed_int}"
        )

    def _calculate_stressed_ev(
        self,
        prediction: float,
        sigma: float,
        line: float,
        decimal_odds: float,
        direction: str,
        line_stress: float,
        sigma_stress_mult: float,
    ) -> float:
        """
        Calculate EV under stress conditions (line movement + sigma inflation).

        Stress Test Rationale:
            - Lines can move ±0.5 pts between selection and placement
            - Player variance may be underestimated (injuries, load management)
            - Bets that remain +EV under stress are more robust

        Args:
            prediction: Model PRA prediction
            sigma: Player-specific standard deviation
            line: Betting line
            decimal_odds: Decimal odds for this side
            direction: 'OVER' or 'UNDER'
            line_stress: Line movement to stress (signed, +0.5 for OVER means line goes up)  # noqa: E501
            sigma_stress_mult: Sigma multiplier (1.10 = +10% inflation)

        Returns:
            EV under stressed conditions

        Mathematical Formula:
            - Stressed line: L' = L + (line_stress × direction_sign)
            - Stressed sigma: σ' = σ × sigma_stress_mult
            - P(win) = 1 - CDF(L', μ, σ') for OVER, CDF(L', μ, σ') for UNDER
            - EV = P(win) × (odds - 1) - P(lose)

        Example:
            OVER 35.5 @ -110 (1.91):
            - Stressed line: 36.0 (line moved against us)
            - Stressed sigma: 7.0 → 7.7 (+10% less confident)
            - If EV still positive → bet passes stress test
        """
        # Direction multiplier (+1 for OVER, -1 for UNDER)
        direction_sign = 1 if direction == "OVER" else -1

        # Apply stress: line moves AGAINST us, sigma INCREASES
        stressed_line = line + (line_stress * direction_sign)
        stressed_sigma = sigma * sigma_stress_mult

        # Calculate stressed probability
        if direction == "OVER":
            prob_win = 1 - norm.cdf(stressed_line, loc=prediction, scale=stressed_sigma)
        else:  # UNDER
            prob_win = norm.cdf(stressed_line, loc=prediction, scale=stressed_sigma)

        # Calculate stressed EV
        ev = prob_win * (decimal_odds - 1) - (1 - prob_win)

        return ev

    def apply_sensitivity_filter(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 1: Filter bets that fail sensitivity stress tests.

        Removes fragile bets that lose +EV under realistic market/model stress:
        - Line movement: ±0.5 points (typical closing line movement)
        - Sigma inflation: +10% (model may underestimate variance)

        A robust bet maintains positive EV even if:
        1. Line moves 0.5 pts against us before we can place bet
        2. Player variance is 10% higher than estimated

        Args:
            bets_df: DataFrame with betting opportunities
                    Required columns: ['predicted_PRA', 'player_sigma', 'line',
                                      'decimal_odds', 'direction', 'ev']

        Returns:
            Filtered DataFrame with only bets passing both stress tests

        Implementation Notes:
            - Test each stress independently (not combined worst-case)
            - Log filtered bets for transparency
            - Return empty DataFrame if all bets fail (graceful degradation)
        """
        initial_count = len(bets_df)

        if initial_count == 0:
            logger.warning("Empty input to sensitivity filter")
            return bets_df

        # Create copy to avoid modifying original
        df = bets_df.copy()

        # Calculate stressed EVs
        df["ev_line_stress"] = df.apply(
            lambda row: self._calculate_stressed_ev(
                prediction=row["predicted_PRA"],
                sigma=row["player_sigma"],
                line=row["line"],
                decimal_odds=row["decimal_odds"],
                direction=row["direction"],
                line_stress=self.line_stress,
                sigma_stress_mult=1.0,  # No sigma stress for line test
            ),
            axis=1,
        )

        df["ev_sigma_stress"] = df.apply(
            lambda row: self._calculate_stressed_ev(
                prediction=row["predicted_PRA"],
                sigma=row["player_sigma"],
                line=row["line"],
                decimal_odds=row["decimal_odds"],
                direction=row["direction"],
                line_stress=0.0,  # No line stress for sigma test
                sigma_stress_mult=1.0 + self.sigma_stress,
            ),
            axis=1,
        )

        # Filter: must pass BOTH stress tests
        df["passes_line_stress"] = df["ev_line_stress"] > 0
        df["passes_sigma_stress"] = df["ev_sigma_stress"] > 0
        df["passes_sensitivity"] = (
            df["passes_line_stress"] & df["passes_sigma_stress"]
        )  # noqa: E501

        # Apply filter
        filtered = df[df["passes_sensitivity"]].copy()

        # Log results
        failed_count = initial_count - len(filtered)
        if failed_count > 0:
            logger.info(
                f"Sensitivity filter: {failed_count}/{initial_count} bets failed "  # noqa: E501
                f"(Line: {(~df['passes_line_stress']).sum()}, "
                f"Sigma: {(~df['passes_sigma_stress']).sum()})"
            )

            # Log examples of filtered bets
            failed_bets = df[~df["passes_sensitivity"]]
            for idx, bet in failed_bets.head(3).iterrows():
                logger.debug(
                    f"  Filtered: {
                        bet['player_name']} {
                        bet['direction']} {
                        bet['line']} - "
                    f"EV: {
                        bet['ev']:.2%} → Line stress: {
                        bet['ev_line_stress']:.2%}, "
                    f"Sigma stress: {
                            bet['ev_sigma_stress']:.2%}"
                )

        return filtered

    def apply_game_concentration_limits(
        self, bets_df: pd.DataFrame, max_bets: Optional[int] = None
    ) -> pd.DataFrame:
        """
        STAGE 2: Limit bets per game to avoid concentration risk.

        Same-game bets are highly correlated:
        - Blowouts affect all player props (starters sit in 4Q)
        - Pace changes (overtime, slow tempo) affect all players
        - Injuries mid-game can impact multiple bets

        Strategy:
            - Rank bets within each game by EV (descending)
            - Keep top N bets per game (default 2)
            - This diversifies across games while preserving best edges

        Args:
            bets_df: DataFrame with betting opportunities
                    Required columns: ['game_id', 'ev']
            max_bets: Max bets per game (None = use self.max_bets_per_game)

        Returns:
            Filtered DataFrame with game concentration limits applied

        Edge Cases:
            - All bets from one game: return top N bets (can't diversify further)  # noqa: E501
            - Multiple games with ties: preserve all tied bets at cutoff
        """
        max_bets = max_bets or self.max_bets_per_game
        initial_count = len(bets_df)

        if initial_count == 0:
            logger.warning("Empty input to game concentration filter")
            return bets_df

        # Create game_id if not present (away @ home)
        if "game_id" not in bets_df.columns:
            bets_df["game_id"] = bets_df["away_team"] + " @ " + bets_df["home_team"]

        # Rank bets within each game by EV
        df = bets_df.copy()
        df["rank_in_game"] = df.groupby("game_id")["ev"].rank(ascending=False, method="min")

        # Keep top N per game
        filtered = df[df["rank_in_game"] <= max_bets].copy()

        # Log results
        filtered_count = initial_count - len(filtered)
        if filtered_count > 0:
            games_affected = df[df["rank_in_game"] > max_bets]["game_id"].nunique()
            logger.info(
                f"Game concentration limit: {filtered_count}/{initial_count} bets filtered "  # noqa: E501
                f"({games_affected} games exceeded {max_bets} bet limit)"
            )

            # Show games with high concentration
            games_count = df.groupby("game_id").size()
            high_conc_games = games_count[games_count > max_bets]

            if len(high_conc_games) > 0:
                logger.info(f"  Games with >{max_bets} qualifying bets:")
                for game, count in high_conc_games.items():
                    logger.info(f"    {game}: {count} bets → kept top {max_bets}")

        return filtered.drop(columns=["rank_in_game"])

    def calculate_clv_expectancy(self, bets_df: pd.DataFrame) -> pd.DataFrame:
        """
        STAGE 3: Calculate expected CLV for ranking bets.

        CLV (Closing Line Value) is the gold standard for bet quality.
        Bets with higher CLV expectancy are more likely to be sharp.

        CLV Expectancy Factors:
        1. Edge size: Larger edges more likely to beat closing
        2. Sigma (confidence): Lower sigma = higher confidence = higher CLV
        3. Market efficiency: UNDER bets typically have lower CLV (OT bias)

        Formula:
            CLV_expectancy = (EV / 0.10) × (base_sigma / player_sigma) × side_multiplier

            Where:
            - EV normalization: 10% EV → score of 1.0
            - Sigma adjustment: Lower sigma → higher score
            - Side multiplier: OVER = 1.0, UNDER = 0.9 (10% penalty for OT bias)  # noqa: E501

        Args:
            bets_df: DataFrame with betting opportunities
                    Required columns: ['ev', 'player_sigma', 'direction']

        Returns:
            DataFrame with added 'clv_expectancy' column

        Usage:
            Rank bets by clv_expectancy to prioritize those most likely to
            beat closing line (proxy for bet quality).
        """
        df = bets_df.copy()

        # Base sigma (typical player variance)
        base_sigma = df["player_sigma"].median() if len(df) > 0 else 7.0

        # Calculate components
        ev_component = df["ev"] / 0.10  # Normalize to 10% EV = 1.0
        # Lower sigma = higher score
        sigma_component = base_sigma / df["player_sigma"]

        # Side multiplier (UNDER penalty for OT/tail risk)
        side_multiplier = df["direction"].map({"OVER": 1.0, "UNDER": 0.9})

        # Combined CLV expectancy score
        df["clv_expectancy"] = ev_component * sigma_component * side_multiplier

        logger.debug(
            "CLV expectancy calculated: "
            f"range=[{
                df['clv_expectancy'].min():.2f}, {
                df['clv_expectancy'].max():.2f}], "
            f"median={
                df['clv_expectancy'].median():.2f}"
        )

        return df

    def apply_ev_band_randomization(
        self, bets_df: pd.DataFrame, target_count: int, cutoff_ev: float
    ) -> pd.DataFrame:
        """
        STAGE 4: Randomize selection within EV band near cutoff.

        Problem:
            - Selecting exactly top N bets creates arbitrary cutoffs
            - Bet ranked #11 with 5.1% EV vs #10 with 5.2% EV may be equally good  # noqa: E501
            - Randomization provides diversification benefit

        Solution:
            - Among bets within ±2pp of EV cutoff, randomize selection
            - This diversifies portfolio while maintaining quality threshold
            - Seeded randomization ensures reproducibility

        Example:
            Target 10 bets, cutoff at 5.0% EV:
            - Bets with EV ≥ 7.0%: Always include (clearly above cutoff)
            - Bets with EV 3.0 - 7.0%: Randomize selection (within ±2pp band)
            - Bets with EV < 3.0%: Always exclude (clearly below cutoff)

        Args:
            bets_df: DataFrame with betting opportunities (sorted by priority)
            target_count: Desired number of bets to select
            cutoff_ev: EV threshold for cutoff (e.g., 10th best EV)

        Returns:
            DataFrame with ~target_count bets (may vary ±1 - 2 due to ties)

        Edge Cases:
            - Fewer bets than target: return all
            - All bets in band: randomize all, select target_count
            - No bets in band: return top target_count deterministically
        """
        if len(bets_df) <= target_count:
            logger.info(
                f"EV band randomization: {
                    len(bets_df)} ≤ {target_count} target, "
                "returning all bets"
            )  # noqa: E501
            return bets_df

        df = bets_df.copy()

        # Define EV band around cutoff
        lower_bound = cutoff_ev - self.ev_band_width
        upper_bound = cutoff_ev + self.ev_band_width

        # Segment bets into three groups
        # Clearly above cutoff
        always_include = df[df["ev"] > upper_bound].copy()
        randomize_pool = df[(df["ev"] >= lower_bound) & (df["ev"] <= upper_bound)].copy()
        # Clearly below cutoff
        always_exclude = df[df["ev"] < lower_bound].copy()

        logger.info(
            "EV band randomization: "
            f"Always include: {len(always_include)}, "
            f"Randomize pool: {len(randomize_pool)}, "
            f"Always exclude: {len(always_exclude)}"
        )

        # Calculate how many to select from randomize pool
        needed_from_pool = max(0, target_count - len(always_include))

        if needed_from_pool == 0:
            # Already have enough from always_include
            selected = always_include
        elif needed_from_pool >= len(randomize_pool):
            # Need all from pool + some from exclude
            selected = pd.concat([always_include, randomize_pool], ignore_index=True)
        else:
            # Randomly sample from pool
            sampled = randomize_pool.sample(n=needed_from_pool, random_state=self._get_seed())
            selected = pd.concat([always_include, sampled], ignore_index=True)

            logger.info(
                f"  Sampled {needed_from_pool}/{len(randomize_pool)} bets from randomize pool"  # noqa: E501
            )

        return selected

    def _get_seed(self) -> int:
        """Generate reproducible random seed from date."""
        return int(self.seed_date.replace("-", ""))

    def select_portfolio(
        self, bets_df: pd.DataFrame, target_count: int = 10, min_ev: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Select diversified betting portfolio using multi-criteria algorithm.

        Full Pipeline:
        1. Sensitivity filtering (remove fragile bets)
        2. Game concentration limits (max N per game)
        3. CLV expectancy ranking (prefer sharp bets)
        4. EV-band randomization (diversify within cutoff band)

        Args:
            bets_df: DataFrame with all qualifying bets
                    Required columns: ['player_name', 'predicted_PRA', 'player_sigma',
                                      'line', 'decimal_odds', 'direction', 'ev',  # noqa: E501
                                      'away_team', 'home_team']
            target_count: Target number of bets to select (default 10)
            min_ev: Minimum EV to consider (None = no filter)

        Returns:
            Tuple of (selected_bets, diagnostics)

            selected_bets: DataFrame with selected bets (≈target_count rows)
            diagnostics: Dict with selection statistics
                {
                    'input_count': int,
                    'sensitivity_filtered': int,
                    'game_concentration_filtered': int,
                    'final_selected': int,
                    'avg_ev': float,
                    'games_represented': int,
                    'max_bets_per_game': int,
                    'over_under_split': Dict[str, int]
                }

        Example:
            >>> selector = BetPortfolioSelector(max_bets_per_game=2, seed_date="2025 - 10 - 26")
            >>> selected, diag = selector.select_portfolio(all_bets_df, target_count=10)
            >>> print(f"Selected {diag['final_selected']} bets from {diag['games_represented']} games")
        """
        initial_count = len(bets_df)

        if initial_count == 0:
            logger.warning("No bets to select from")
            return pd.DataFrame(), {
                "input_count": 0,
                "sensitivity_filtered": 0,
                "game_concentration_filtered": 0,
                "final_selected": 0,
                "avg_ev": 0.0,
                "games_represented": 0,
                "max_bets_per_game": 0,
                "over_under_split": {"OVER": 0, "UNDER": 0},
            }

        logger.info(f"Starting portfolio selection: {initial_count} input bets")

        # Apply minimum EV filter if specified
        if min_ev is not None:
            bets_df = bets_df[bets_df["ev"] >= min_ev].copy()
            logger.info(
                f"  Min EV filter: {
                    len(bets_df)} bets remain (EV ≥ {
                    min_ev:.1%})"
            )

        # STAGE 1: Sensitivity filtering
        df = self.apply_sensitivity_filter(bets_df)
        sensitivity_filtered = initial_count - len(df)

        if len(df) == 0:
            logger.warning("All bets failed sensitivity filter")
            return pd.DataFrame(), {
                "input_count": initial_count,
                "sensitivity_filtered": sensitivity_filtered,
                "game_concentration_filtered": 0,
                "final_selected": 0,
                "avg_ev": 0.0,
                "games_represented": 0,
                "max_bets_per_game": 0,
                "over_under_split": {"OVER": 0, "UNDER": 0},
            }

        # STAGE 2: Game concentration limits
        df = self.apply_game_concentration_limits(df)
        game_conc_filtered = len(bets_df) - sensitivity_filtered - len(df)

        # STAGE 3: CLV expectancy ranking
        df = self.calculate_clv_expectancy(df)
        df = df.sort_values("clv_expectancy", ascending=False)

        # STAGE 4: EV-band randomization
        if len(df) > target_count:
            cutoff_ev = (
                df.iloc[target_count - 1]["ev"] if target_count <= len(df) else df["ev"].min()
            )
            df = self.apply_ev_band_randomization(df, target_count, cutoff_ev)

        # Final selection
        selected = df.head(target_count).copy()

        # Calculate diagnostics
        diagnostics = {
            "input_count": initial_count,
            "sensitivity_filtered": sensitivity_filtered,
            "game_concentration_filtered": game_conc_filtered,
            "final_selected": len(selected),
            "avg_ev": selected["ev"].mean() if len(selected) > 0 else 0.0,
            "games_represented": (
                selected["game_id"].nunique() if len(selected) > 0 else 0
            ),  # noqa: E501
            "max_bets_per_game": (
                selected.groupby("game_id").size().max() if len(selected) > 0 else 0
            ),  # noqa: E501
            "over_under_split": {
                "OVER": (
                    (selected["direction"] == "OVER").sum() if len(selected) > 0 else 0
                ),  # noqa: E501
                "UNDER": (
                    (selected["direction"] == "UNDER").sum() if len(selected) > 0 else 0
                ),  # noqa: E501
            },
        }

        # Log summary
        logger.info(
            "Portfolio selection complete: "
            f"{diagnostics['final_selected']} bets selected from "
            f"{diagnostics['games_represented']} games "
            f"(avg EV: {diagnostics['avg_ev']:.2%})"
        )
        logger.info(
            "  Filters applied: "
            f"Sensitivity: -{sensitivity_filtered}, "
            f"Game concentration: -{game_conc_filtered}"
        )
        logger.info(
            "  Direction split: "
            f"{diagnostics['over_under_split']['OVER']} OVER / "
            f"{diagnostics['over_under_split']['UNDER']} UNDER"
        )

        return selected, diagnostics


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("BET PORTFOLIO SELECTOR - EXAMPLE")
    print("=" * 80)

    # Simulate 30 qualifying bets from 8 games
    np.random.seed(42)

    games = [
        "CHI @ PHI",
        "CHI @ PHI",
        "CHI @ PHI",
        "CHI @ PHI",
        "CHI @ PHI",  # High concentration
        "LAL @ BOS",
        "LAL @ BOS",
        "LAL @ BOS",
        "MIA @ NYK",
        "MIA @ NYK",
        "MIA @ NYK",
        "DEN @ GSW",
        "DEN @ GSW",
        "DAL @ PHX",
        "DAL @ PHX",
        "MEM @ SAC",
        "ATL @ CLE",
        "ATL @ CLE",
        "POR @ UTA",
    ]

    # Extend to 30 bets
    while len(games) < 30:
        games.extend(["CHI @ PHI", "LAL @ BOS"])

    games = games[:30]

    # Generate bet data
    bets_data = []
    for i, game in enumerate(games):
        away, home = game.split(" @ ")

        # Simulate bet characteristics
        prediction = np.random.uniform(25, 40)
        line = prediction + np.random.uniform(-3, 3)
        sigma = np.random.uniform(5.5, 8.5)

        # Calculate probability and EV
        direction = np.random.choice(["OVER", "UNDER"])
        if direction == "OVER":
            prob = 1 - norm.cdf(line, loc=prediction, scale=sigma)
        else:
            prob = norm.cdf(line, loc=prediction, scale=sigma)

        decimal_odds = 1.91  # -110
        ev = prob * (decimal_odds - 1) - (1 - prob)

        # Add some noise to EV
        ev = ev + np.random.uniform(-0.02, 0.02)

        bets_data.append(
            {
                "player_name": f"Player_{i}",
                "predicted_PRA": prediction,
                "player_sigma": sigma,
                "line": line,
                "decimal_odds": decimal_odds,
                "direction": direction,
                "ev": max(0.01, ev),  # Ensure positive
                "away_team": away,
                "home_team": home,
                "game_id": game,
            }
        )

    bets_df = pd.DataFrame(bets_data)

    print(f"\n📊 Input: {len(bets_df)} qualifying bets")
    print(f"   Games: {bets_df['game_id'].nunique()}")
    print(f"   Avg EV: {bets_df['ev'].mean():.2%}")
    print(
        f"   EV range: [{
            bets_df['ev'].min():.2%}, {
            bets_df['ev'].max():.2%}]"
    )

    # Show game concentration
    game_counts = bets_df.groupby("game_id").size().sort_values(ascending=False)
    print("\n📋 Bets per game:")
    for game, count in game_counts.items():
        print(f"   {game}: {count} bets")

    # Initialize selector
    selector = BetPortfolioSelector(
        max_bets_per_game=2,
        ev_band_width=0.02,
        line_stress=0.5,
        sigma_stress=0.10,
        seed_date="2025 - 10 - 26",
    )

    # Select portfolio
    selected, diagnostics = selector.select_portfolio(bets_df=bets_df, target_count=10, min_ev=0.02)

    # Print results
    print("\n" + "=" * 80)
    print("PORTFOLIO SELECTION RESULTS")
    print("=" * 80)

    print("\n📊 Selection Statistics:")
    print(f"   Input bets:                 {diagnostics['input_count']}")
    print(f"   Sensitivity filtered:       -{diagnostics['sensitivity_filtered']}")  # noqa: E501
    print(
        f"   Game concentration filtered: -{diagnostics['game_concentration_filtered']}"
    )  # noqa: E501
    print(f"   Final selected:             {diagnostics['final_selected']}")

    print("\n📈 Portfolio Quality:")
    print(f"   Average EV:                 {diagnostics['avg_ev']:.2%}")
    print(f"   Games represented:          {diagnostics['games_represented']}")
    print(f"   Max bets per game:          {diagnostics['max_bets_per_game']}")

    print("\n📋 Direction Split:")
    print(f"   OVER:  {diagnostics['over_under_split']['OVER']}")
    print(f"   UNDER: {diagnostics['over_under_split']['UNDER']}")

    # Show selected bets
    print("\n🎯 Selected Bets:")
    print("-" * 80)
    for idx, bet in selected.iterrows():
        print(
            f"{bet['player_name']:12s} | {bet['game_id']:12s} | "
            f"{bet['direction']:5s} {bet['line']:4.1f} | "
            f"EV: {bet['ev']:5.2%} | CLV: {bet['clv_expectancy']:5.2f}"
        )

    # Show game distribution
    print("\n📊 Selected Bets by Game:")
    selected_game_counts = selected.groupby("game_id").size().sort_values(ascending=False)
    for game, count in selected_game_counts.items():
        print(f"   {game}: {count} bets")

    print("\n" + "=" * 80)
    print("✅ PORTFOLIO SELECTION TEST COMPLETE")
    print("=" * 80)
