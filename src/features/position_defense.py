"""
Position-Specific Defensive Features

This module creates position-specific opponent defensive features to replace
generic team-level features that had 0% importance in Day 4.

Key Innovation:
- Instead of "How good is opponent team at defense?" (too generic)
- Ask "How good is opponent at defending THIS position?" (matchup-specific)

Example:
- Stephen Curry (Point) vs Celtics with elite PG defender (Jrue Holiday)
- vs Celtics without Jrue Holiday
- Position-specific features capture this difference

Author: NBA Props Model - Phase 2 Week 1
Date: October 14, 2025
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionDefenseFeatureBuilder:
    """
    Builds position-specific opponent defensive features.

    Positions (from CTG data):
    - Point: Traditional point guards
    - Combo: Combo guards (PG/SG tweeners)
    - Wing: Shooting guards and small forwards
    - Forward: Power forwards
    - Big: Centers

    Features created:
    - opp_DRtg_vs_{Position}: Defensive rating vs this position
    - opp_PRA_allowed_vs_{Position}: Average PRA allowed to this position
    - opp_FG_pct_allowed_vs_{Position}: Field goal % allowed to this position
    """

    def __init__(self, ctg_data_path: str = "data/ctg_data_organized/players"):
        """
        Initialize position defense feature builder.

        Args:
            ctg_data_path: Path to CTG player data
        """
        self.ctg_data_path = Path(ctg_data_path)
        self.position_map = {}  # player_name → position
        self.defense_cache = {}  # (team, date, position) → defensive metrics

        self.positions = ["Point", "Combo", "Wing", "Forward", "Big"]

    def load_position_mappings(self, seasons: list = None) -> Dict[str, str]:
        """
        Load player position mappings from CTG data.

        Args:
            seasons: List of seasons to load (e.g., ['2023-24', '2024-25'])
                    If None, loads all available seasons

        Returns:
            Dictionary mapping player_name → position
        """
        logger.info("Loading player position mappings...")

        if seasons is None:
            # Get all available seasons
            seasons = [d.name for d in self.ctg_data_path.iterdir() if d.is_dir()]

        position_data = []

        for season in seasons:
            offensive_file = (
                self.ctg_data_path
                / season
                / "regular_season"
                / "offensive_overview"
                / "offensive_overview.csv"
            )

            if not offensive_file.exists():
                logger.warning(f"   Missing offensive overview for {season}")
                continue

            try:
                df = pd.read_csv(offensive_file)
                df["SEASON"] = season

                # Extract player and position
                if "Player" in df.columns and "Pos" in df.columns:
                    season_positions = df[["Player", "Pos", "SEASON"]].copy()
                    season_positions = season_positions.dropna(subset=["Player", "Pos"])
                    position_data.append(season_positions)

            except Exception as e:
                logger.warning(f"   Error loading {season}: {e}")
                continue

        if not position_data:
            raise ValueError("No position data loaded!")

        # Combine all seasons
        all_positions = pd.concat(position_data, ignore_index=True)

        # For players appearing in multiple seasons, use most recent position
        all_positions = all_positions.sort_values("SEASON")
        all_positions = all_positions.drop_duplicates(subset=["Player"], keep="last")

        # Create mapping
        self.position_map = dict(zip(all_positions["Player"], all_positions["Pos"]))

        logger.info(f"   Loaded {len(self.position_map):,} player positions")
        logger.info(f"   Position distribution:")

        position_counts = pd.Series(list(self.position_map.values())).value_counts()
        for pos, count in position_counts.items():
            logger.info(f"      {pos}: {count}")

        return self.position_map

    def get_player_position(self, player_name: str) -> Optional[str]:
        """
        Get position for a player.

        Args:
            player_name: Player name

        Returns:
            Position string or None if unknown
        """
        return self.position_map.get(player_name)

    def calculate_opponent_defense_by_position(
        self,
        games_df: pd.DataFrame,
        opponent_team: str,
        current_date: pd.Timestamp,
        window: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate how opponent team defends each position.

        Uses last N games before current_date to calculate:
        - Average PRA allowed to each position
        - Average FG% allowed to each position
        - Defensive rating vs each position (approximation)

        Args:
            games_df: All game logs with positions
            opponent_team: Team to calculate defense for
            current_date: Current date (only use games before this)
            window: Number of recent games to use (default 10)

        Returns:
            Dictionary: {position: {metric: value}}
        """
        # Check cache first
        cache_key = (opponent_team, current_date.strftime("%Y-%m-%d"))
        if cache_key in self.defense_cache:
            return self.defense_cache[cache_key]

        # Get games where opponent_team played (before current_date)
        past_games = (
            games_df[
                (games_df["GAME_DATE"] < current_date) & (games_df["OPP_TEAM"] == opponent_team)
            ]
            .sort_values("GAME_DATE", ascending=False)
            .head(window * 10)
        )  # Get more games to have enough per position

        if len(past_games) == 0:
            # Return league averages if no data
            return self._get_league_average_defense()

        # Calculate defense by position
        defense_by_position = {}

        for position in self.positions:
            # Get games against this position
            position_games = past_games[past_games["POSITION"] == position]

            if len(position_games) < 3:  # Need minimum 3 games
                # Use overall stats if insufficient data for this position
                defense_by_position[position] = {
                    "PRA_allowed": past_games["PRA"].mean() if len(past_games) > 0 else 30.0,
                    "FG_pct_allowed": past_games["FG_PCT"].mean() if len(past_games) > 0 else 0.45,
                    "n_games": 0,
                }
            else:
                # Use last N games for this position
                position_games = position_games.head(window)

                defense_by_position[position] = {
                    "PRA_allowed": position_games["PRA"].mean(),
                    "FG_pct_allowed": position_games["FG_PCT"].mean(),
                    "PTS_allowed": position_games["PTS"].mean(),
                    "REB_allowed": position_games["REB"].mean(),
                    "AST_allowed": position_games["AST"].mean(),
                    "n_games": len(position_games),
                }

        # Cache result
        self.defense_cache[cache_key] = defense_by_position

        return defense_by_position

    def _get_league_average_defense(self) -> Dict[str, Dict[str, float]]:
        """Return league average defensive metrics as fallback."""
        league_avg = {
            "PRA_allowed": 30.0,
            "FG_pct_allowed": 0.45,
            "PTS_allowed": 20.0,
            "REB_allowed": 6.0,
            "AST_allowed": 4.0,
            "n_games": 0,
        }

        return {position: league_avg.copy() for position in self.positions}

    def get_position_specific_features(
        self,
        player_name: str,
        opponent_team: str,
        games_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Get position-specific features for a player.

        Args:
            player_name: Player name
            opponent_team: Opponent team
            games_df: All game logs with positions
            current_date: Current game date

        Returns:
            Dictionary of position-specific features
        """
        features = {}

        # Get player position
        position = self.get_player_position(player_name)

        if position is None:
            # Return empty features if position unknown
            logger.debug(f"   Unknown position for {player_name}, using defaults")
            return self._get_default_features()

        # Get opponent defense vs this position
        defense_by_position = self.calculate_opponent_defense_by_position(
            games_df, opponent_team, current_date
        )

        position_defense = defense_by_position.get(position, {})

        # Create features
        features[f"opp_PRA_allowed_vs_{position}"] = position_defense.get("PRA_allowed", 30.0)
        features[f"opp_FG_pct_allowed_vs_{position}"] = position_defense.get("FG_pct_allowed", 0.45)
        features[f"opp_PTS_allowed_vs_{position}"] = position_defense.get("PTS_allowed", 20.0)
        features[f"opp_REB_allowed_vs_{position}"] = position_defense.get("REB_allowed", 6.0)
        features[f"opp_AST_allowed_vs_{position}"] = position_defense.get("AST_allowed", 4.0)

        # Add position as feature (one-hot encoded)
        for pos in self.positions:
            features[f"is_{pos}"] = 1 if pos == position else 0

        # Add sample size indicator
        features["opp_defense_n_games"] = position_defense.get("n_games", 0)

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when position is unknown."""
        features = {
            "opp_PRA_allowed_vs_Unknown": 30.0,
            "opp_FG_pct_allowed_vs_Unknown": 0.45,
            "opp_PTS_allowed_vs_Unknown": 20.0,
            "opp_REB_allowed_vs_Unknown": 6.0,
            "opp_AST_allowed_vs_Unknown": 4.0,
            "opp_defense_n_games": 0,
        }

        # All position flags are 0
        for pos in self.positions:
            features[f"is_{pos}"] = 0

        return features

    def add_positions_to_game_logs(
        self,
        game_logs_df: pd.DataFrame,
        output_path: str = "data/processed/game_logs_with_positions.csv",
    ) -> pd.DataFrame:
        """
        Add position column to game logs.

        Args:
            game_logs_df: Game logs DataFrame
            output_path: Path to save enriched game logs

        Returns:
            Game logs with POSITION column added
        """
        logger.info("Adding positions to game logs...")

        # Add position column
        game_logs_df["POSITION"] = game_logs_df["PLAYER_NAME"].map(self.position_map)

        # Check coverage
        total_games = len(game_logs_df)
        games_with_position = game_logs_df["POSITION"].notna().sum()
        coverage_pct = games_with_position / total_games * 100

        logger.info(
            f"   Position coverage: {games_with_position:,}/{total_games:,} ({coverage_pct:.1f}%)"
        )

        # Show distribution
        logger.info(f"   Position distribution:")
        position_counts = game_logs_df["POSITION"].value_counts()
        for pos, count in position_counts.items():
            pct = count / games_with_position * 100
            logger.info(f"      {pos}: {count:,} ({pct:.1f}%)")

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        game_logs_df.to_csv(output_path, index=False)

        logger.info(f"   ✅ Saved to {output_path}")

        return game_logs_df
