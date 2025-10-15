"""
Centralized Feature Calculator for NBA Props Model

Eliminates 75% code duplication by consolidating all feature engineering logic.
All scripts should use this FeatureCalculator instead of duplicating functions.

Author: NBA Props Model Refactoring
Date: October 15, 2025
"""

import logging
from datetime import timedelta
from typing import Dict

import pandas as pd

from config import feature_config
from src.exceptions import FeatureCalculationError, InsufficientDataError

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    Centralized feature calculator for NBA player prop predictions.
    
    Consolidates all temporal, efficiency, normalization, and contextual features.
    """

    def __init__(self):
        """Initialize feature calculator with configuration."""
        self.config = feature_config

    def calculate_lag_features(
        self, player_history: pd.DataFrame, lags: list = None
    ) -> Dict[str, float]:
        """
        Calculate lag features using ONLY historical games.
        
        Args:
            player_history: Player's games before current date (sorted by date desc)
            lags: List of lag values (default from config)
            
        Returns:
            Dictionary of lag features
        """
        if lags is None:
            lags = self.config.LAG_VALUES

        features = {}

        if len(player_history) == 0:
            for lag in lags:
                features[f"PRA_lag{lag}"] = 0
                features[f"MIN_lag{lag}"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=False)

        for lag in lags:
            if len(history) >= lag:
                features[f"PRA_lag{lag}"] = history.iloc[lag - 1]["PRA"]
                features[f"MIN_lag{lag}"] = history.iloc[lag - 1].get("MIN", 0)
            else:
                features[f"PRA_lag{lag}"] = 0
                features[f"MIN_lag{lag}"] = 0

        return features

    def calculate_rolling_features(
        self, player_history: pd.DataFrame, windows: list = None
    ) -> Dict[str, float]:
        """
        Calculate rolling average features.
        
        Args:
            player_history: Player's games before current date
            windows: List of window sizes (default from config)
            
        Returns:
            Dictionary of rolling features
        """
        if windows is None:
            windows = self.config.ROLLING_WINDOWS

        features = {}

        if len(player_history) == 0:
            for window in windows:
                features[f"PRA_L{window}_mean"] = 0
                features[f"PRA_L{window}_std"] = 0
                features[f"MIN_L{window}_mean"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=False)

        for window in windows:
            if len(history) >= window:
                recent_games = history.iloc[:window]
                features[f"PRA_L{window}_mean"] = recent_games["PRA"].mean()
                features[f"PRA_L{window}_std"] = recent_games["PRA"].std()
                features[f"MIN_L{window}_mean"] = recent_games.get("MIN", pd.Series([0])).mean()
            elif len(history) >= 3:
                features[f"PRA_L{window}_mean"] = history["PRA"].mean()
                features[f"PRA_L{window}_std"] = history["PRA"].std()
                features[f"MIN_L{window}_mean"] = history.get("MIN", pd.Series([0])).mean()
            else:
                features[f"PRA_L{window}_mean"] = 0
                features[f"PRA_L{window}_std"] = 0
                features[f"MIN_L{window}_mean"] = 0

        return features

    def calculate_ewma_features(
        self, player_history: pd.DataFrame, spans: list = None
    ) -> Dict[str, float]:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) features.
        
        Args:
            player_history: Player's games before current date
            spans: List of span values (default from config)
            
        Returns:
            Dictionary of EWMA features
        """
        if spans is None:
            spans = self.config.EWMA_SPANS

        features = {}

        if len(player_history) < 3:
            for span in spans:
                features[f"PRA_ewma{span}"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=True)

        for span in spans:
            ewma_value = history["PRA"].ewm(span=span, min_periods=1).mean().iloc[-1]
            features[f"PRA_ewma{span}"] = ewma_value

        return features

    def calculate_rest_features(
        self, player_history: pd.DataFrame, current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate rest and schedule fatigue features.
        
        Args:
            player_history: Player's games before current date
            current_date: Current game date
            
        Returns:
            Dictionary of rest features
        """
        features = {}

        if len(player_history) == 0:
            features["days_rest"] = 7
            features["is_b2b"] = 0
            features["games_last_7d"] = 0
            return features

        last_game = player_history.sort_values("GAME_DATE", ascending=False).iloc[0]
        last_game_date = last_game["GAME_DATE"]

        days_rest = (current_date - last_game_date).days
        features["days_rest"] = min(days_rest, 7)
        features["is_b2b"] = 1 if days_rest <= 1 else 0

        week_ago = current_date - timedelta(days=7)
        recent_games = player_history[player_history["GAME_DATE"] >= week_ago]
        features["games_last_7d"] = len(recent_games)

        return features

    def calculate_trend_features(self, player_history: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trend features (L5 vs L10-15).
        
        Args:
            player_history: Player's games before current date
            
        Returns:
            Dictionary of trend features
        """
        features = {}

        if len(player_history) < 10:
            features["PRA_trend"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=False)

        if len(history) >= 15:
            l5_mean = history.iloc[:5]["PRA"].mean()
            l10_mean = history.iloc[5:15]["PRA"].mean()
            features["PRA_trend"] = l5_mean - l10_mean
        else:
            features["PRA_trend"] = 0

        return features

    def calculate_efficiency_features(
        self, player_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate efficiency features: TS%, PER, usage per 36, points per shot.
        
        Args:
            player_history: Player's games before current date
            
        Returns:
            Dictionary of efficiency features
        """
        features = {}

        if len(player_history) < 5:
            features["TS_pct"] = 0
            features["PER"] = 0
            features["USG_per_36"] = 0
            features["PTS_per_shot"] = 0
            features["eFG_pct"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]

        # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
        pts_total = history["PTS"].sum()
        fga_total = history.get("FGA", pd.Series([0])).sum()
        fta_total = history.get("FTA", pd.Series([0])).sum()

        ts_denominator = 2 * (fga_total + 0.44 * fta_total)
        features["TS_pct"] = pts_total / ts_denominator if ts_denominator > 0 else 0

        # Effective FG% = (FGM + 0.5 * FG3M) / FGA
        fgm_total = history.get("FGM", pd.Series([0])).sum()
        fg3m_total = history.get("FG3M", pd.Series([0])).sum()
        features["eFG_pct"] = (fgm_total + 0.5 * fg3m_total) / fga_total if fga_total > 0 else 0

        # Points per shot attempt
        features["PTS_per_shot"] = pts_total / fga_total if fga_total > 0 else 0

        # Usage per 36 minutes
        min_total = history.get("MIN", pd.Series([0])).sum()
        tov_total = history.get("TOV", pd.Series([0])).sum()
        usage_total = fga_total + 0.44 * fta_total + tov_total
        features["USG_per_36"] = (usage_total / min_total * 36) if min_total > 0 else 0

        # Simplified PER
        reb_total = history.get("REB", pd.Series([0])).sum()
        ast_total = history.get("AST", pd.Series([0])).sum()
        stl_total = history.get("STL", pd.Series([0])).sum()
        blk_total = history.get("BLK", pd.Series([0])).sum()
        ftm_total = history.get("FTM", pd.Series([0])).sum()

        per_numerator = (
            pts_total
            + reb_total
            + ast_total
            + stl_total
            + blk_total
            - tov_total
            - (fga_total - fgm_total)
            - (fta_total - ftm_total)
        )
        features["PER"] = (per_numerator / min_total * 15) if min_total > 0 else 0

        return features

    def calculate_normalization_features(
        self, player_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate per-36 and per-100 possession stats.
        
        Args:
            player_history: Player's games before current date
            
        Returns:
            Dictionary of normalized features
        """
        features = {}

        if len(player_history) < 5:
            features["PRA_per_36"] = 0
            features["PTS_per_36"] = 0
            features["REB_per_36"] = 0
            features["AST_per_36"] = 0
            features["MIN_avg"] = 0
            return features

        history = player_history.sort_values("GAME_DATE", ascending=False).iloc[:10]

        min_total = history.get("MIN", pd.Series([0])).sum()

        if min_total == 0:
            features["PRA_per_36"] = 0
            features["PTS_per_36"] = 0
            features["REB_per_36"] = 0
            features["AST_per_36"] = 0
            features["MIN_avg"] = 0
            return features

        # Per-36 stats
        pra_total = history["PRA"].sum()
        pts_total = history["PTS"].sum()
        reb_total = history.get("REB", pd.Series([0])).sum()
        ast_total = history.get("AST", pd.Series([0])).sum()

        features["PRA_per_36"] = (pra_total / min_total) * 36
        features["PTS_per_36"] = (pts_total / min_total) * 36
        features["REB_per_36"] = (reb_total / min_total) * 36
        features["AST_per_36"] = (ast_total / min_total) * 36

        # Average minutes played
        features["MIN_avg"] = min_total / len(history)

        return features

    def calculate_opponent_features(
        self, opponent_team: str, all_games: pd.DataFrame, current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """
        Calculate opponent DEFENSIVE features.
        
        Args:
            opponent_team: Name of opponent team
            all_games: All games data (for calculating opponent stats)
            current_date: Current game date
            
        Returns:
            Dictionary of opponent features
        """
        features = {}

        # Get games BEFORE current date
        past_games = all_games[all_games["GAME_DATE"] < current_date]

        # Get games where OTHER teams played AGAINST this opponent
        opponent_defense_games = past_games[past_games["OPP_TEAM"] == opponent_team]

        if len(opponent_defense_games) < 5:
            features["opp_DRtg"] = 110.0  # League average
            features["opp_pace"] = 100.0  # League average
            features["opp_PRA_allowed"] = 30.0  # League average
            return features

        # Use last 20 games for better sample
        recent_def = opponent_defense_games.sort_values("GAME_DATE", ascending=False).iloc[:20]

        # PRA allowed = average PRA scored BY OPPONENTS against this team
        features["opp_PRA_allowed"] = recent_def["PRA"].mean()

        # Defensive Rating: Higher PRA allowed = worse defense = higher DRtg
        features["opp_DRtg"] = 95.0 + (features["opp_PRA_allowed"] - 30.0) * 0.5

        # Pace calculation
        opp_games = past_games[past_games["TEAM_NAME"] == opponent_team]
        if len(opp_games) >= 5:
            recent_opp = opp_games.sort_values("GAME_DATE", ascending=False).iloc[:10]
            features["opp_pace"] = 90.0 + (recent_opp["PRA"].mean() - 30.0) * 0.3
        else:
            features["opp_pace"] = 100.0

        return features

    def calculate_all_features(
        self,
        player_history: pd.DataFrame,
        current_date: pd.Timestamp,
        player_name: str,
        opponent_team: str,
        season: str,
        ctg_builder,
        all_games: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate ALL features for a single prediction.
        
        This is the master method that consolidates all feature types.
        
        Args:
            player_history: Player's games before current date
            current_date: Current game date
            player_name: Player name for CTG lookup
            opponent_team: Opponent team name
            season: Current season
            ctg_builder: CTG feature builder instance
            all_games: All games data (for opponent features)
            
        Returns:
            Dictionary of all features
            
        Raises:
            FeatureCalculationError: If feature calculation fails
            InsufficientDataError: If insufficient data for prediction
        """
        try:
            features = {}

            # Temporal features
            features.update(self.calculate_lag_features(player_history))
            features.update(self.calculate_rolling_features(player_history))
            features.update(self.calculate_ewma_features(player_history))
            features.update(self.calculate_rest_features(player_history, current_date))
            features.update(self.calculate_trend_features(player_history))

            # Advanced features
            features.update(self.calculate_efficiency_features(player_history))
            features.update(self.calculate_normalization_features(player_history))
            features.update(
                self.calculate_opponent_features(opponent_team, all_games, current_date)
            )

            # CTG season stats
            ctg_feats = ctg_builder.get_player_ctg_features(player_name, season)
            features.update(ctg_feats)

            return features

        except KeyError as e:
            raise FeatureCalculationError(f"Missing required column: {e}") from e
        except ValueError as e:
            raise FeatureCalculationError(f"Invalid value in feature calculation: {e}") from e
        except Exception as e:
            raise FeatureCalculationError(f"Unexpected error in feature calculation: {e}") from e
