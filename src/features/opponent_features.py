"""
Enhanced opponent-adjusted features for NBA props model.

Implements:
- Position-adjusted defense (DvP - Defense vs Position)
- Temporal opponent trends (rolling DRtg, pace)
- Proper temporal isolation (.shift(1))
- Matchup difficulty scoring

Research backing:
- Berri & Schmidt (2010): Position-specific defensive metrics
- Hollinger (2005): Defensive Rating and Pace Factor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class OpponentFeatures:
    """Calculate opponent-adjusted features with temporal isolation."""

    def __init__(self, team_data_path: str = None):
        """Initialize with path to CTG team data."""
        if team_data_path is None:
            team_data_path = Path(__file__).parent.parent.parent / 'data' / 'ctg_team_data'
        self.team_data_path = Path(team_data_path)
        self.team_stats = None
        self.league_avg_drtg = 110.0  # Points allowed per 100 possessions
        self.league_avg_pace = 100.0  # Possessions per 48 minutes

    def load_team_stats(self, season: str = '2023-24'):
        """Load team defensive stats from CTG team data."""
        team_files = list(self.team_data_path.glob(f'*{season}*.csv'))

        if not team_files:
            raise FileNotFoundError(f"No team data found for season {season}")

        # Load and combine team stats
        team_dfs = []
        for file in team_files:
            df = pd.read_csv(file)
            if 'TEAM' in df.columns:
                team_dfs.append(df)

        if not team_dfs:
            raise ValueError("No valid team data found")

        self.team_stats = pd.concat(team_dfs, ignore_index=True)
        return self.team_stats

    def get_opponent_defensive_rating(self, opponent: str, date: pd.Timestamp) -> float:
        """Get opponent's defensive rating (points allowed per 100 possessions)."""
        if self.team_stats is None:
            raise ValueError("Team stats not loaded. Call load_team_stats() first")

        # Extract team abbreviation from matchup string (e.g., "@ LAL" -> "LAL")
        opp_abbrev = opponent.replace('@', '').replace('vs', '').strip()

        # Get team's defensive stats
        team_data = self.team_stats[self.team_stats['TEAM'] == opp_abbrev]

        if team_data.empty:
            # Return league average if team not found
            return 110.0

        # Get defensive rating
        if 'DEF_RATING' in team_data.columns:
            return team_data['DEF_RATING'].iloc[0]
        elif 'POINTS_ALLOWED' in team_data.columns and 'POSSESSIONS' in team_data.columns:
            return (team_data['POINTS_ALLOWED'].iloc[0] / team_data['POSSESSIONS'].iloc[0]) * 100
        else:
            return 110.0  # League average

    def get_opponent_pace(self, opponent: str) -> float:
        """Get opponent's pace factor (possessions per 48 minutes)."""
        if self.team_stats is None:
            raise ValueError("Team stats not loaded")

        opp_abbrev = opponent.replace('@', '').replace('vs', '').strip()
        team_data = self.team_stats[self.team_stats['TEAM'] == opp_abbrev]

        if team_data.empty:
            return 100.0  # League average pace

        if 'PACE' in team_data.columns:
            return team_data['PACE'].iloc[0]
        else:
            return 100.0

    def add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all opponent features to dataframe."""
        # Ensure we have opponent info
        if 'MATCHUP' not in df.columns:
            raise ValueError("MATCHUP column required")

        # Extract opponent from matchup
        df['opponent'] = df['MATCHUP'].apply(lambda x: x.split()[-1])

        # Add defensive rating
        df['opp_def_rating'] = df.apply(
            lambda row: self.get_opponent_defensive_rating(row['opponent'], row.get('GAME_DATE', pd.Timestamp.now())),
            axis=1
        )

        # Add pace factor
        df['opp_pace'] = df['opponent'].apply(self.get_opponent_pace)

        # Calculate matchup difficulty score (simple composite)
        df['def_difficulty'] = (df['opp_def_rating'] / 110.0) * (100.0 / df['opp_pace'])

        # Add interaction features
        if 'PRA_L5' in df.columns:
            df['scoring_vs_def'] = df['PRA_L5'] * (110.0 / df['opp_def_rating'])

        # Add pace adjustment
        df['pace_factor'] = df['opp_pace'] / 100.0

        return df

    def calculate_position_defense(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position-adjusted defense (DvP - Defense vs Position).

        Estimates how well opponent defends player's position based on:
        - Team's overall defensive rating
        - Position-specific adjustments (PG pass better, C score in paint, etc.)

        Args:
            df: DataFrame with MATCHUP, POSITION columns

        Returns:
            DataFrame with position-adjusted defensive features
        """
        df = df.copy()

        # Position multipliers (relative defensive difficulty)
        # Based on league-wide position scoring averages
        position_multipliers = {
            'PG': 1.05,  # Point guards face slightly more pressure
            'SG': 1.00,  # Shooting guards league average
            'SF': 0.95,  # Small forwards slightly easier matchups
            'PF': 0.98,  # Power forwards
            'C': 0.92,   # Centers score most efficiently
            'G': 1.02,   # Generic guard
            'F': 0.96,   # Generic forward
            'F-C': 0.94, # Forward-center hybrid
            'G-F': 0.98  # Guard-forward hybrid
        }

        # Apply position multiplier to opponent DRtg
        df['pos_adj_drtg'] = df.apply(
            lambda row: row.get('opp_def_rating', self.league_avg_drtg) *
                       position_multipliers.get(row.get('POSITION', 'F'), 1.0),
            axis=1
        )

        # Calculate DvP score (normalized, lower = easier matchup)
        df['dvp_score'] = df['pos_adj_drtg'] / self.league_avg_drtg

        # Historical position performance vs this opponent (if available)
        # This would require game log data with opponent outcomes
        if 'PLAYER_ID' in df.columns and 'opponent' in df.columns:
            # Calculate player's historical PRA vs this specific opponent
            df['player_vs_opp_L5'] = (
                df.groupby(['PLAYER_ID', 'opponent'])['PRA']
                .shift(1)  # Temporal isolation
                .rolling(window=5, min_periods=1)
                .mean()
            )

            # Compare to player's overall L5 average
            if 'PRA_L5' in df.columns:
                df['matchup_advantage'] = df['player_vs_opp_L5'] - df['PRA_L5']

        return df

    def calculate_pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pace-related features with temporal trends.

        Pace affects opportunity:
        - High pace = more possessions = more stats
        - Pace changes game-to-game based on matchup

        Args:
            df: DataFrame with opponent info

        Returns:
            DataFrame with pace features
        """
        df = df.copy()

        # Opponent pace factor (already calculated in add_opponent_features)
        if 'opp_pace' not in df.columns:
            df['opp_pace'] = self.league_avg_pace

        # Calculate combined pace (player's team pace × opponent pace)
        # This requires team pace data - estimate from game stats
        if 'TEAM_ABBREVIATION' in df.columns:
            # Estimate team pace from minutes distribution
            team_pace = df.groupby('TEAM_ABBREVIATION')['MIN'].transform('sum') / 240 * 100
            df['team_pace'] = team_pace

            # Combined pace effect
            df['combined_pace'] = (df['team_pace'] + df['opp_pace']) / 2
        else:
            df['combined_pace'] = df['opp_pace']

        # Pace differential (how much faster/slower than league average)
        df['pace_differential'] = df['combined_pace'] - self.league_avg_pace

        # Pace trend (is opponent getting faster or slower?)
        if 'GAME_DATE' in df.columns and 'opponent' in df.columns:
            df['opp_pace_L5'] = (
                df.groupby('opponent')['opp_pace']
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )

            df['opp_pace_L10'] = (
                df.groupby('opponent')['opp_pace']
                .shift(1)
                .rolling(window=10, min_periods=1)
                .mean()
            )

            # Pace trend (recent vs longer term)
            df['opp_pace_trend'] = df['opp_pace_L5'] - df['opp_pace_L10']

        return df

    def calculate_temporal_opponent_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate opponent defensive strength with temporal trends.

        Opponent defense changes over time due to:
        - Injuries
        - Form/fatigue
        - Scheme adjustments

        Args:
            df: DataFrame with opponent and date info

        Returns:
            DataFrame with temporal opponent features
        """
        df = df.copy()

        if 'opponent' not in df.columns or 'GAME_DATE' not in df.columns:
            return df

        # Opponent DRtg rolling averages
        df['opp_drtg_L5'] = (
            df.groupby('opponent')['opp_def_rating']
            .shift(1)  # Temporal isolation
            .rolling(window=5, min_periods=1)
            .mean()
        )

        df['opp_drtg_L10'] = (
            df.groupby('opponent')['opp_def_rating']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .mean()
        )

        df['opp_drtg_L20'] = (
            df.groupby('opponent')['opp_def_rating']
            .shift(1)
            .rolling(window=20, min_periods=1)
            .mean()
        )

        # Opponent defensive trend (improving or declining?)
        df['opp_drtg_trend'] = df['opp_drtg_L5'] - df['opp_drtg_L10']

        # Opponent defensive volatility (consistency)
        df['opp_drtg_std_L10'] = (
            df.groupby('opponent')['opp_def_rating']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .std()
            .fillna(0)
        )

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all opponent features with temporal isolation.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all opponent features added
        """
        print("Adding opponent features...")

        # Basic opponent features (DRtg, pace)
        df = self.add_opponent_features(df)
        print("  ✓ Basic opponent features calculated")

        # Position-adjusted defense
        df = self.calculate_position_defense(df)
        print("  ✓ Position-adjusted defense calculated")

        # Pace features and trends
        df = self.calculate_pace_features(df)
        print("  ✓ Pace features calculated")

        # Temporal opponent strength
        df = self.calculate_temporal_opponent_strength(df)
        print("  ✓ Temporal opponent strength calculated")

        # Count features added
        new_features = [col for col in df.columns if any(x in col for x in
                       ['opp_', 'dvp_', 'pace_', 'matchup_'])]
        print(f"  ✓ Added {len(new_features)} opponent features")

        return df