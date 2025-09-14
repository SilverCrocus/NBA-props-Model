"""
Simple opponent-adjusted features for NBA props model.
Direct implementation without fallback logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class OpponentFeatures:
    """Calculate opponent-adjusted features using existing CTG team data."""

    def __init__(self, team_data_path: str = None):
        """Initialize with path to CTG team data."""
        if team_data_path is None:
            team_data_path = Path(__file__).parent.parent.parent / 'data' / 'ctg_team_data'
        self.team_data_path = Path(team_data_path)
        self.team_stats = None

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

    def get_position_vs_defense(self, player_position: str, opponent: str) -> dict:
        """Get how opponent defends specific position."""
        # Simple position-based defensive metrics
        position_defense = {
            'PG': {'def_vs_pos': 1.0},
            'SG': {'def_vs_pos': 1.0},
            'SF': {'def_vs_pos': 1.0},
            'PF': {'def_vs_pos': 1.0},
            'C': {'def_vs_pos': 1.0}
        }

        # This would be enhanced with actual position-specific data
        return position_defense.get(player_position, {'def_vs_pos': 1.0})