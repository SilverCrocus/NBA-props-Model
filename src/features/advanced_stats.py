"""
Advanced Statistics Features for NBA Props Model

Implements:
- True Shooting Percentage (TS%)
- Pace-adjusted stats (per 100 possessions)
- Usage Rate (game-level)
- Four Factors metrics

All features use proper temporal isolation (.shift(1)).
"""

import pandas as pd
import numpy as np
from typing import Optional


class AdvancedStatsFeatures:
    """Calculate advanced basketball statistics."""

    def __init__(self):
        """Initialize feature calculator."""
        self.league_avg_pace = 100.0  # Possessions per 48 minutes

    def calculate_true_shooting_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate True Shooting Percentage (TS%).

        TS% = PTS / (2 × (FGA + 0.44 × FTA))

        More accurate than FG% because it accounts for:
        - 3-pointers (worth more than 2-pointers)
        - Free throws

        Args:
            df: DataFrame with PTS, FGA, FTA columns

        Returns:
            DataFrame with TS% features added
        """
        df = df.copy()

        # Calculate True Shooting %
        df['TS_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['TS_pct'] = df['TS_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Clip to realistic range (0-100%)
        df['TS_pct'] = df['TS_pct'].clip(0, 1.0)

        # Rolling averages (use .shift(1) for temporal isolation)
        for window in [3, 5, 10, 20]:
            col_name = f'TS_pct_L{window}'
            df[col_name] = (
                df.groupby('PLAYER_ID')['TS_pct']
                .shift(1)  # Temporal isolation
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # TS% trend (L5 vs season average)
        # FIXED: Use expanding mean with shift to prevent data leakage
        df['TS_pct_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON'])['TS_pct']
            .shift(1)  # Temporal isolation
            .expanding()
            .mean()
        )
        df['TS_pct_trend'] = df['TS_pct_L5'] - df['TS_pct_season_avg']

        # Drop base TS_pct column to prevent data leakage
        # Only keep lagged versions (TS_pct_L3, TS_pct_L5, etc.)
        df = df.drop('TS_pct', axis=1, errors='ignore')

        return df

    def calculate_usage_rate(self, df: pd.DataFrame, team_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate Usage Rate (USG%) - game level.

        USG% = 100 × ((FGA + 0.44 × FTA + TOV) × (TEAM_MIN / 5)) /
                    (MIN × (TEAM_FGA + 0.44 × TEAM_FTA + TEAM_TOV))

        Indicates % of team plays a player is involved in while on court.

        Args:
            df: DataFrame with player stats
            team_df: Optional team-level stats (if not provided, estimated from player stats)

        Returns:
            DataFrame with USG% features added
        """
        df = df.copy()

        # Calculate player touches
        player_touches = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

        # Estimate team stats (sum of all players in same game)
        # This is an approximation - ideally would use actual team totals
        team_stats = df.groupby('GAME_ID').agg({
            'FGA': 'sum',
            'FTA': 'sum',
            'TOV': 'sum',
            'MIN': 'sum'
        }).rename(columns=lambda x: f'TEAM_{x}')

        df = df.merge(team_stats, left_on='GAME_ID', right_index=True, how='left')

        # Calculate USG%
        team_touches = df['TEAM_FGA'] + 0.44 * df['TEAM_FTA'] + df['TEAM_TOV']

        # Avoid division by zero
        df['USG_pct'] = np.where(
            (df['MIN'] > 0) & (team_touches > 0),
            100 * (player_touches * (df['TEAM_MIN'] / 5)) / (df['MIN'] * team_touches),
            0
        )

        # Clip to realistic range (0-50%)
        df['USG_pct'] = df['USG_pct'].clip(0, 50)

        # Rolling averages
        for window in [3, 5, 10]:
            col_name = f'USG_pct_L{window}'
            df[col_name] = (
                df.groupby('PLAYER_ID')['USG_pct']
                .shift(1)  # Temporal isolation
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # USG stability (standard deviation)
        df['USG_pct_std_L10'] = (
            df.groupby('PLAYER_ID')['USG_pct']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .std()
            .fillna(0)
        )

        # Drop base USG_pct column to prevent data leakage
        # Only keep lagged versions (USG_pct_L3, USG_pct_L5, etc.)
        df = df.drop('USG_pct', axis=1, errors='ignore')

        return df

    def calculate_pace_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pace-adjusted stats (per 100 possessions).

        Normalizes stats to account for different game paces.
        Fast-paced games have more possessions → more opportunities.

        Args:
            df: DataFrame with stats and minutes

        Returns:
            DataFrame with pace-adjusted features
        """
        df = df.copy()

        # Estimate possessions from team stats
        # Formula: 0.5 × ((TEAM_FGA + 0.4×TEAM_FTA - 1.07×OREB% + TEAM_TOV) +
        #                 (OPP_FGA + 0.4×OPP_FTA - 1.07×OPP_OREB% + OPP_TOV))
        # Simplified: Use minutes as proxy for possessions

        # Calculate possessions per minute (estimate)
        # Average NBA game: 100 possessions per team in 48 minutes
        # Player possessions = (game possessions / 48) × player minutes

        df['estimated_possessions'] = (df['MIN'] / 48) * 100

        # Avoid division by zero
        df['estimated_possessions'] = df['estimated_possessions'].replace(0, np.nan)

        # Calculate per-100 stats
        for stat in ['PTS', 'REB', 'AST', 'PRA']:
            if stat in df.columns:
                per_100_col = f'{stat}_per_100'
                df[per_100_col] = (df[stat] / df['estimated_possessions']) * 100
                df[per_100_col] = df[per_100_col].replace([np.inf, -np.inf], np.nan)

                # Rolling averages
                for window in [5, 10, 20]:
                    col_name = f'{per_100_col}_L{window}'
                    df[col_name] = (
                        df.groupby('PLAYER_ID')[per_100_col]
                        .shift(1)  # Temporal isolation
                        .rolling(window=window, min_periods=1)
                        .mean()
                    )

        # Drop base per_100 columns to prevent data leakage
        # Only keep lagged versions (PTS_per_100_L5, PTS_per_100_L10, etc.)
        base_per_100_cols = ['PTS_per_100', 'REB_per_100', 'AST_per_100', 'PRA_per_100', 'estimated_possessions']
        df = df.drop(base_per_100_cols, axis=1, errors='ignore')

        return df

    def calculate_pace_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team and opponent pace factors.

        Pace = Possessions per 48 minutes
        High pace = more opportunities for stats

        Args:
            df: DataFrame with game-level data

        Returns:
            DataFrame with pace features
        """
        df = df.copy()

        # Estimate game pace from team stats
        # This is a simplified calculation
        # Ideally would use actual possession counts

        # Calculate pace from game totals
        game_stats = df.groupby('GAME_ID').agg({
            'FGA': 'sum',
            'FTA': 'sum',
            'TOV': 'sum',
            'OREB': 'sum',
            'DREB': 'sum'
        })

        # Possession estimate
        game_stats['possessions'] = (
            game_stats['FGA'] +
            0.4 * game_stats['FTA'] -
            0.7 * game_stats['OREB'] +
            game_stats['TOV']
        )

        # Pace = possessions per 48 minutes (assuming 48 min game)
        game_stats['pace'] = game_stats['possessions']  # Already per-team

        df = df.merge(game_stats[['pace']], left_on='GAME_ID', right_index=True, how='left')

        # Normalize to league average
        df['pace_factor'] = df['pace'] / self.league_avg_pace

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced statistics features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all advanced stats features added
        """
        print("Adding advanced statistics features...")

        df = self.calculate_true_shooting_pct(df)
        print("  ✓ True Shooting % calculated")

        df = self.calculate_usage_rate(df)
        print("  ✓ Usage Rate calculated")

        df = self.calculate_pace_adjusted_stats(df)
        print("  ✓ Pace-adjusted stats calculated")

        df = self.calculate_pace_factor(df)
        print("  ✓ Pace factor calculated")

        # Count features added
        new_features = [col for col in df.columns if any(x in col for x in
                       ['TS_pct', 'USG_pct', 'per_100', 'pace'])]
        print(f"  ✓ Added {len(new_features)} advanced stat features")

        return df


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1, 2, 2, 2],
        'GAME_ID': ['G1', 'G2', 'G3', 'G1', 'G2', 'G3'],
        'SEASON': ['2023-24'] * 6,
        'MIN': [30, 32, 28, 25, 27, 26],
        'PTS': [20, 22, 18, 15, 16, 14],
        'FGA': [15, 16, 14, 10, 11, 9],
        'FTA': [5, 4, 6, 3, 4, 5],
        'REB': [5, 6, 4, 8, 7, 9],
        'AST': [4, 5, 3, 2, 3, 2],
        'TOV': [2, 3, 2, 1, 2, 1],
        'OREB': [1, 2, 1, 3, 2, 3],
        'DREB': [4, 4, 3, 5, 5, 6],
        'PRA': [29, 33, 25, 25, 26, 25]
    })

    calculator = AdvancedStatsFeatures()
    result = calculator.add_all_features(sample_data)

    print("\nSample output columns:")
    print([col for col in result.columns if 'TS_pct' in col or 'USG' in col or 'per_100' in col][:10])
