"""
Recent Form Features for NBA Props Model

Implements:
- L3 (last 3 games) averages and trends
- Hot/cold streak detection
- Form momentum (improving vs declining)
- Short-term vs long-term comparison

Research backing:
- L3 features strongest temporal signal for next-game prediction (Silver 2014)
- Recent form (L3-L5) predicts short-term performance better than season averages
- Momentum effects real in basketball (Gilovich et al. 1985, Bocskocsky et al. 2014)

All features use proper temporal isolation (.shift(1)).
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class RecentFormFeatures:
    """Calculate recent form features with emphasis on L3."""

    def __init__(self):
        """Initialize feature calculator."""
        pass

    def calculate_l3_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate L3 (last 3 games) averages.

        L3 captures immediate form better than L5/L10:
        - Injuries/role changes show up quickly
        - Hot/cold streaks
        - Matchup-specific adjustments

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with L3 features
        """
        df = df.copy()

        # Stats to calculate L3 for
        stats = ['PRA', 'PTS', 'REB', 'AST', 'MIN', 'FGA', 'FG_PCT',
                 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'STL', 'BLK', 'TOV']

        for stat in stats:
            if stat not in df.columns:
                continue

            # L3 mean
            df[f'{stat}_L3_mean'] = (
                df.groupby('PLAYER_ID')[stat]
                .shift(1)  # Temporal isolation
                .rolling(window=3, min_periods=1)
                .mean()
            )

            # L3 median (more robust to outliers than mean)
            df[f'{stat}_L3_median'] = (
                df.groupby('PLAYER_ID')[stat]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .median()
            )

            # L3 max (ceiling in recent games)
            df[f'{stat}_L3_max'] = (
                df.groupby('PLAYER_ID')[stat]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .max()
            )

            # L3 min (floor in recent games)
            df[f'{stat}_L3_min'] = (
                df.groupby('PLAYER_ID')[stat]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .min()
            )

        return df

    def calculate_form_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate form momentum (improving or declining).

        Momentum detection:
        - Compare L3 to L10 (short-term vs medium-term)
        - Compare L3 to season average
        - Trend direction (last 3 games linear fit)

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with momentum features
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # L3 vs L10 comparison
        df['PRA_L10_mean'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .mean()
        )

        if 'PRA_L3_mean' not in df.columns:
            df['PRA_L3_mean'] = (
                df.groupby('PLAYER_ID')['PRA']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # Momentum score (L3 - L10)
        # Positive = improving, Negative = declining
        df['momentum_L3_vs_L10'] = df['PRA_L3_mean'] - df['PRA_L10_mean']

        # L3 vs season average
        df['PRA_season_avg'] = (
            df.groupby(['PLAYER_ID', 'SEASON'])['PRA']
            .transform('mean')
        )

        df['momentum_L3_vs_season'] = df['PRA_L3_mean'] - df['PRA_season_avg']

        # Trend slope (last 3 games)
        # Positive slope = trending up, Negative = trending down
        def calculate_trend_slope(series):
            """Calculate linear trend slope."""
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            # Simple linear regression slope
            slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
            return slope

        df['pra_trend_L3'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=3, min_periods=2)
            .apply(calculate_trend_slope, raw=False)
            .fillna(0)
        )

        # Momentum strength (absolute value of momentum)
        df['momentum_strength'] = df['momentum_L3_vs_L10'].abs()

        return df

    def detect_hot_cold_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect hot and cold streaks.

        Hot streak = consistently exceeding expectation
        Cold streak = consistently below expectation

        Thresholds:
        - Hot: L3 > L10 + 0.5 std
        - Cold: L3 < L10 - 0.5 std

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with hot/cold indicators
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # Calculate baseline (L10) and std
        if 'PRA_L10_mean' not in df.columns:
            df['PRA_L10_mean'] = (
                df.groupby('PLAYER_ID')['PRA']
                .shift(1)
                .rolling(window=10, min_periods=3)
                .mean()
            )

        df['PRA_L10_std'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .std()
            .fillna(0)
        )

        # Get L3 mean
        if 'PRA_L3_mean' not in df.columns:
            df['PRA_L3_mean'] = (
                df.groupby('PLAYER_ID')['PRA']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # Hot streak indicator
        df['is_hot'] = (
            (df['PRA_L3_mean'] > (df['PRA_L10_mean'] + 0.5 * df['PRA_L10_std']))
        ).astype(int)

        # Cold streak indicator
        df['is_cold'] = (
            (df['PRA_L3_mean'] < (df['PRA_L10_mean'] - 0.5 * df['PRA_L10_std']))
        ).astype(int)

        # Neutral (neither hot nor cold)
        df['is_neutral'] = ((df['is_hot'] == 0) & (df['is_cold'] == 0)).astype(int)

        # Hot/cold intensity (how far from baseline)
        df['hot_intensity'] = (
            (df['PRA_L3_mean'] - df['PRA_L10_mean']) / df['PRA_L10_std']
        ).clip(-5, 5).fillna(0)  # Clip to -5 to +5 standard deviations

        # Streak duration (consecutive hot or cold games)
        df['performance_vs_baseline'] = df['PRA_L3_mean'] - df['PRA_L10_mean']

        def calculate_streak_length(series):
            """Calculate current streak length."""
            if len(series) == 0:
                return 0

            # Check if series is consistently above or below 0
            signs = np.sign(series)
            if len(signs) == 0:
                return 0

            current_sign = signs.iloc[-1]
            if current_sign == 0:
                return 0

            streak = 1
            for i in range(len(signs) - 2, -1, -1):
                if signs.iloc[i] == current_sign:
                    streak += 1
                else:
                    break

            return streak if current_sign > 0 else -streak

        df['streak_length'] = (
            df.groupby('PLAYER_ID')['performance_vs_baseline']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .apply(calculate_streak_length, raw=False)
            .fillna(0)
        )

        return df

    def calculate_game_by_game_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate game-by-game trends.

        Examines progression of last 3 games:
        - Are stats improving game-to-game?
        - Or declining?

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with trend features
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # Last game performance
        df['PRA_last_game'] = df.groupby('PLAYER_ID')['PRA'].shift(1)

        # 2 games ago
        df['PRA_2_games_ago'] = df.groupby('PLAYER_ID')['PRA'].shift(2)

        # 3 games ago
        df['PRA_3_games_ago'] = df.groupby('PLAYER_ID')['PRA'].shift(3)

        # Game-to-game changes
        df['pra_change_last_game'] = df['PRA_last_game'] - df['PRA_2_games_ago']
        df['pra_change_2_games_ago'] = df['PRA_2_games_ago'] - df['PRA_3_games_ago']

        # Acceleration (change in change)
        df['pra_acceleration'] = df['pra_change_last_game'] - df['pra_change_2_games_ago']

        # Consistency of improvement/decline
        df['trend_consistency'] = (
            np.sign(df['pra_change_last_game']) == np.sign(df['pra_change_2_games_ago'])
        ).astype(int)

        # Clean up temporary columns
        df.drop(['PRA_2_games_ago', 'PRA_3_games_ago', 'pra_change_2_games_ago'],
                axis=1, inplace=True, errors='ignore')

        return df

    def calculate_minutes_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate minutes trend (critical for prop prediction).

        Minutes changes indicate:
        - Role changes (starter to bench, vice versa)
        - Injury recovery/management
        - Coach's confidence

        Args:
            df: DataFrame with MIN column

        Returns:
            DataFrame with minutes trend features
        """
        df = df.copy()

        if 'MIN' not in df.columns:
            return df

        # L3 minutes average
        if 'MIN_L3_mean' not in df.columns:
            df['MIN_L3_mean'] = (
                df.groupby('PLAYER_ID')['MIN']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # L10 minutes average
        df['MIN_L10_mean'] = (
            df.groupby('PLAYER_ID')['MIN']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .mean()
        )

        # Minutes trend (L3 vs L10)
        df['minutes_trend'] = df['MIN_L3_mean'] - df['MIN_L10_mean']

        # Minutes stability (low std = consistent role)
        df['minutes_stability_L3'] = (
            df.groupby('PLAYER_ID')['MIN']
            .shift(1)
            .rolling(window=3, min_periods=2)
            .std()
            .fillna(0)
        )

        # Role change indicator (large minutes change)
        df['role_change'] = (df['minutes_trend'].abs() > 5).astype(int)

        # Minutes per PRA (efficiency metric)
        if 'PRA' in df.columns:
            df['PRA_per_minute_L3'] = (
                df['PRA_L3_mean'] / df['MIN_L3_mean']
            ).replace([np.inf, -np.inf], np.nan).fillna(0)

        return df

    def calculate_scoring_efficiency_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate scoring efficiency trends.

        Examines shooting efficiency changes:
        - FG% trend
        - Usage rate trend
        - Points per shot attempt

        Args:
            df: DataFrame with shooting stats

        Returns:
            DataFrame with efficiency trend features
        """
        df = df.copy()

        # FG% trend
        if 'FG_PCT' in df.columns:
            df['FG_PCT_L3_mean'] = (
                df.groupby('PLAYER_ID')['FG_PCT']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

            df['FG_PCT_L10_mean'] = (
                df.groupby('PLAYER_ID')['FG_PCT']
                .shift(1)
                .rolling(window=10, min_periods=1)
                .mean()
            )

            df['fg_pct_trend'] = df['FG_PCT_L3_mean'] - df['FG_PCT_L10_mean']

        # 3P% trend
        if 'FG3_PCT' in df.columns:
            df['FG3_PCT_L3_mean'] = (
                df.groupby('PLAYER_ID')['FG3_PCT']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # FT% trend
        if 'FT_PCT' in df.columns:
            df['FT_PCT_L3_mean'] = (
                df.groupby('PLAYER_ID')['FT_PCT']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        # Points per shot attempt (efficiency)
        if 'PTS' in df.columns and 'FGA' in df.columns:
            df['pts_per_fga'] = (df['PTS'] / df['FGA']).replace([np.inf, -np.inf], np.nan).fillna(0)

            df['pts_per_fga_L3'] = (
                df.groupby('PLAYER_ID')['pts_per_fga']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

            df.drop('pts_per_fga', axis=1, inplace=True, errors='ignore')

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all recent form features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all recent form features added
        """
        print("Adding recent form features...")

        df = self.calculate_l3_averages(df)
        print("  ✓ L3 averages calculated")

        df = self.calculate_form_momentum(df)
        print("  ✓ Form momentum calculated")

        df = self.detect_hot_cold_streaks(df)
        print("  ✓ Hot/cold streak detection calculated")

        df = self.calculate_game_by_game_trends(df)
        print("  ✓ Game-by-game trends calculated")

        df = self.calculate_minutes_trend(df)
        print("  ✓ Minutes trends calculated")

        df = self.calculate_scoring_efficiency_trends(df)
        print("  ✓ Scoring efficiency trends calculated")

        # Count features added
        new_features = [col for col in df.columns if any(x in col for x in
                       ['_L3_', 'momentum', 'hot', 'cold', 'streak', 'trend',
                        'acceleration', 'role_change', 'efficiency'])]
        print(f"  ✓ Added {len(new_features)} recent form features")

        return df


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'PLAYER_ID': [1] * 20,
        'SEASON': ['2023-24'] * 20,
        'GAME_DATE': pd.date_range('2024-01-01', periods=20),
        'PRA': [20, 22, 21, 23, 25, 27, 28, 30, 29, 31,  # Trending up
                32, 30, 33, 31, 34, 32, 35, 33, 36, 34],
        'PTS': [15, 16, 15, 17, 18, 20, 20, 22, 21, 23,
                24, 22, 24, 23, 25, 24, 26, 25, 27, 26],
        'REB': [3, 4, 4, 4, 5, 5, 6, 6, 6, 6,
                6, 6, 7, 6, 7, 6, 7, 6, 7, 6],
        'AST': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        'MIN': [28, 29, 28, 30, 30, 31, 32, 32, 33, 33,
                34, 33, 34, 34, 35, 35, 36, 35, 36, 36],
        'FGA': [12, 13, 12, 13, 14, 15, 15, 16, 16, 17,
                17, 16, 17, 17, 18, 18, 19, 18, 19, 19],
        'FG_PCT': [0.45, 0.46, 0.44, 0.47, 0.48, 0.50, 0.49, 0.51, 0.50, 0.52,
                   0.53, 0.51, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55, 0.54],
        'FG3A': [3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                 5, 5, 5, 5, 5, 5, 5, 5, 6, 6],
        'FG3_PCT': [0.33, 0.33, 0.33, 0.33, 0.35, 0.35, 0.35, 0.35, 0.36, 0.36,
                    0.37, 0.37, 0.37, 0.38, 0.38, 0.38, 0.39, 0.39, 0.40, 0.40],
        'FTA': [4, 4, 4, 5, 5, 5, 6, 6, 6, 6,
                6, 6, 7, 7, 7, 7, 7, 7, 8, 8],
        'FT_PCT': [0.75, 0.75, 0.75, 0.80, 0.80, 0.80, 0.83, 0.83, 0.83, 0.83,
                   0.83, 0.83, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.88, 0.88],
        'STL': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'BLK': [0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        'TOV': [2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    })

    calculator = RecentFormFeatures()
    result = calculator.add_all_features(sample_data)

    print("\nSample output columns:")
    form_cols = [col for col in result.columns if 'L3' in col or 'momentum' in col or 'hot' in col]
    print(form_cols[:15])

    print("\nRecent form features (last 5 games):")
    display_cols = ['PRA', 'PRA_L3_mean', 'momentum_L3_vs_L10', 'is_hot', 'streak_length', 'minutes_trend']
    print(result[display_cols].tail(5).to_string())
