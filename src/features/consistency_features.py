"""
Player Consistency Features for NBA Props Model

Implements:
- Coefficient of Variation (CV = std/mean)
- Volatility metrics (rolling standard deviation)
- Boom/bust tendency detection
- Consistency scoring

Research backing:
- Lower CV = more consistent = more predictable for betting
- High-usage players tend to be less consistent (Berri & Schmidt 2010)
- Volatility predicts prop bet success (Lopez & Matthews 2015)

All features use proper temporal isolation (.shift(1)).
"""

import pandas as pd
import numpy as np
from typing import Optional


class ConsistencyFeatures:
    """Calculate player consistency metrics."""

    def __init__(self):
        """Initialize feature calculator."""
        pass

    def calculate_coefficient_of_variation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Coefficient of Variation (CV).

        CV = (standard deviation / mean) × 100

        Dimensionless consistency metric:
        - CV < 20% = very consistent
        - CV 20-40% = moderate consistency
        - CV > 40% = high volatility

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with CV features added
        """
        df = df.copy()

        # Calculate CV for key stats over different windows
        for stat in ['PRA', 'PTS', 'REB', 'AST']:
            if stat not in df.columns:
                continue

            for window in [5, 10, 20]:
                # Calculate rolling mean
                mean_col = f'{stat}_mean_L{window}'
                df[mean_col] = (
                    df.groupby('PLAYER_ID')[stat]
                    .shift(1)  # Temporal isolation
                    .rolling(window=window, min_periods=2)
                    .mean()
                )

                # Calculate rolling std
                std_col = f'{stat}_std_L{window}'
                df[std_col] = (
                    df.groupby('PLAYER_ID')[stat]
                    .shift(1)
                    .rolling(window=window, min_periods=2)
                    .std()
                )

                # Calculate CV
                cv_col = f'{stat}_CV_L{window}'
                df[cv_col] = (df[std_col] / df[mean_col]) * 100
                df[cv_col] = df[cv_col].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Clip to reasonable range (0-200%)
                df[cv_col] = df[cv_col].clip(0, 200)

        # Calculate overall consistency score (inverse of CV)
        # Lower CV = higher consistency score
        df['consistency_score_L10'] = 100 - df['PRA_CV_L10'].clip(0, 100)

        return df

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility metrics.

        Volatility = standard deviation of performance.
        High volatility = unpredictable (risky for betting).

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with volatility features
        """
        df = df.copy()

        # Rolling standard deviation (already calculated in CV)
        # Add additional volatility metrics

        # Volatility trend (is player becoming more or less consistent?)
        if 'PRA_std_L5' in df.columns and 'PRA_std_L10' in df.columns:
            df['volatility_trend'] = df['PRA_std_L5'] - df['PRA_std_L10']
            # Positive = becoming more volatile
            # Negative = becoming more consistent

        # Normalized volatility (relative to player's average)
        if 'PRA_std_L10' in df.columns and 'PRA_mean_L10' in df.columns:
            df['normalized_volatility'] = df['PRA_std_L10'] / df['PRA_mean_L10']

        # Minutes volatility (inconsistent minutes = inconsistent stats)
        if 'MIN' in df.columns:
            for window in [5, 10]:
                df[f'MIN_std_L{window}'] = (
                    df.groupby('PLAYER_ID')['MIN']
                    .shift(1)
                    .rolling(window=window, min_periods=2)
                    .std()
                    .fillna(0)
                )

            # Minutes consistency affects PRA consistency
            df['minutes_consistency_score'] = 100 - (df['MIN_std_L10'] * 2).clip(0, 100)

        return df

    def detect_boom_bust_tendency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect boom/bust tendency.

        Boom/bust players alternate between high and low performances.
        Problematic for betting because outcomes are binary.

        Detection:
        - Count games significantly above/below average
        - Measure oscillation frequency

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with boom/bust features
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # Calculate player's average PRA (L20)
        df['PRA_avg_L20'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=20, min_periods=5)
            .mean()
        )

        # Define boom/bust threshold (1 std deviation)
        df['PRA_std_L20'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=20, min_periods=5)
            .std()
        )

        # Shifted PRA for comparison
        df['PRA_shifted'] = df.groupby('PLAYER_ID')['PRA'].shift(1)

        # Count boom games (performance > avg + 1 std)
        df['is_boom'] = (df['PRA_shifted'] > (df['PRA_avg_L20'] + df['PRA_std_L20'])).astype(int)
        df['boom_rate_L10'] = (
            df.groupby('PLAYER_ID')['is_boom']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .mean()
            .fillna(0)
        )

        # Count bust games (performance < avg - 1 std)
        df['is_bust'] = (df['PRA_shifted'] < (df['PRA_avg_L20'] - df['PRA_std_L20'])).astype(int)
        df['bust_rate_L10'] = (
            df.groupby('PLAYER_ID')['is_bust']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .mean()
            .fillna(0)
        )

        # Boom/bust score (0 = consistent, 1 = extreme boom/bust)
        df['boom_bust_score'] = df['boom_rate_L10'] + df['bust_rate_L10']

        # Oscillation detection (alternating high/low)
        # Calculate game-to-game changes
        df['pra_change'] = df.groupby('PLAYER_ID')['PRA'].diff()
        df['pra_change_sign'] = np.sign(df['pra_change'])

        # Count sign changes (indicates oscillation)
        df['sign_change'] = (df['pra_change_sign'] != df.groupby('PLAYER_ID')['pra_change_sign'].shift(1)).astype(int)
        df['oscillation_rate_L10'] = (
            df.groupby('PLAYER_ID')['sign_change']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .mean()
            .fillna(0)
        )

        # Clean up temporary columns
        df.drop(['PRA_shifted', 'is_boom', 'is_bust', 'pra_change', 'pra_change_sign', 'sign_change'],
                axis=1, inplace=True, errors='ignore')

        return df

    def calculate_floor_ceiling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate floor and ceiling (min/max expected performance).

        Floor = 10th percentile of recent performance
        Ceiling = 90th percentile of recent performance

        Tight range = consistent player
        Wide range = volatile player

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with floor/ceiling features
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # Calculate floor (10th percentile) and ceiling (90th percentile)
        for window in [10, 20]:
            # Floor (10th percentile)
            df[f'PRA_floor_L{window}'] = (
                df.groupby('PLAYER_ID')['PRA']
                .shift(1)
                .rolling(window=window, min_periods=5)
                .quantile(0.10)
            )

            # Ceiling (90th percentile)
            df[f'PRA_ceiling_L{window}'] = (
                df.groupby('PLAYER_ID')['PRA']
                .shift(1)
                .rolling(window=window, min_periods=5)
                .quantile(0.90)
            )

            # Range (ceiling - floor)
            df[f'PRA_range_L{window}'] = df[f'PRA_ceiling_L{window}'] - df[f'PRA_floor_L{window}']

        # Consistency from range (tight range = consistent)
        # Normalize to 0-100 scale
        df['range_consistency_L10'] = 100 - (df['PRA_range_L10'] / 2).clip(0, 100)

        return df

    def calculate_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate streak features (hot/cold streaks).

        Streaks indicate short-term consistency:
        - Current streak length
        - Longest recent streak
        - Streak volatility

        Args:
            df: DataFrame with performance stats

        Returns:
            DataFrame with streak features
        """
        df = df.copy()

        if 'PRA' not in df.columns:
            return df

        # Calculate if player exceeded their average
        df['PRA_L5_avg'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
        )

        df['PRA_shifted'] = df.groupby('PLAYER_ID')['PRA'].shift(1)
        df['exceeded_avg'] = (df['PRA_shifted'] > df['PRA_L5_avg']).astype(int)

        # Calculate current streak length
        def calculate_streak(series):
            """Calculate current streak length."""
            if len(series) == 0:
                return 0
            current = series.iloc[-1]
            streak = 1
            for i in range(len(series) - 2, -1, -1):
                if series.iloc[i] == current:
                    streak += 1
                else:
                    break
            return streak if current == 1 else -streak

        df['current_streak'] = (
            df.groupby('PLAYER_ID')['exceeded_avg']
            .shift(1)
            .rolling(window=10, min_periods=1)
            .apply(calculate_streak, raw=False)
            .fillna(0)
        )

        # Streak consistency (low streak variance = more predictable)
        df['streak_std_L10'] = (
            df.groupby('PLAYER_ID')['current_streak']
            .shift(1)
            .rolling(window=10, min_periods=3)
            .std()
            .fillna(0)
        )

        # Clean up temporary columns
        df.drop(['PRA_L5_avg', 'PRA_shifted', 'exceeded_avg'], axis=1, inplace=True, errors='ignore')

        return df

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all consistency features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with all consistency features added
        """
        print("Adding consistency features...")

        df = self.calculate_coefficient_of_variation(df)
        print("  ✓ Coefficient of Variation calculated")

        df = self.calculate_volatility_metrics(df)
        print("  ✓ Volatility metrics calculated")

        df = self.detect_boom_bust_tendency(df)
        print("  ✓ Boom/bust detection calculated")

        df = self.calculate_floor_ceiling(df)
        print("  ✓ Floor/ceiling calculated")

        df = self.calculate_streak_features(df)
        print("  ✓ Streak features calculated")

        # Count features added
        new_features = [col for col in df.columns if any(x in col for x in
                       ['_CV_', '_std_', 'volatility', 'consistency', 'boom', 'bust',
                        'floor', 'ceiling', 'range', 'streak', 'oscillation'])]
        print(f"  ✓ Added {len(new_features)} consistency features")

        return df


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_data = pd.DataFrame({
        'PLAYER_ID': [1] * 30 + [2] * 30,
        'GAME_DATE': pd.date_range('2024-01-01', periods=30).tolist() * 2,
        'PRA': [25, 27, 24, 26, 25, 28, 23, 26, 27, 25,  # Player 1: consistent
                26, 25, 27, 24, 26, 25, 28, 24, 26, 27,
                25, 26, 24, 27, 25, 26, 28, 25, 24, 26,
                15, 35, 12, 38, 18, 32, 10, 40, 16, 34,  # Player 2: boom/bust
                14, 36, 19, 31, 11, 39, 17, 33, 13, 37,
                16, 35, 12, 38, 15, 36, 18, 32, 14, 40],
        'PTS': [18, 19, 17, 18, 18, 20, 16, 18, 19, 18] * 6,
        'REB': [5, 6, 5, 6, 5, 6, 5, 6, 6, 5] * 6,
        'AST': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 6,
        'MIN': [32, 33, 31, 32, 32, 34, 30, 32, 33, 32] * 6
    })

    calculator = ConsistencyFeatures()
    result = calculator.add_all_features(sample_data)

    print("\nSample output columns:")
    consistency_cols = [col for col in result.columns if 'CV' in col or 'consistency' in col]
    print(consistency_cols[:10])

    print("\nPlayer 1 (consistent) vs Player 2 (boom/bust):")
    print(result[result['PLAYER_ID'] == 1][['PRA_CV_L10', 'consistency_score_L10', 'boom_bust_score']].tail(1))
    print(result[result['PLAYER_ID'] == 2][['PRA_CV_L10', 'consistency_score_L10', 'boom_bust_score']].tail(1))
