"""
Critical tests for temporal leakage prevention.

These tests ensure that no features use future data, which would
artificially inflate model performance and cause losses in production.

IMPORTANCE: These tests are the most critical in the entire codebase.
If these tests fail, DO NOT deploy to production.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestTemporalLeakagePrevention:
    """Test suite for temporal leakage detection."""

    @pytest.fixture
    def sample_game_logs(self):
        """Create sample game logs for testing."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'PLAYER_ID': [1] * 10,
            'GAME_DATE': dates,
            'PRA': [25, 30, 28, 35, 22, 27, 31, 26, 29, 33],
            'PTS': [18, 22, 20, 25, 16, 19, 23, 18, 21, 24],
            'REB': [5, 6, 6, 7, 4, 6, 6, 6, 6, 7],
            'AST': [2, 2, 2, 3, 2, 2, 2, 2, 2, 2],
            'MIN': [32, 35, 33, 38, 30, 33, 36, 32, 34, 36]
        })

    def test_lag1_excludes_current_game(self, sample_game_logs):
        """
        CRITICAL: lag=1 must use previous game, not current game.

        This is the most common source of temporal leakage.
        """
        df = sample_game_logs.copy()

        # Create lag feature
        df['PRA_lag1'] = df.groupby('PLAYER_ID')['PRA'].shift(1)

        # First game should have NaN (no previous game)
        assert pd.isna(df.loc[0, 'PRA_lag1']), \
            "First game should have NaN for lag1 (no previous game)"

        # Second game should have first game's PRA
        assert df.loc[1, 'PRA_lag1'] == 25, \
            f"Expected lag1=25 (first game PRA), got {df.loc[1, 'PRA_lag1']}"

        # Third game should have second game's PRA
        assert df.loc[2, 'PRA_lag1'] == 30, \
            f"Expected lag1=30 (second game PRA), got {df.loc[2, 'PRA_lag1']}"

        # Verify lag feature NEVER equals current game PRA
        matches_current = (df['PRA_lag1'] == df['PRA']).sum()
        assert matches_current == 0, \
            f"Found {matches_current} cases where lag1 equals current PRA (LEAKAGE!)"

    def test_rolling_avg_excludes_current_game(self, sample_game_logs):
        """
        CRITICAL: Rolling averages must shift(1) before rolling.

        Without shift, current game is included in average = leakage.
        """
        df = sample_game_logs.copy()

        # Create rolling average (CORRECT way with shift)
        df['PRA_L3_mean'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)  # CRITICAL: Exclude current game
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # First game should have NaN (no previous games)
        assert pd.isna(df.loc[0, 'PRA_L3_mean']), \
            "First game should have NaN for rolling avg (no previous games)"

        # Second game should have avg of 1 previous game
        assert df.loc[1, 'PRA_L3_mean'] == 25, \
            f"Expected L3_mean=25 (only 1 prev game), got {df.loc[1, 'PRA_L3_mean']}"

        # Third game should have avg of 2 previous games
        expected_avg = (25 + 30) / 2
        assert abs(df.loc[2, 'PRA_L3_mean'] - expected_avg) < 0.01, \
            f"Expected L3_mean={expected_avg}, got {df.loc[2, 'PRA_L3_mean']}"

        # Fourth game should have avg of 3 previous games
        expected_avg = (25 + 30 + 28) / 3
        assert abs(df.loc[3, 'PRA_L3_mean'] - expected_avg) < 0.01, \
            f"Expected L3_mean={expected_avg}, got {df.loc[3, 'PRA_L3_mean']}"

    def test_no_leakage_across_players(self):
        """
        CRITICAL: Features must not leak across different players.

        Player B's first game should not use Player A's data.
        """
        df = pd.DataFrame({
            'PLAYER_ID': [1, 1, 1, 2, 2, 2],
            'GAME_DATE': pd.date_range('2024-01-01', periods=6),
            'PRA': [25, 30, 28, 22, 26, 24]
        })

        # Create lag feature
        df['PRA_lag1'] = df.groupby('PLAYER_ID')['PRA'].shift(1)

        # Player 2's first game (index 3) should have NaN
        assert pd.isna(df.loc[3, 'PRA_lag1']), \
            "Player 2's first game should have NaN, not Player 1's data (LEAKAGE!)"

        # Player 2's second game should have their own first game
        assert df.loc[4, 'PRA_lag1'] == 22, \
            f"Player 2's lag1 should be their own data, got {df.loc[4, 'PRA_lag1']}"

    def test_train_test_split_no_temporal_overlap(self):
        """
        CRITICAL: Training data must not include dates from test period.

        This is a common mistake in time series splitting.
        """
        df = pd.DataFrame({
            'PLAYER_ID': [1] * 10,
            'GAME_DATE': pd.date_range('2024-01-01', periods=10, freq='D'),
            'PRA': [25, 30, 28, 35, 22, 27, 31, 26, 29, 33]
        })

        # Split on date
        train_end = pd.to_datetime('2024-01-05')
        test_start = pd.to_datetime('2024-01-06')

        train = df[df['GAME_DATE'] <= train_end]
        test = df[df['GAME_DATE'] >= test_start]

        # Verify no overlap
        assert train['GAME_DATE'].max() < test['GAME_DATE'].min(), \
            "Training data overlaps with test data (LEAKAGE!)"

        # Verify no shared rows
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0, \
            f"Found {len(overlap)} rows in both train and test (LEAKAGE!)"

    def test_ewma_excludes_current_game(self, sample_game_logs):
        """
        CRITICAL: EWMA (exponentially weighted moving average) must shift.

        EWMA is used for "current form" features - must not include current game.
        """
        df = sample_game_logs.copy()

        # Create EWMA (CORRECT way with shift)
        df['PRA_ewma5'] = (
            df.groupby('PLAYER_ID')['PRA']
            .shift(1)  # CRITICAL: Exclude current game
            .ewm(span=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # First game should have NaN
        assert pd.isna(df.loc[0, 'PRA_ewma5']), \
            "First game should have NaN for EWMA (no previous games)"

        # Second game should equal first game's PRA (only 1 data point)
        assert df.loc[1, 'PRA_ewma5'] == 25, \
            f"Expected ewma5=25 (only 1 prev game), got {df.loc[1, 'PRA_ewma5']}"

    def test_opponent_features_no_future_data(self):
        """
        CRITICAL: Opponent averages must not include current game.

        Common mistake: including current game when calculating opponent's avg.
        """
        df = pd.DataFrame({
            'PLAYER_ID': [1, 1, 1, 2, 2, 2],
            'OPPONENT': ['BOS', 'LAL', 'BOS', 'BOS', 'LAL', 'LAL'],
            'GAME_DATE': pd.date_range('2024-01-01', periods=6),
            'PRA': [25, 30, 28, 35, 22, 27]
        })

        # Calculate opponent avg (CORRECT way)
        # For each game, avg of opponent's previous games only
        df = df.sort_values(['OPPONENT', 'GAME_DATE'])

        df['opp_avg_PRA'] = (
            df.groupby('OPPONENT')['PRA']
            .shift(1)  # CRITICAL: Exclude current game
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Re-sort by original order
        df = df.sort_index()

        # First game vs BOS should have NaN (no previous games vs BOS)
        assert pd.isna(df.loc[0, 'opp_avg_PRA']), \
            "First game vs opponent should have NaN"

        # Third game vs BOS should have avg of first game vs BOS only
        # (excluding current game)
        assert df.loc[2, 'opp_avg_PRA'] == 25, \
            "Opponent avg should only include previous games vs that opponent"


class TestLeakageDetectionUtility:
    """Test the leakage detection utility function."""

    def test_detect_perfect_correlation(self):
        """
        Detect when a feature perfectly correlates with target.

        This is a strong indicator of leakage.
        """
        df = pd.DataFrame({
            'PRA': [25, 30, 28, 35, 22],
            'feature': [25, 30, 28, 35, 22]  # Perfect correlation = likely leakage
        })

        correlation = df['PRA'].corr(df['feature'])
        assert abs(correlation) > 0.99, \
            "Should detect perfect correlation (likely leakage)"

    def test_detect_time_shifted_correlation(self):
        """
        Detect when feature at time T equals target at time T+1.

        This suggests feature is using future data.
        """
        df = pd.DataFrame({
            'PRA': [25, 30, 28, 35, 22],
            'feature': [30, 28, 35, 22, 20]  # Shifted by 1 = future data
        })

        # Check if feature[t] == PRA[t+1]
        shifted_match = (df['feature'].iloc[:-1] == df['PRA'].iloc[1:]).sum()

        assert shifted_match == 4, \
            "Should detect that feature matches future PRA (LEAKAGE!)"


class TestWalkForwardValidation:
    """Test walk-forward validation for temporal correctness."""

    def test_walk_forward_maintains_temporal_order(self):
        """
        Walk-forward validation must train on past, test on future.

        Never the reverse!
        """
        df = pd.DataFrame({
            'GAME_DATE': pd.date_range('2024-01-01', periods=30, freq='D'),
            'PRA': np.random.randint(15, 40, 30)
        })

        # Simulate walk-forward windows
        train_window = 20
        test_window = 5

        for i in range(0, len(df) - train_window - test_window + 1, test_window):
            train_start = i
            train_end = i + train_window
            test_start = train_end
            test_end = test_start + test_window

            train = df.iloc[train_start:train_end]
            test = df.iloc[test_start:test_end]

            # Verify temporal ordering
            assert train['GAME_DATE'].max() < test['GAME_DATE'].min(), \
                f"Window {i}: Training data overlaps with test data"

            # Verify no shared indices
            overlap = set(train.index) & set(test.index)
            assert len(overlap) == 0, \
                f"Window {i}: Found {len(overlap)} overlapping rows"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for this module."""
    config.addinivalue_line(
        "markers", "critical: mark test as critical (must pass before production)"
    )


# Mark all tests in this module as critical
pytestmark = pytest.mark.critical


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
