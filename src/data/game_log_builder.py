"""
Game-Level Dataset Builder for NBA Props Model

This module builds the training dataset by:
1. Loading 561K game logs with real PRA targets
2. Merging CTG season stats for context
3. Creating temporal features (lag, rolling, EWMA)
4. Adding opponent and rest/schedule features
5. Ensuring no data leakage with proper time-based features

Author: NBA Props Model
Date: 2024
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameLogDatasetBuilder:
    """Builds game-level training dataset from raw game logs and CTG stats."""

    def __init__(
        self,
        game_logs_path: str = "data/game_logs/all_game_logs_combined.csv",
        ctg_data_dir: str = "data/ctg_data_organized/players",
        output_dir: str = "data/processed",
    ):
        """
        Initialize the dataset builder.

        Args:
            game_logs_path: Path to combined game logs CSV
            ctg_data_dir: Directory containing CTG organized data
            output_dir: Where to save processed datasets
        """
        self.game_logs_path = Path(game_logs_path)
        self.ctg_data_dir = Path(ctg_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized GameLogDatasetBuilder")
        logger.info(f"Game logs: {self.game_logs_path}")
        logger.info(f"CTG data: {self.ctg_data_dir}")

    def load_game_logs(self) -> pd.DataFrame:
        """Load all game logs with proper data types."""
        logger.info("Loading game logs...")

        df = pd.read_csv(self.game_logs_path)

        # Convert GAME_DATE to datetime
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Sort by player and date (critical for temporal features)
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

        # Create basic derived features
        df["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
        df["OPPONENT"] = df["MATCHUP"].str.extract(r"(vs\.|@)\s*([A-Z]{3})")[1]

        logger.info(f"Loaded {len(df):,} game logs")
        logger.info(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        logger.info(f"Unique players: {df['PLAYER_ID'].nunique()}")

        return df

    def load_ctg_season_stats(self, season: str, season_type: str) -> Dict[str, pd.DataFrame]:
        """
        Load all CTG stat categories for a given season.

        Args:
            season: Season string (e.g., '2017-18')
            season_type: 'regular_season' or 'playoffs'

        Returns:
            Dictionary mapping category name to DataFrame
        """
        season_dir = self.ctg_data_dir / season / season_type

        if not season_dir.exists():
            logger.warning(f"Season directory not found: {season_dir}")
            return {}

        ctg_stats = {}

        # Load each category
        categories = [
            "shooting_frequency",
            "shooting_efficiency",
            "defense_and_rebounding",
            "foul_drawing",
            "on_off/team_shooting_accuracy",
            "on_off/team_offense",
            "on_off/team_defense",
        ]

        for category in categories:
            category_path = season_dir / category / f"{category.split('/')[-1]}.csv"

            if category_path.exists():
                df = pd.read_csv(category_path)
                # Use last part of category as key (e.g., 'team_defense')
                key = category.split("/")[-1]
                ctg_stats[key] = df
                logger.debug(f"Loaded {key}: {len(df)} players")

        return ctg_stats

    def merge_ctg_stats_to_games(
        self, game_logs: pd.DataFrame, min_minutes: int = 200
    ) -> pd.DataFrame:
        """
        Merge CTG season stats to each game log.

        Important: CTG stats are season-level, so they provide CONTEXT
        but temporal features will come from game logs themselves.

        Args:
            game_logs: DataFrame of game logs
            min_minutes: Minimum season minutes to include CTG stats

        Returns:
            DataFrame with CTG stats merged
        """
        logger.info("Merging CTG season stats to game logs...")

        result = game_logs.copy()

        # Get unique season-type combinations
        seasons = game_logs["SEASON"].unique()
        season_types_map = {"Regular Season": "regular_season", "Playoffs": "playoffs"}

        all_ctg_data = []

        for season in seasons:
            for season_type_label, season_type_dir in season_types_map.items():
                # Load CTG stats for this season
                ctg_stats = self.load_ctg_season_stats(season, season_type_dir)

                if not ctg_stats:
                    continue

                # Start with shooting_frequency as base (has Player, Team, MIN)
                if "shooting_frequency" not in ctg_stats:
                    continue

                base_df = ctg_stats["shooting_frequency"].copy()
                base_df = base_df[base_df["MIN"] >= min_minutes]  # Filter low-minute players

                # Add season identifiers
                base_df["SEASON"] = season
                base_df["SEASON_TYPE"] = season_type_label

                # Rename columns to avoid conflicts
                base_df = base_df.rename(
                    columns={
                        "MIN": "CTG_MIN",
                        "eFG%": "CTG_eFG_pct",
                        "Rim": "CTG_Rim_freq",
                        "Short Mid": "CTG_ShortMid_freq",
                        "Long Mid": "CTG_LongMid_freq",
                        "Corner Three": "CTG_Corner3_freq",
                        "Non Corner": "CTG_NonCorner3_freq",
                        "All Three": "CTG_All3_freq",
                    }
                )

                # Merge other CTG categories
                for category, cat_df in ctg_stats.items():
                    if category == "shooting_frequency":
                        continue

                    # CRITICAL FIX: Deduplicate players BEFORE merging
                    # If a player appears multiple times (traded, multiple teams), keep first occurrence
                    cat_df_dedup = cat_df.drop_duplicates(subset=["Player"], keep="first")

                    # Merge on Player
                    merge_cols = [
                        col
                        for col in cat_df_dedup.columns
                        if col not in ["Age", "Team", "Pos", "MIN"]
                    ]
                    base_df = base_df.merge(
                        cat_df_dedup[merge_cols],
                        on="Player",
                        how="left",
                        suffixes=("", f"_{category}"),
                    )

                all_ctg_data.append(base_df)

        # Combine all seasons
        if all_ctg_data:
            ctg_combined = pd.concat(all_ctg_data, ignore_index=True)
            logger.info(f"Combined CTG stats: {len(ctg_combined):,} player-seasons")

            # CRITICAL FIX: Deduplicate CTG combined BEFORE merging
            # Ensure one row per player-season-seasontype
            ctg_combined_dedup = ctg_combined.drop_duplicates(
                subset=["Player", "SEASON", "SEASON_TYPE"], keep="first"
            )

            logger.info(
                f"CTG before dedup: {len(ctg_combined):,}, after dedup: {len(ctg_combined_dedup):,}"
            )

            # Merge to game logs
            result = result.merge(
                ctg_combined_dedup,
                left_on=["PLAYER_NAME", "SEASON", "SEASON_TYPE"],
                right_on=["Player", "SEASON", "SEASON_TYPE"],
                how="left",
            )

            logger.info(f"Merged CTG stats. Shape: {result.shape}")
        else:
            logger.warning("No CTG stats found to merge")

        return result

    def create_lag_features(
        self,
        df: pd.DataFrame,
        stats: List[str] = ["PRA", "PTS", "REB", "AST", "MIN"],
        lags: List[int] = [1, 3, 5, 7, 10],
    ) -> pd.DataFrame:
        """
        Create lag features (previous game values).

        CRITICAL: Uses .shift() to prevent data leakage.
        lag=1 means previous game, NOT current game.

        Args:
            df: Game logs DataFrame (must be sorted by PLAYER_ID, GAME_DATE)
            stats: Statistics to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lag features added
        """
        logger.info(f"Creating lag features for {len(stats)} stats, lags={lags}...")

        result = df.copy()

        for stat in stats:
            if stat not in df.columns:
                logger.warning(f"Stat '{stat}' not found in DataFrame")
                continue

            for lag in lags:
                col_name = f"{stat}_lag{lag}"
                result[col_name] = result.groupby("PLAYER_ID")[stat].shift(lag)

        logger.info(f"Created {len(stats) * len(lags)} lag features")
        return result

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        stats: List[str] = ["PRA", "PTS", "REB", "AST", "MIN", "FG_PCT"],
        windows: List[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Create rolling average features.

        CRITICAL: Uses .shift(1) before rolling to prevent data leakage.

        Args:
            df: Game logs DataFrame
            stats: Statistics to calculate rolling averages
            windows: Window sizes for rolling calculations

        Returns:
            DataFrame with rolling features added
        """
        logger.info(f"Creating rolling features for {len(stats)} stats, windows={windows}...")

        result = df.copy()

        for stat in stats:
            if stat not in df.columns:
                logger.warning(f"Stat '{stat}' not found in DataFrame")
                continue

            for window in windows:
                # Mean
                col_mean = f"{stat}_L{window}_mean"
                result[col_mean] = (
                    result.groupby("PLAYER_ID")[stat]
                    .shift(1)  # Shift to exclude current game
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                # Std (volatility)
                col_std = f"{stat}_L{window}_std"
                result[col_std] = (
                    result.groupby("PLAYER_ID")[stat]
                    .shift(1)
                    .rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(level=0, drop=True)
                )

                # Max (ceiling/upside potential)
                col_max = f"{stat}_L{window}_max"
                result[col_max] = (
                    result.groupby("PLAYER_ID")[stat]
                    .shift(1)  # Shift to exclude current game
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=0, drop=True)
                )

        logger.info(f"Created {len(stats) * len(windows) * 3} rolling features (mean, std, max)")
        return result

    def create_ewma_features(
        self,
        df: pd.DataFrame,
        stats: List[str] = ["PRA", "PTS", "REB", "AST", "MIN"],
        spans: List[int] = [5, 10, 15],
    ) -> pd.DataFrame:
        """
        Create Exponentially Weighted Moving Average features.

        EWMA gives more weight to recent games, which is ideal for
        capturing current form.

        Args:
            df: Game logs DataFrame
            stats: Statistics to calculate EWMA
            spans: Span parameters (higher = less weight on recent)

        Returns:
            DataFrame with EWMA features added
        """
        logger.info(f"Creating EWMA features for {len(stats)} stats, spans={spans}...")

        result = df.copy()

        for stat in stats:
            if stat not in df.columns:
                continue

            for span in spans:
                col_name = f"{stat}_ewma{span}"
                result[col_name] = (
                    result.groupby("PLAYER_ID")[stat]
                    .shift(1)  # Prevent leakage
                    .ewm(span=span, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

        logger.info(f"Created {len(stats) * len(spans)} EWMA features")
        return result

    def create_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rest and schedule features.

        Research shows:
        - +37.6% win probability with 1+ day rest
        - Back-to-back games significantly impact performance

        Args:
            df: Game logs DataFrame

        Returns:
            DataFrame with rest features added
        """
        logger.info("Creating rest and schedule features...")

        result = df.copy()

        # Days since last game
        result["days_rest"] = (
            result.groupby("PLAYER_ID")["GAME_DATE"]
            .diff()
            .dt.days.fillna(7)  # First game of tracking, assume rested
        )

        # Back-to-back game indicator
        result["is_b2b"] = (result["days_rest"] <= 1).astype(int)

        # Games in last 7 days (fatigue indicator)
        # Count games in rolling 7-day window for each player
        def count_games_last_7d(group):
            games_count = []
            for idx, date in enumerate(group):
                seven_days_ago = date - timedelta(days=7)
                count = ((group[:idx] >= seven_days_ago) & (group[:idx] < date)).sum()
                games_count.append(count)
            return pd.Series(games_count, index=group.index)

        result["games_last_7d"] = result.groupby("PLAYER_ID")["GAME_DATE"].transform(
            count_games_last_7d
        )

        logger.info("Created rest/schedule features")
        return result

    def create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create opponent-related features.

        Note: Advanced opponent features (defensive rating by position)
        require team data integration, which will be added separately.

        Args:
            df: Game logs DataFrame

        Returns:
            DataFrame with basic opponent features
        """
        logger.info("Creating opponent features...")

        result = df.copy()

        # Opponent's recent defensive performance (simple version)
        # Calculate average PRA allowed by opponent in last 10 games
        opp_stats = (
            result.groupby(["OPPONENT", "GAME_DATE"])
            .agg({"PRA": "mean", "PTS": "mean", "REB": "mean", "AST": "mean"})
            .reset_index()
        )

        opp_stats = opp_stats.sort_values("GAME_DATE")

        for stat in ["PRA", "PTS", "REB", "AST"]:
            col_name = f"opp_allowed_{stat}_L10"
            opp_stats[col_name] = (
                opp_stats.groupby("OPPONENT")[stat]
                .shift(1)
                .rolling(window=10, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Merge back
        result = result.merge(
            opp_stats[
                ["OPPONENT", "GAME_DATE"]
                + [f"opp_allowed_{s}_L10" for s in ["PRA", "PTS", "REB", "AST"]]
            ],
            on=["OPPONENT", "GAME_DATE"],
            how="left",
        )

        logger.info("Created opponent features")
        return result

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend features (is player improving or declining?).

        Compares recent performance (L5) to longer-term (L20).

        Args:
            df: Game logs DataFrame

        Returns:
            DataFrame with trend features
        """
        logger.info("Creating trend features...")

        result = df.copy()

        stats = ["PRA", "PTS", "REB", "AST", "MIN"]

        for stat in stats:
            if stat not in df.columns:
                continue

            # L5 average
            l5 = (
                result.groupby("PLAYER_ID")[stat]
                .shift(1)
                .rolling(window=5, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # L20 average
            l20 = (
                result.groupby("PLAYER_ID")[stat]
                .shift(1)
                .rolling(window=20, min_periods=10)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Trend = (L5 - L20) / L20
            result[f"{stat}_trend"] = ((l5 - l20) / l20).fillna(0)

        logger.info("Created trend features")
        return result

    def create_hot_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hot streak indicator features.

        Identifies when a player is on a hot streak based on recent performance
        exceeding their baseline.

        Args:
            df: Game logs DataFrame

        Returns:
            DataFrame with hot streak features
        """
        logger.info("Creating hot streak features...")

        result = df.copy()

        stats = ["PRA", "PTS", "REB", "AST", "MIN"]

        for stat in stats:
            if stat not in df.columns:
                continue

            # L3 average (very recent form)
            l3 = (
                result.groupby("PLAYER_ID")[stat]
                .shift(1)
                .rolling(window=3, min_periods=2)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # L10 average (baseline)
            l10 = (
                result.groupby("PLAYER_ID")[stat]
                .shift(1)
                .rolling(window=10, min_periods=5)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Hot streak: L3 > L10 by at least 10%
            result[f"{stat}_hot_streak"] = ((l3 - l10) / l10 > 0.10).astype(int).fillna(0)

            # Hot streak magnitude (how much hotter)
            result[f"{stat}_hot_streak_magnitude"] = ((l3 - l10) / l10).fillna(0)

            # Count consecutive games above L10 average
            # Simplified approach: count how many of last 5 games were above L10 avg
            shifted_stat = result.groupby("PLAYER_ID")[stat].shift(1)
            above_avg = (shifted_stat > l10).astype(int).fillna(0)

            result[f"{stat}_consecutive_above_avg"] = (
                above_avg.groupby(result["PLAYER_ID"])
                .rolling(window=5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

        logger.info("Created hot streak features")
        return result

    def build_complete_dataset(
        self, merge_ctg: bool = True, min_minutes_per_game: float = 10.0, min_games_played: int = 5
    ) -> pd.DataFrame:
        """
        Build the complete game-level training dataset.

        Steps:
        1. Load game logs (561K records)
        2. Merge CTG season stats (optional)
        3. Create temporal features (lag, rolling, EWMA)
        4. Create rest/schedule features
        5. Create opponent features
        6. Create trend features
        7. Filter to relevant games

        Args:
            merge_ctg: Whether to merge CTG season stats
            min_minutes_per_game: Minimum minutes to include game
            min_games_played: Minimum games for player to be included

        Returns:
            Complete training dataset
        """
        logger.info("=" * 80)
        logger.info("Building Complete Game-Level Training Dataset")
        logger.info("=" * 80)

        # Step 1: Load game logs
        df = self.load_game_logs()
        initial_count = len(df)

        # Step 2: Merge CTG stats (optional)
        if merge_ctg:
            df = self.merge_ctg_stats_to_games(df)

        # Step 3: Create temporal features
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_ewma_features(df)

        # Step 4: Create rest features
        df = self.create_rest_features(df)

        # Step 5: Create opponent features
        df = self.create_opponent_features(df)

        # Step 6: Create trend features
        df = self.create_trend_features(df)

        # Step 7: Create hot streak features
        df = self.create_hot_streak_features(df)

        # Step 8: Filter to relevant games
        logger.info("Filtering dataset...")

        # Filter by minutes
        df = df[df["MIN"] >= min_minutes_per_game].copy()

        # Filter players with minimum games
        player_game_counts = df.groupby("PLAYER_ID").size()
        valid_players = player_game_counts[player_game_counts >= min_games_played].index
        df = df[df["PLAYER_ID"].isin(valid_players)].copy()

        # Drop rows with missing PRA target
        df = df.dropna(subset=["PRA"])

        # CRITICAL FIX: Final deduplication safety check
        # Remove any duplicate (PLAYER_ID, GAME_DATE) combinations
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE"], keep="first")
        after_dedup = len(df)

        if before_dedup != after_dedup:
            logger.warning(
                f"⚠️  Removed {before_dedup - after_dedup:,} duplicate player-game combinations!"
            )

        final_count = len(df)
        logger.info(
            f"Filtered: {initial_count:,} -> {final_count:,} games ({final_count/initial_count*100:.1f}%)"
        )

        # Summary stats
        logger.info("=" * 80)
        logger.info("Dataset Summary")
        logger.info("=" * 80)
        logger.info(f"Total games: {len(df):,}")
        logger.info(f"Unique players: {df['PLAYER_ID'].nunique():,}")
        logger.info(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
        logger.info(f"Seasons: {sorted(df['SEASON'].unique())}")
        logger.info(f"Total features: {len(df.columns)}")
        logger.info(f"PRA range: {df['PRA'].min():.1f} to {df['PRA'].max():.1f}")
        logger.info(f"PRA mean: {df['PRA'].mean():.2f} ± {df['PRA'].std():.2f}")

        return df

    def save_dataset(
        self, df: pd.DataFrame, filename: str = "game_level_training_data.parquet"
    ) -> Path:
        """
        Save the dataset to parquet format.

        Args:
            df: Dataset to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        logger.info(f"Saving dataset to {output_path}...")
        df.to_parquet(output_path, index=False)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df):,} records ({file_size_mb:.1f} MB)")

        return output_path

    def create_train_test_split(
        self, df: pd.DataFrame, train_end_date: str = "2023-06-30", val_end_date: str = "2024-06-30"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test split.

        CRITICAL: Must use chronological split for time series data.
        Never use random split!

        Args:
            df: Complete dataset
            train_end_date: Last date for training set
            val_end_date: Last date for validation set

        Returns:
            train, validation, test DataFrames
        """
        logger.info("Creating time-based train/val/test split...")

        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)

        train = df[df["GAME_DATE"] <= train_end].copy()
        val = df[(df["GAME_DATE"] > train_end) & (df["GAME_DATE"] <= val_end)].copy()
        test = df[df["GAME_DATE"] > val_end].copy()

        logger.info(
            f"Train: {len(train):,} games ({train['GAME_DATE'].min()} to {train['GAME_DATE'].max()})"
        )
        logger.info(
            f"Val:   {len(val):,} games ({val['GAME_DATE'].min()} to {val['GAME_DATE'].max()})"
        )
        logger.info(
            f"Test:  {len(test):,} games ({test['GAME_DATE'].min()} to {test['GAME_DATE'].max()})"
        )

        return train, val, test


if __name__ == "__main__":
    # Example usage
    builder = GameLogDatasetBuilder()

    # Build complete dataset
    dataset = builder.build_complete_dataset(
        merge_ctg=True,
        min_minutes_per_game=15.0,  # Only games with 15+ minutes
        min_games_played=10,  # Only players with 10+ games
    )

    # Save full dataset
    builder.save_dataset(dataset, "game_level_training_data.parquet")

    # Create train/val/test splits
    train, val, test = builder.create_train_test_split(dataset)

    # Save splits
    builder.save_dataset(train, "train.parquet")
    builder.save_dataset(val, "val.parquet")
    builder.save_dataset(test, "test.parquet")

    logger.info("✅ Dataset building complete!")
