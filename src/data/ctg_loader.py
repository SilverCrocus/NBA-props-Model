"""
Simple CTG data loader
Load and combine CTG player stats for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class CTGDataLoader:
    """
    Load CTG data from organized directory structure
    Simple and direct - no fallback logic
    """

    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.player_data_path = self.data_path / "ctg_data_organized" / "players"

    def load_season_data(self, season: str, stat_type: str = "all") -> pd.DataFrame:
        """
        Load player data for a specific season

        Args:
            season: Season string (e.g., '2023-24')
            stat_type: Type of stats to load ('shooting', 'defense', 'all')
        """
        season_path = self.player_data_path / season / "regular_season"

        if not season_path.exists():
            raise ValueError(f"Season path does not exist: {season_path}")

        # Map stat types to directory names
        stat_dirs = {
            'shooting': ['shooting_frequency', 'shooting_accuracy'],
            'defense': ['defense_and_rebounding'],
            'passing': ['passing'],
            'all': ['shooting_frequency', 'shooting_accuracy',
                   'defense_and_rebounding', 'passing', 'foul_drawing']
        }

        if stat_type not in stat_dirs:
            stat_type = 'all'

        combined_df = None

        for stat_dir in stat_dirs[stat_type]:
            dir_path = season_path / stat_dir
            if dir_path.exists():
                csv_files = list(dir_path.glob("*.csv"))
                if csv_files:
                    # Use the first CSV file found
                    df = pd.read_csv(csv_files[0])
                    df['Season'] = season
                    df['StatType'] = stat_dir

                    if combined_df is None:
                        combined_df = df
                    else:
                        # Merge on common columns
                        merge_cols = ['Player', 'Team', 'Season']
                        available_merge_cols = [col for col in merge_cols if col in df.columns]

                        combined_df = pd.merge(
                            combined_df, df,
                            on=available_merge_cols,
                            how='outer',
                            suffixes=('', f'_{stat_dir}')
                        )

        if combined_df is None:
            raise ValueError(f"No data found for season {season}")

        logger.info(f"Loaded {len(combined_df)} players for season {season}")
        return combined_df

    def load_multiple_seasons(self, seasons: List[str]) -> pd.DataFrame:
        """
        Load and combine data from multiple seasons
        """
        all_data = []

        for season in seasons:
            try:
                season_data = self.load_season_data(season)
                all_data.append(season_data)
                logger.info(f"Loaded season {season}: {len(season_data)} rows")
            except Exception as e:
                logger.error(f"Failed to load season {season}: {e}")

        if not all_data:
            raise ValueError("No data loaded from any season")

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total data loaded: {len(combined)} rows from {len(all_data)} seasons")

        return combined

    def create_game_log_format(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to game log format for model training
        Add synthetic game dates and opponents for now
        """
        # This is a simplified version - you'd need actual game logs
        # For now, create synthetic data structure

        game_logs = []

        for _, row in player_stats.iterrows():
            # Create multiple game entries per player
            # This is placeholder - replace with actual game data

            player = row['Player']
            season = row['Season']
            team = row.get('Team', 'UNK')

            # Create 20 synthetic games per player-season
            for game_num in range(20):
                game_date = pd.Timestamp(f"2024-01-01") + pd.Timedelta(days=game_num * 3)

                # Synthetic opponent rotation
                opponents = ['BOS', 'LAL', 'MIA', 'GSW', 'BKN', 'PHI', 'DEN', 'MIL']
                opponent = opponents[game_num % len(opponents)]

                game_entry = {
                    'Player': player,
                    'Date': game_date,
                    'Season': season,
                    'Team': team,
                    'Opponent': opponent,
                    'GameNum': game_num + 1
                }

                # Add stats with some variance
                for col in row.index:
                    if col not in ['Player', 'Season', 'Team', 'Age', 'Pos']:
                        if pd.notna(row[col]) and isinstance(row[col], (int, float)):
                            # Add random variance to create game-by-game variation
                            base_value = row[col]
                            variance = base_value * 0.2 if base_value != 0 else 1
                            game_value = np.random.normal(base_value, abs(variance))
                            game_entry[col] = max(0, game_value)  # Ensure non-negative

                game_logs.append(game_entry)

        return pd.DataFrame(game_logs)

    def get_available_seasons(self) -> List[str]:
        """
        Get list of available seasons in the data directory
        """
        seasons = []

        if self.player_data_path.exists():
            for season_dir in self.player_data_path.iterdir():
                if season_dir.is_dir() and '-' in season_dir.name:
                    seasons.append(season_dir.name)

        return sorted(seasons)


# Example usage
if __name__ == "__main__":
    loader = CTGDataLoader()

    # Get available seasons
    seasons = loader.get_available_seasons()
    print(f"Available seasons: {seasons}")

    if seasons:
        # Load most recent season
        recent_season = seasons[-1]
        data = loader.load_season_data(recent_season)
        print(f"Loaded {recent_season}: {data.shape}")
        print(f"Columns: {list(data.columns[:10])}")  # First 10 columns