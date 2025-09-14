#!/usr/bin/env python3
"""
Fetch NBA game box scores for all seasons (2003-04 to 2024-25)
Includes actual PRA values from real games
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
import sys

# NBA API imports
try:
    from nba_api.stats.endpoints import leaguegamelog, playergamelog
    from nba_api.stats.static import players, teams
except ImportError:
    print("ERROR: nba_api not installed. Please run: uv add nba-api")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_logs_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBAGameLogFetcher:
    """
    Fetches game-by-game box scores for NBA seasons
    """

    def __init__(self, data_dir: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_dir = Path(data_dir)
        self.game_logs_dir = self.data_dir / "game_logs"
        self.game_logs_dir.mkdir(exist_ok=True)

        # Progress tracking
        self.progress_file = self.game_logs_dir / "fetch_progress.json"
        self.progress = self.load_progress()

        # Rate limiting
        self.request_delay = 0.6  # 600ms between requests to be safe
        self.retry_delay = 5.0    # 5 seconds on error
        self.max_retries = 3

        # Seasons to fetch (matching your CTG data)
        self.seasons = [
            '2003-04', '2004-05', '2005-06', '2006-07', '2007-08',
            '2008-09', '2009-10', '2010-11', '2011-12', '2012-13',
            '2013-14', '2014-15', '2015-16', '2016-17', '2017-18',
            '2018-19', '2019-20', '2020-21', '2021-22', '2022-23',
            '2023-24'  # 2024-25 might not have games yet
        ]

    def load_progress(self) -> Dict:
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'completed_seasons': [],
            'partial_seasons': {},
            'total_games_fetched': 0,
            'last_update': None
        }

    def save_progress(self):
        """Save progress to file"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def fetch_season_game_logs(self, season: str, season_type: str = 'Regular Season') -> Optional[pd.DataFrame]:
        """
        Fetch all player game logs for a season

        Args:
            season: Season string (e.g., '2023-24')
            season_type: 'Regular Season' or 'Playoffs'

        Returns:
            DataFrame with all player game logs
        """
        logger.info(f"Fetching {season_type} game logs for {season}...")

        for attempt in range(self.max_retries):
            try:
                # Add delay to respect rate limits
                time.sleep(self.request_delay)

                # Fetch game logs
                game_log = leaguegamelog.LeagueGameLog(
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation='P'  # 'P' for players
                )

                df = game_log.get_data_frames()[0]

                # Add metadata
                df['SEASON'] = season
                df['SEASON_TYPE'] = season_type

                # Calculate PRA (Points + Rebounds + Assists)
                df['PRA'] = df['PTS'] + df['REB'] + df['AST']

                # Add additional useful metrics
                df['DOUBLE_DOUBLE'] = ((df['PTS'] >= 10).astype(int) +
                                       (df['REB'] >= 10).astype(int) +
                                       (df['AST'] >= 10).astype(int)) >= 2

                df['TRIPLE_DOUBLE'] = ((df['PTS'] >= 10).astype(int) +
                                       (df['REB'] >= 10).astype(int) +
                                       (df['AST'] >= 10).astype(int)) == 3

                # Calculate fantasy points (DraftKings scoring)
                df['DK_POINTS'] = (df['PTS'] * 1.0 +
                                  df['REB'] * 1.25 +
                                  df['AST'] * 1.5 +
                                  df['STL'] * 2.0 +
                                  df['BLK'] * 2.0 -
                                  df['TOV'] * 0.5 +
                                  df['DOUBLE_DOUBLE'] * 1.5 +
                                  df['TRIPLE_DOUBLE'] * 3.0)

                # Calculate FanDuel points
                df['FD_POINTS'] = (df['PTS'] * 1.0 +
                                  df['REB'] * 1.2 +
                                  df['AST'] * 1.5 +
                                  df['STL'] * 3.0 +
                                  df['BLK'] * 3.0 -
                                  df['TOV'] * 1.0)

                logger.info(f"✓ Fetched {len(df)} game records for {season} {season_type}")
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {season}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch {season} after {self.max_retries} attempts")
                    return None

        return None

    def fetch_all_seasons(self, include_playoffs: bool = True):
        """
        Fetch game logs for all seasons

        Args:
            include_playoffs: Whether to fetch playoff games
        """
        total_seasons = len(self.seasons)
        logger.info(f"Starting to fetch {total_seasons} seasons of NBA game data")
        logger.info(f"Previously completed: {len(self.progress['completed_seasons'])} seasons")

        all_games = []
        season_stats = []

        for i, season in enumerate(self.seasons, 1):
            # Skip if already completed
            if season in self.progress['completed_seasons']:
                logger.info(f"[{i}/{total_seasons}] Skipping {season} (already completed)")
                continue

            logger.info(f"\n[{i}/{total_seasons}] Processing {season}")
            logger.info("=" * 60)

            season_games = []

            # Fetch regular season
            regular_df = self.fetch_season_game_logs(season, 'Regular Season')
            if regular_df is not None:
                season_games.append(regular_df)

            # Fetch playoffs if requested
            if include_playoffs:
                playoff_df = self.fetch_season_game_logs(season, 'Playoffs')
                if playoff_df is not None and len(playoff_df) > 0:
                    season_games.append(playoff_df)

            if season_games:
                # Combine regular season and playoffs
                season_df = pd.concat(season_games, ignore_index=True)

                # Save season file
                season_file = self.game_logs_dir / f"game_logs_{season}.csv"
                season_df.to_csv(season_file, index=False)
                logger.info(f"✓ Saved {len(season_df)} games to {season_file}")

                # Calculate season statistics
                season_stat = {
                    'season': season,
                    'total_games': len(season_df),
                    'unique_players': season_df['PLAYER_NAME'].nunique(),
                    'avg_pra': season_df['PRA'].mean(),
                    'max_pra': season_df['PRA'].max(),
                    'player_max_pra': season_df.loc[season_df['PRA'].idxmax(), 'PLAYER_NAME'] if len(season_df) > 0 else None,
                    'date_max_pra': season_df.loc[season_df['PRA'].idxmax(), 'GAME_DATE'] if len(season_df) > 0 else None
                }
                season_stats.append(season_stat)

                # Update progress
                self.progress['completed_seasons'].append(season)
                self.progress['total_games_fetched'] += len(season_df)
                self.save_progress()

                all_games.append(season_df)

                # Show progress
                pct_complete = (len(self.progress['completed_seasons']) / total_seasons) * 100
                logger.info(f"Progress: {pct_complete:.1f}% complete")
                logger.info(f"Total games fetched so far: {self.progress['total_games_fetched']:,}")

        # Combine all seasons
        if all_games:
            logger.info("\nCombining all seasons...")
            all_games_df = pd.concat(all_games, ignore_index=True)

            # Save combined file
            combined_file = self.game_logs_dir / "all_game_logs_combined.csv"
            all_games_df.to_csv(combined_file, index=False)
            logger.info(f"✓ Saved combined file with {len(all_games_df)} total games")

            # Save season statistics
            if season_stats:
                stats_df = pd.DataFrame(season_stats)
                stats_file = self.game_logs_dir / "season_statistics.csv"
                stats_df.to_csv(stats_file, index=False)
                logger.info(f"✓ Saved season statistics to {stats_file}")

            # Print summary statistics
            self.print_summary(all_games_df)

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("FETCH COMPLETE - SUMMARY STATISTICS")
        logger.info("=" * 60)

        logger.info(f"\nDataset Overview:")
        logger.info(f"  Total game records: {len(df):,}")
        logger.info(f"  Unique players: {df['PLAYER_NAME'].nunique():,}")
        logger.info(f"  Seasons covered: {df['SEASON'].nunique()}")
        logger.info(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

        logger.info(f"\nPRA Statistics:")
        logger.info(f"  Mean PRA: {df['PRA'].mean():.2f}")
        logger.info(f"  Median PRA: {df['PRA'].median():.2f}")
        logger.info(f"  Std Dev: {df['PRA'].std():.2f}")
        logger.info(f"  Min PRA: {df['PRA'].min():.2f}")
        logger.info(f"  Max PRA: {df['PRA'].max():.2f}")

        # Top performances
        top_pra = df.nlargest(5, 'PRA')[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST', 'PRA']]
        logger.info(f"\nTop 5 PRA Performances:")
        for _, row in top_pra.iterrows():
            logger.info(f"  {row['PLAYER_NAME']} ({row['GAME_DATE']}): "
                       f"{row['PRA']:.0f} PRA ({row['PTS']}P/{row['REB']}R/{row['AST']}A)")

        # Distribution
        logger.info(f"\nPRA Distribution:")
        logger.info(f"  Games with PRA > 60: {(df['PRA'] > 60).sum():,}")
        logger.info(f"  Games with PRA > 50: {(df['PRA'] > 50).sum():,}")
        logger.info(f"  Games with PRA > 40: {(df['PRA'] > 40).sum():,}")
        logger.info(f"  Games with PRA > 30: {(df['PRA'] > 30).sum():,}")
        logger.info(f"  Games with PRA > 20: {(df['PRA'] > 20).sum():,}")

        logger.info(f"\nData saved to: {self.game_logs_dir}")
        logger.info(f"Progress file: {self.progress_file}")

    def verify_data_quality(self):
        """Verify the quality of fetched data"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY CHECK")
        logger.info("=" * 60)

        for season in self.progress['completed_seasons']:
            season_file = self.game_logs_dir / f"game_logs_{season}.csv"
            if season_file.exists():
                df = pd.read_csv(season_file)

                # Check for missing values
                missing = df[['PLAYER_NAME', 'PTS', 'REB', 'AST', 'MIN']].isnull().sum()
                if missing.any():
                    logger.warning(f"{season}: Missing values found - {missing.to_dict()}")

                # Check for anomalies
                if (df['PRA'] < 0).any():
                    logger.warning(f"{season}: Negative PRA values found!")

                if (df['MIN'] > 70).any():  # NBA game is 48 minutes + OT
                    logger.warning(f"{season}: Minutes > 70 found (multiple OT games)")

                logger.info(f"✓ {season}: {len(df)} games verified")


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("NBA GAME LOGS FETCHER")
    print("=" * 60)
    print("\nThis script will fetch game-by-game box scores for:")
    print("- Seasons: 2003-04 to 2023-24 (21 seasons)")
    print("- Both regular season and playoff games")
    print("- Includes PRA calculations and fantasy points")
    print("\nEstimated time: 15-30 minutes (with rate limiting)")
    print("The script will resume from where it left off if interrupted")

    response = input("\nProceed with fetching? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Create fetcher and run
    fetcher = NBAGameLogFetcher()

    try:
        # Fetch all seasons
        fetcher.fetch_all_seasons(include_playoffs=True)

        # Verify data quality
        fetcher.verify_data_quality()

        print("\n✅ SUCCESS! All game logs fetched successfully.")
        print(f"📁 Data saved to: {fetcher.game_logs_dir}")
        print(f"📊 Total games: {fetcher.progress['total_games_fetched']:,}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Progress has been saved.")
        print("Run the script again to resume from where you left off.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()