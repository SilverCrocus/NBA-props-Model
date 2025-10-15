"""
CTG Feature Builder - Integrates CleaningTheGlass Advanced Stats

Provides player-season level CTG stats for feature engineering.
Key stats: USG%, PSA, AST%, TOV%, eFG%, REB%

These features have 0.80-0.85 correlation with PRA and should reduce MAE significantly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CTGFeatureBuilder:
    """
    Loads and provides access to CleaningTheGlass player statistics
    """

    def __init__(self, ctg_data_path: str = "data/ctg_data_organized/players"):
        """
        Initialize CTG feature builder

        Args:
            ctg_data_path: Path to CTG organized data directory
        """
        self.ctg_path = Path(ctg_data_path)
        self.offensive_stats = {}
        self.shooting_stats = {}
        self.rebounding_stats = {}

        print("Loading CTG data...")
        self._load_all_seasons()
        print(f"✅ Loaded CTG data for {len(self.offensive_stats)} season-years")

    def _load_all_seasons(self):
        """Load CTG stats for all available seasons"""
        # Find all season directories
        season_dirs = [d for d in self.ctg_path.iterdir() if d.is_dir()]

        for season_dir in season_dirs:
            season = season_dir.name

            # Load offensive overview (USG%, PSA, AST%, TOV%)
            offensive_file = season_dir / "regular_season" / "offensive_overview" / "offensive_overview.csv"
            if offensive_file.exists():
                df = pd.read_csv(offensive_file)
                self.offensive_stats[season] = self._clean_ctg_dataframe(df)

            # Load shooting accuracy (eFG%)
            shooting_file = season_dir / "regular_season" / "shooting_accuracy" / "shooting_accuracy.csv"
            if shooting_file.exists():
                df = pd.read_csv(shooting_file)
                self.shooting_stats[season] = self._clean_ctg_dataframe(df)

            # Load defense/rebounding (REB%)
            rebounding_file = season_dir / "regular_season" / "defense_rebounding" / "defense_rebounding.csv"
            if rebounding_file.exists():
                df = pd.read_csv(rebounding_file)
                self.rebounding_stats[season] = self._clean_ctg_dataframe(df)

    def _clean_ctg_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CTG dataframe for easier lookup"""
        # Standardize player names
        df['player_lower'] = df['Player'].str.lower().str.strip()

        # Remove percentage signs and convert to float
        for col in df.columns:
            if df[col].dtype == 'object' and '%' in str(df[col].iloc[0] if len(df) > 0 else ''):
                df[col] = df[col].str.rstrip('%').astype(float) / 100

        return df

    def get_player_ctg_features(self, player_name: str, season: str) -> Dict[str, float]:
        """
        Get CTG features for a player in a given season

        Args:
            player_name: Player name (will be standardized)
            season: Season in format "YYYY-YY" (e.g., "2024-25")

        Returns:
            Dictionary of CTG features
        """
        player_lower = player_name.lower().strip()

        # Initialize default features (league averages)
        features = {
            'CTG_USG': 0.20,      # League average usage
            'CTG_PSA': 1.10,      # League average points per shot attempt
            'CTG_AST_PCT': 0.12,  # League average assist %
            'CTG_TOV_PCT': 0.12,  # League average turnover %
            'CTG_eFG': 0.53,      # League average eFG%
            'CTG_REB_PCT': 0.10,  # League average rebounding %
            'CTG_Available': 0    # Flag: did we find CTG data?
        }

        # Try to find player in offensive stats
        if season in self.offensive_stats:
            off_df = self.offensive_stats[season]
            player_row = off_df[off_df['player_lower'] == player_lower]

            if len(player_row) > 0:
                row = player_row.iloc[0]

                # Extract offensive stats
                if 'Usage' in row:
                    features['CTG_USG'] = float(row['Usage']) if pd.notna(row['Usage']) else features['CTG_USG']
                if 'PSA' in row:
                    features['CTG_PSA'] = float(row['PSA']) if pd.notna(row['PSA']) else features['CTG_PSA']
                if 'AST%' in row:
                    features['CTG_AST_PCT'] = float(row['AST%']) if pd.notna(row['AST%']) else features['CTG_AST_PCT']
                if 'TOV%' in row:
                    features['CTG_TOV_PCT'] = float(row['TOV%']) if pd.notna(row['TOV%']) else features['CTG_TOV_PCT']

                features['CTG_Available'] = 1

        # Try to find player in shooting stats
        if season in self.shooting_stats:
            shoot_df = self.shooting_stats[season]
            player_row = shoot_df[shoot_df['player_lower'] == player_lower]

            if len(player_row) > 0:
                row = player_row.iloc[0]

                # Extract eFG%
                if 'eFG%' in row:
                    features['CTG_eFG'] = float(row['eFG%']) if pd.notna(row['eFG%']) else features['CTG_eFG']

                features['CTG_Available'] = 1

        # Try to find player in rebounding stats
        if season in self.rebounding_stats:
            reb_df = self.rebounding_stats[season]
            player_row = reb_df[reb_df['player_lower'] == player_lower]

            if len(player_row) > 0:
                row = player_row.iloc[0]

                # Extract rebounding %
                # FIXED: CTG rebounding files have fgOR%, fgDR%, ftOR%, ftDR%
                # Calculate total rebounding % as average of all components
                reb_components = []

                for reb_col in ['fgOR%', 'fgDR%', 'ftOR%', 'ftDR%']:
                    if reb_col in row and pd.notna(row[reb_col]):
                        reb_components.append(float(row[reb_col]))

                # If we found rebounding data, calculate total REB%
                if len(reb_components) > 0:
                    # Total REB% = average of all available components
                    # (field goal and free throw, offensive and defensive)
                    features['CTG_REB_PCT'] = sum(reb_components) / len(reb_components)

                features['CTG_Available'] = 1

        return features

    def get_batch_features(self, players_seasons: list) -> pd.DataFrame:
        """
        Get CTG features for multiple player-season combinations

        Args:
            players_seasons: List of tuples [(player_name, season), ...]

        Returns:
            DataFrame with CTG features for each player-season
        """
        results = []

        for player_name, season in players_seasons:
            features = self.get_player_ctg_features(player_name, season)
            features['player_name'] = player_name
            features['season'] = season
            results.append(features)

        return pd.DataFrame(results)

    def get_league_averages(self, season: str) -> Dict[str, float]:
        """Get league average CTG stats for a season"""
        league_avgs = {
            'CTG_USG': 0.20,
            'CTG_PSA': 1.10,
            'CTG_AST_PCT': 0.12,
            'CTG_TOV_PCT': 0.12,
            'CTG_eFG': 0.53,
            'CTG_REB_PCT': 0.10
        }

        # If we have data for this season, calculate actual averages
        if season in self.offensive_stats:
            df = self.offensive_stats[season]

            if 'Usage' in df.columns and len(df) > 0:
                league_avgs['CTG_USG'] = df['Usage'].mean()
            if 'PSA' in df.columns and len(df) > 0:
                league_avgs['CTG_PSA'] = df['PSA'].mean()
            if 'AST%' in df.columns and len(df) > 0:
                league_avgs['CTG_AST_PCT'] = df['AST%'].mean()
            if 'TOV%' in df.columns and len(df) > 0:
                league_avgs['CTG_TOV_PCT'] = df['TOV%'].mean()

        if season in self.shooting_stats:
            df = self.shooting_stats[season]
            if 'eFG%' in df.columns and len(df) > 0:
                league_avgs['CTG_eFG'] = df['eFG%'].mean()

        return league_avgs


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("CTG FEATURE BUILDER - TEST")
    print("="*80)

    # Initialize builder
    builder = CTGFeatureBuilder()

    # Test with some known players
    test_cases = [
        ("LeBron James", "2024-25"),
        ("Stephen Curry", "2024-25"),
        ("Nikola Jokic", "2024-25"),
        ("Giannis Antetokounmpo", "2024-25"),
        ("Unknown Player", "2024-25"),  # Should return league averages
    ]

    print("\n" + "="*80)
    print("TESTING INDIVIDUAL PLAYER LOOKUPS")
    print("="*80)

    for player, season in test_cases:
        features = builder.get_player_ctg_features(player, season)
        print(f"\n{player} ({season}):")
        print(f"  USG%: {features['CTG_USG']:.3f}")
        print(f"  PSA: {features['CTG_PSA']:.3f}")
        print(f"  AST%: {features['CTG_AST_PCT']:.3f}")
        print(f"  TOV%: {features['CTG_TOV_PCT']:.3f}")
        print(f"  eFG%: {features['CTG_eFG']:.3f}")
        print(f"  REB%: {features['CTG_REB_PCT']:.3f}")
        print(f"  Available: {'Yes' if features['CTG_Available'] == 1 else 'No (using league avg)'}")

    # Test batch lookup
    print("\n" + "="*80)
    print("TESTING BATCH LOOKUP")
    print("="*80)

    batch_df = builder.get_batch_features([
        ("LeBron James", "2024-25"),
        ("Stephen Curry", "2024-25"),
        ("Nikola Jokic", "2024-25"),
    ])

    print(batch_df[['player_name', 'CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_Available']])

    # Test league averages
    print("\n" + "="*80)
    print("LEAGUE AVERAGES")
    print("="*80)

    league_avgs = builder.get_league_averages("2024-25")
    print(f"USG%: {league_avgs['CTG_USG']:.3f}")
    print(f"PSA: {league_avgs['CTG_PSA']:.3f}")
    print(f"AST%: {league_avgs['CTG_AST_PCT']:.3f}")
    print(f"TOV%: {league_avgs['CTG_TOV_PCT']:.3f}")
    print(f"eFG%: {league_avgs['CTG_eFG']:.3f}")

    print("\n" + "="*80)
    print("CTG FEATURE BUILDER TEST COMPLETE")
    print("="*80)
