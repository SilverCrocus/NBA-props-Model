"""
Data Preprocessing Module for NBA Player Props Model
Handles data loading, merging, cleaning, and preparation for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import glob
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBADataPreprocessor:
    """
    Comprehensive data preprocessing for NBA player props prediction
    """
    
    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.ctg_player_path = self.data_path / "ctg_data_organized" / "players"
        self.ctg_team_path = self.data_path / "ctg_team_data"
        
        # Data type mappings for CTG stats
        self.stat_categories = {
            'offensive': ['offensive_overview', 'shooting_frequency', 'shooting_accuracy', 'shooting_overall'],
            'defensive': ['defense_rebounding', 'foul_drawing'],
            'on_off': ['efficiency_four_factors', 'team_shooting_frequency', 'opponent_shooting_frequency'],
            'context': ['team_transition', 'opponent_transition', 'team_halfcourt_putbacks']
        }
        
    def load_season_data(self, season: str, data_type: str = 'regular_season') -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific season
        
        Args:
            season: Season string (e.g., '2023-24')
            data_type: 'regular_season' or 'playoffs'
        
        Returns:
            Dictionary of dataframes by category
        """
        season_path = self.ctg_player_path / season / data_type
        data_dict = {}
        
        if not season_path.exists():
            logger.warning(f"Season path does not exist: {season_path}")
            return data_dict
        
        # Load offensive overview (main file)
        offensive_path = season_path / 'offensive_overview' / 'offensive_overview.csv'
        if offensive_path.exists():
            data_dict['offensive_overview'] = pd.read_csv(offensive_path)
            logger.info(f"Loaded offensive overview: {len(data_dict['offensive_overview'])} rows")
        
        # Load defense/rebounding
        defense_path = season_path / 'defense_rebounding' / 'defense_rebounding.csv'
        if defense_path.exists():
            data_dict['defense_rebounding'] = pd.read_csv(defense_path)
            logger.info(f"Loaded defense/rebounding: {len(data_dict['defense_rebounding'])} rows")
        
        # Load shooting data
        shooting_freq_path = season_path / 'shooting_frequency' / 'shooting_frequency.csv'
        if shooting_freq_path.exists():
            data_dict['shooting_frequency'] = pd.read_csv(shooting_freq_path)
        
        shooting_acc_path = season_path / 'shooting_accuracy' / 'shooting_accuracy.csv'
        if shooting_acc_path.exists():
            data_dict['shooting_accuracy'] = pd.read_csv(shooting_acc_path)
        
        # Load on/off data
        on_off_path = season_path / 'on_off'
        if on_off_path.exists():
            for subdir in on_off_path.iterdir():
                if subdir.is_dir():
                    csv_files = list(subdir.glob('*.csv'))
                    if csv_files:
                        key = f"on_off_{subdir.name}"
                        data_dict[key] = pd.read_csv(csv_files[0])
        
        return data_dict
    
    def merge_player_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple data sources for players
        """
        if not data_dict:
            return pd.DataFrame()
        
        # Start with offensive overview as base
        if 'offensive_overview' in data_dict:
            merged = data_dict['offensive_overview'].copy()
        else:
            # Use first available dataframe as base
            merged = list(data_dict.values())[0].copy()
        
        # Key columns for merging
        merge_keys = ['Player', 'Team', 'Pos']
        
        # Merge other dataframes
        for key, df in data_dict.items():
            if key == 'offensive_overview':
                continue
            
            # Identify columns to add (exclude duplicates)
            cols_to_add = [col for col in df.columns if col not in merged.columns or col in merge_keys]
            
            if len(cols_to_add) > len(merge_keys):
                try:
                    # Handle players who played for multiple teams
                    merged = merged.merge(
                        df[cols_to_add],
                        on=merge_keys,
                        how='left',
                        suffixes=('', f'_{key}')
                    )
                except:
                    # Try merging on Player only if Team causes issues
                    merged = merged.merge(
                        df[cols_to_add],
                        on='Player',
                        how='left',
                        suffixes=('', f'_{key}')
                    )
        
        return merged
    
    def load_team_data(self, teams: List[str] = None) -> pd.DataFrame:
        """
        Load and consolidate team data
        """
        team_data_list = []
        
        if teams is None:
            # Load all teams
            team_dirs = [d for d in self.ctg_team_path.iterdir() if d.is_dir()]
        else:
            team_dirs = [self.ctg_team_path / team for team in teams]
        
        for team_dir in team_dirs:
            if not team_dir.exists():
                continue
            
            team_name = team_dir.name
            
            # Load team efficiency data
            efficiency_file = team_dir / 'team_efficiency_and_four_factors_all_seasons.csv'
            if efficiency_file.exists():
                df = pd.read_csv(efficiency_file)
                df['team_name'] = team_name
                team_data_list.append(df)
        
        if team_data_list:
            team_data = pd.concat(team_data_list, ignore_index=True)
            return team_data
        
        return pd.DataFrame()
    
    def clean_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean percentage columns (remove % sign and convert to float)
        """
        df = df.copy()
        
        for col in df.columns:
            if '%' in col or df[col].dtype == 'object':
                try:
                    # Check if column contains percentage values
                    if df[col].astype(str).str.contains('%').any():
                        df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100.0
                except:
                    pass
        
        return df
    
    def handle_multi_team_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle players who played for multiple teams in a season
        """
        df = df.copy()
        
        # Identify multi-team players
        player_teams = df.groupby('Player')['Team'].nunique()
        multi_team_players = player_teams[player_teams > 1].index
        
        if len(multi_team_players) > 0:
            logger.info(f"Found {len(multi_team_players)} players who played for multiple teams")
            
            # For each multi-team player, weight stats by minutes played
            for player in multi_team_players:
                player_data = df[df['Player'] == player]
                
                if 'MIN' in df.columns:
                    # Calculate weighted averages based on minutes
                    total_minutes = player_data['MIN'].sum()
                    
                    # Identify numeric columns for averaging
                    numeric_cols = player_data.select_dtypes(include=[np.number]).columns
                    numeric_cols = [col for col in numeric_cols if col != 'MIN']
                    
                    # Calculate weighted averages
                    weighted_stats = {}
                    for col in numeric_cols:
                        weighted_stats[col] = (player_data[col] * player_data['MIN']).sum() / total_minutes
                    
                    # Create combined row
                    combined_row = player_data.iloc[0].copy()
                    combined_row['Team'] = 'TOT'  # Total for all teams
                    combined_row['MIN'] = total_minutes
                    
                    for col, value in weighted_stats.items():
                        combined_row[col] = value
                    
                    # Add combined row to dataframe
                    df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
        
        return df
    
    def create_game_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game-level features from season aggregates
        """
        df = df.copy()
        
        # Estimate games played from minutes (assuming ~30 min per game average)
        if 'MIN' in df.columns:
            df['estimated_games'] = df['MIN'] / 30
            df['minutes_per_game'] = df['MIN'] / df['estimated_games']
        
        # Create per-game estimates for counting stats if available
        if 'Usage' in df.columns and 'MIN' in df.columns:
            # Usage is already a percentage, no conversion needed
            pass
        
        # Create position groups
        if 'Pos' in df.columns:
            df['position_group'] = df['Pos'].map({
                'Guard': 'Guard',
                'Wing': 'Wing', 
                'Big': 'Big',
                'Forward': 'Wing',  # Map forwards to wings
                'Center': 'Big'      # Map centers to bigs
            }).fillna('Wing')  # Default to wing if unknown
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, 
                           box_scores: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add PRA (Points + Rebounds + Assists) target variable
        """
        if box_scores is not None and not box_scores.empty:
            # If we have actual box scores, calculate PRA
            if all(col in box_scores.columns for col in ['PTS', 'REB', 'AST']):
                box_scores['PRA'] = box_scores['PTS'] + box_scores['REB'] + box_scores['AST']
                
                # Merge with features
                df = df.merge(
                    box_scores[['Player', 'game_date', 'PRA']],
                    on=['Player', 'game_date'],
                    how='left'
                )
        else:
            # Create estimated PRA from available stats
            logger.warning("No box scores available, creating estimated PRA from CTG stats")
            
            # Estimate based on usage and efficiency
            if 'Usage' in df.columns and 'PSA' in df.columns:
                # Rough estimation: higher usage and efficiency = higher PRA
                df['PRA_estimate'] = (
                    df['Usage'] * 100 +  # Scale usage
                    df['PSA'] / 5 +       # Scale PSA
                    df.get('AST%', 0) * 50 +  # Scale assist percentage
                    df.get('fgOR%', 0) * 20 + df.get('fgDR%', 0) * 30  # Rebounding
                )
            else:
                df['PRA_estimate'] = np.nan
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split preserving temporal order
        """
        # Sort by date if available
        if 'game_date' in df.columns:
            df = df.sort_values('game_date')
            
            # Use last 20% of games as test set
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            # Random split if no temporal data
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality validation
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'unique_players': 0,
            'unique_teams': 0,
            'numeric_columns': [],
            'categorical_columns': [],
            'potential_issues': []
        }
        
        # Basic stats
        if 'Player' in df.columns:
            report['unique_players'] = df['Player'].nunique()
        
        if 'Team' in df.columns:
            report['unique_teams'] = df['Team'].nunique()
        
        # Check missing values
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 0:
                report['missing_values'][col] = round(missing_pct, 2)
        
        # Categorize columns
        for col in df.columns:
            dtype = str(df[col].dtype)
            report['data_types'][col] = dtype
            
            if df[col].dtype in ['int64', 'float64']:
                report['numeric_columns'].append(col)
            else:
                report['categorical_columns'].append(col)
        
        # Check for potential issues
        
        # Check for duplicate players
        if 'Player' in df.columns:
            duplicates = df.groupby('Player').size()
            if (duplicates > 1).any():
                report['potential_issues'].append(
                    f"Found {(duplicates > 1).sum()} players with multiple entries"
                )
        
        # Check for negative values in percentage columns
        for col in df.columns:
            if '%' in col and df[col].dtype in ['int64', 'float64']:
                if (df[col] < 0).any():
                    report['potential_issues'].append(f"Negative values in {col}")
                if (df[col] > 100).any():
                    report['potential_issues'].append(f"Values > 100 in percentage column {col}")
        
        # Check for outliers in key stats
        if 'MIN' in df.columns:
            max_min = df['MIN'].max()
            if max_min > 4000:  # More than ~50 games * 48 minutes
                report['potential_issues'].append(f"Unusually high minutes: {max_min}")
        
        return report
    
    def prepare_modeling_data(self, season: str = '2023-24',
                            include_playoffs: bool = False) -> pd.DataFrame:
        """
        Main method to prepare data for modeling
        """
        logger.info(f"Preparing data for season {season}")
        
        # Load regular season data
        regular_data = self.load_season_data(season, 'regular_season')
        merged_data = self.merge_player_data(regular_data)
        
        # Include playoffs if requested
        if include_playoffs:
            playoff_data = self.load_season_data(season, 'playoffs')
            if playoff_data:
                playoff_merged = self.merge_player_data(playoff_data)
                playoff_merged['is_playoff'] = 1
                merged_data['is_playoff'] = 0
                merged_data = pd.concat([merged_data, playoff_merged], ignore_index=True)
        
        # Clean data
        merged_data = self.clean_percentage_columns(merged_data)
        merged_data = self.handle_multi_team_players(merged_data)
        merged_data = self.create_game_level_features(merged_data)
        
        # Add target variable (if possible)
        merged_data = self.add_target_variable(merged_data)
        
        # Validate data quality
        quality_report = self.validate_data_quality(merged_data)
        
        logger.info(f"Data preparation complete: {quality_report['total_rows']} rows, "
                   f"{quality_report['total_columns']} columns")
        
        if quality_report['potential_issues']:
            logger.warning(f"Potential issues found: {quality_report['potential_issues']}")
        
        return merged_data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_multiple_seasons(seasons: List[str], 
                         preprocessor: NBADataPreprocessor) -> pd.DataFrame:
    """
    Load and combine multiple seasons of data
    """
    all_seasons_data = []
    
    for season in seasons:
        logger.info(f"Loading season {season}")
        season_data = preprocessor.prepare_modeling_data(season)
        season_data['season'] = season
        all_seasons_data.append(season_data)
    
    combined_data = pd.concat(all_seasons_data, ignore_index=True)
    logger.info(f"Loaded {len(seasons)} seasons with {len(combined_data)} total rows")
    
    return combined_data


def create_player_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create player-level aggregates across seasons
    """
    player_stats = df.groupby('Player').agg({
        'MIN': 'sum',
        'Usage': 'mean',
        'PSA': 'mean',
        'AST%': 'mean',
        'TOV%': 'mean',
        'fgOR%': 'mean',
        'fgDR%': 'mean'
    }).round(2)
    
    player_stats['total_seasons'] = df.groupby('Player')['season'].nunique()
    player_stats = player_stats.sort_values('MIN', ascending=False)
    
    return player_stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """
    Example usage of the data preprocessing pipeline
    """
    # Initialize preprocessor
    preprocessor = NBADataPreprocessor()
    
    # Load and prepare data for 2023-24 season
    data = preprocessor.prepare_modeling_data(season='2023-24', include_playoffs=False)
    
    # Display basic information
    print("\n=== Data Summary ===")
    print(f"Shape: {data.shape}")
    print(f"\nColumns: {list(data.columns)[:20]}...")  # Show first 20 columns
    
    # Show sample of data
    print("\n=== Sample Data ===")
    print(data.head())
    
    # Generate quality report
    quality_report = preprocessor.validate_data_quality(data)
    
    print("\n=== Data Quality Report ===")
    print(f"Total Rows: {quality_report['total_rows']}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Unique Players: {quality_report['unique_players']}")
    print(f"Unique Teams: {quality_report['unique_teams']}")
    
    if quality_report['missing_values']:
        print("\nColumns with missing values:")
        for col, pct in list(quality_report['missing_values'].items())[:10]:
            print(f"  - {col}: {pct}%")
    
    if quality_report['potential_issues']:
        print("\nPotential Issues:")
        for issue in quality_report['potential_issues']:
            print(f"  - {issue}")
    
    # Save prepared data
    output_path = "/Users/diyagamah/Documents/nba_props_model/prepared_data_2023_24.csv"
    data.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    # Load multiple seasons example
    print("\n=== Loading Multiple Seasons ===")
    seasons = ['2021-22', '2022-23', '2023-24']
    multi_season_data = load_multiple_seasons(seasons, preprocessor)
    print(f"Combined data shape: {multi_season_data.shape}")
    
    # Create player aggregates
    player_aggregates = create_player_aggregates(multi_season_data)
    print("\n=== Top 10 Players by Total Minutes ===")
    print(player_aggregates.head(10))


if __name__ == "__main__":
    main()