#!/usr/bin/env python3
"""
Feature Engineering from REAL NBA Game Logs
Creates comprehensive features for PRA prediction using actual game data
NO SYNTHETIC VALUES - ALL FROM REAL GAMES
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RealGameFeatureEngineer:
    """
    Feature engineering using real NBA game logs
    Three-tier architecture:
    1. Core Performance (Historical stats)
    2. Contextual Modulators (Game context)
    3. Temporal Dynamics (Recent form)
    """

    def __init__(self):
        self.data_dir = Path('/Users/diyagamah/Documents/nba_props_model/data')
        self.game_logs_dir = self.data_dir / 'game_logs'

        # Check data exists - NO FALLBACK
        if not self.game_logs_dir.exists():
            raise FileNotFoundError(f"Game logs not found: {self.game_logs_dir}")

        # Feature windows
        self.rolling_windows = [3, 5, 10, 15, 20]
        self.ewma_spans = [5, 10, 15]

    def load_season_data(self, season='2023-24'):
        """Load real game data for a season"""
        season_file = self.game_logs_dir / f'game_logs_{season}.csv'
        if not season_file.exists():
            raise FileNotFoundError(
                f"Season data not found: {season_file}\n"
                f"Run: uv run scripts/fetch_all_game_logs.py"
            )

        df = pd.read_csv(season_file)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE'])

        print(f"Loaded {len(df):,} games for {season}")
        print(f"Players: {df['PLAYER_NAME'].nunique()}")
        print(f"Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")

        return df

    def tier1_core_performance(self, df):
        """
        TIER 1: Core Performance Engine
        Historical performance metrics from real games
        """
        print("\n" + "="*60)
        print("TIER 1: CORE PERFORMANCE FEATURES")
        print("="*60)

        # === ROLLING AVERAGES ===
        print("\nCalculating rolling averages...")
        for window in self.rolling_windows:
            # Main stats
            df[f'PRA_L{window}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'PTS_L{window}'] = df.groupby('PLAYER_NAME')['PTS'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'REB_L{window}'] = df.groupby('PLAYER_NAME')['REB'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'AST_L{window}'] = df.groupby('PLAYER_NAME')['AST'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'MIN_L{window}'] = df.groupby('PLAYER_NAME')['MIN'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Advanced stats
            df[f'FG_PCT_L{window}'] = df.groupby('PLAYER_NAME')['FG_PCT'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'FG3_PCT_L{window}'] = df.groupby('PLAYER_NAME')['FG3_PCT'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # === EXPONENTIAL WEIGHTED AVERAGES ===
        print("Calculating EWMA features...")
        for span in self.ewma_spans:
            df[f'PRA_EMA{span}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
            )
            df[f'MIN_EMA{span}'] = df.groupby('PLAYER_NAME')['MIN'].transform(
                lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
            )

        # === USAGE METRICS (from real game data) ===
        print("Calculating usage metrics...")
        # True usage rate approximation
        df['usage_rate'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) /
                           (df['MIN'] / 5))  # Per 5 minutes on court

        df['usage_L5'] = df.groupby('PLAYER_NAME')['usage_rate'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df['usage_L10'] = df.groupby('PLAYER_NAME')['usage_rate'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )

        # === EFFICIENCY METRICS ===
        print("Calculating efficiency metrics...")
        # True Shooting Percentage
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['TS_PCT'] = df['TS_PCT'].replace([np.inf, -np.inf], np.nan)

        df['TS_L5'] = df.groupby('PLAYER_NAME')['TS_PCT'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Effective Field Goal Percentage
        df['eFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
        df['eFG_PCT'] = df['eFG_PCT'].replace([np.inf, -np.inf], np.nan)

        df['eFG_L5'] = df.groupby('PLAYER_NAME')['eFG_PCT'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Assist to Turnover Ratio
        df['AST_TO_ratio'] = df['AST'] / df['TOV'].replace(0, 1)
        df['AST_TO_L5'] = df.groupby('PLAYER_NAME')['AST_TO_ratio'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Rebound Rate (percentage of available rebounds)
        df['REB_rate'] = df['REB'] / df['MIN'] * 48  # Per 48 minutes
        df['REB_rate_L5'] = df.groupby('PLAYER_NAME')['REB_rate'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        print(f"✓ Created {len([c for c in df.columns if '_L' in c or '_EMA' in c])} core features")

        return df

    def tier2_contextual_modulators(self, df):
        """
        TIER 2: Contextual Modulators
        Game context and situational factors
        """
        print("\n" + "="*60)
        print("TIER 2: CONTEXTUAL MODULATOR FEATURES")
        print("="*60)

        # === REST DAYS ===
        print("\nCalculating rest days...")
        df['days_rest'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days
        df['days_rest'] = df['days_rest'].fillna(3).clip(upper=7)

        # Rest categories
        df['rest_category'] = pd.cut(
            df['days_rest'],
            bins=[-1, 0.5, 1.5, 2.5, 100],
            labels=['b2b', '1_day', '2_days', '3+_days']
        )

        # === HOME/AWAY ===
        print("Extracting home/away...")
        df['is_home'] = df['MATCHUP'].str.contains('vs').astype(int)

        # Home/away performance history
        df['PRA_home_L10'] = df[df['is_home'] == 1].groupby('PLAYER_NAME')['PRA'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )
        df['PRA_away_L10'] = df[df['is_home'] == 0].groupby('PLAYER_NAME')['PRA'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean()
        )

        # Fill NaN with overall average
        df['PRA_home_L10'] = df['PRA_home_L10'].fillna(df['PRA_L10'])
        df['PRA_away_L10'] = df['PRA_away_L10'].fillna(df['PRA_L10'])

        # === OPPONENT QUALITY ===
        print("Calculating opponent metrics...")
        # Calculate team defensive rating (PRA allowed)
        team_def_rating = df.groupby(['TEAM_NAME', 'GAME_DATE'])['PRA'].mean().reset_index()
        team_def_rating.columns = ['OPP_TEAM', 'GAME_DATE', 'OPP_PRA_allowed']

        # Extract opponent from MATCHUP
        df['OPP_TEAM'] = df['MATCHUP'].str.extract(r'(?:vs|@)\s+(.+)')[0]

        # === GAME PACE ===
        # Estimate pace from total possessions
        df['possessions'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        df['pace_proxy'] = df.groupby('GAME_ID')['possessions'].transform('sum')

        # === BLOWOUT INDICATOR ===
        df['is_blowout'] = (np.abs(df['PLUS_MINUS']) > 20).astype(int)

        # === SEASON PHASE ===
        print("Adding season phase features...")
        season_start = df['GAME_DATE'].min()
        season_end = df['GAME_DATE'].max()
        season_length = (season_end - season_start).days

        df['days_into_season'] = (df['GAME_DATE'] - season_start).dt.days
        df['season_progress'] = df['days_into_season'] / season_length

        # Season phase categories
        df['season_phase'] = pd.cut(
            df['season_progress'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['early', 'mid_early', 'mid_late', 'late']
        )

        # === GAMES PLAYED ===
        df['games_played'] = df.groupby('PLAYER_NAME').cumcount() + 1
        df['games_played_pct'] = df['games_played'] / df.groupby('PLAYER_NAME')['games_played'].transform('max')

        print(f"✓ Created contextual features")

        return df

    def tier3_temporal_dynamics(self, df):
        """
        TIER 3: Temporal Dynamics
        Trends, momentum, and volatility
        """
        print("\n" + "="*60)
        print("TIER 3: TEMPORAL DYNAMICS FEATURES")
        print("="*60)

        # === PERFORMANCE TRENDS ===
        print("\nCalculating performance trends...")
        for window in [5, 10]:
            # Linear trend (slope of regression line)
            df[f'PRA_trend_{window}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
                    raw=False
                )
            )

            df[f'MIN_trend_{window}'] = df.groupby('PLAYER_NAME')['MIN'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
                    raw=False
                )
            )

        # === VOLATILITY MEASURES ===
        print("Calculating volatility measures...")
        for window in [5, 10, 15]:
            df[f'PRA_std_{window}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
            )
            df[f'MIN_std_{window}'] = df.groupby('PLAYER_NAME')['MIN'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=2).std()
            )

            # Coefficient of variation
            df[f'PRA_cv_{window}'] = df[f'PRA_std_{window}'] / df[f'PRA_L{window}']
            df[f'PRA_cv_{window}'] = df[f'PRA_cv_{window}'].replace([np.inf, -np.inf], np.nan)

        # === MOMENTUM INDICATORS ===
        print("Calculating momentum indicators...")
        # Recent vs longer-term performance
        df['PRA_momentum_5v15'] = (df['PRA_L5'] - df['PRA_L15']) / df['PRA_L15']
        df['PRA_momentum_5v15'] = df['PRA_momentum_5v15'].replace([np.inf, -np.inf], np.nan)

        # Hot/cold streaks
        df['above_avg'] = df.groupby('PLAYER_NAME').apply(
            lambda x: (x['PRA'] > x['PRA'].shift(1).rolling(20, min_periods=5).mean()).astype(int)
        ).reset_index(level=0, drop=True)

        df['hot_streak'] = df.groupby('PLAYER_NAME')['above_avg'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        )

        # === CONSISTENCY SCORES ===
        print("Calculating consistency scores...")
        # How often player meets their average
        df['meets_avg_L10'] = df.groupby('PLAYER_NAME').apply(
            lambda x: (x['PRA'] >= x['PRA_L10'] * 0.8).astype(int)
        ).reset_index(level=0, drop=True)

        df['consistency_score'] = df.groupby('PLAYER_NAME')['meets_avg_L10'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )

        # === RELATIVE PERFORMANCE ===
        # Performance relative to season average
        df['season_avg'] = df.groupby('PLAYER_NAME')['PRA'].transform('mean')
        df['PRA_vs_season'] = df['PRA'] / df['season_avg']

        df['PRA_vs_season_L5'] = df.groupby('PLAYER_NAME')['PRA_vs_season'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        print(f"✓ Created temporal dynamics features")

        return df

    def create_target_variable(self, df):
        """Create target variable (next game PRA)"""
        print("\n" + "="*60)
        print("CREATING TARGET VARIABLE")
        print("="*60)

        # Next game PRA
        df['target_PRA'] = df.groupby('PLAYER_NAME')['PRA'].shift(-1)

        # Also create component targets
        df['target_PTS'] = df.groupby('PLAYER_NAME')['PTS'].shift(-1)
        df['target_REB'] = df.groupby('PLAYER_NAME')['REB'].shift(-1)
        df['target_AST'] = df.groupby('PLAYER_NAME')['AST'].shift(-1)

        # Binary targets for classification
        for threshold in [20, 25, 30, 35, 40]:
            df[f'target_PRA_over_{threshold}'] = (df['target_PRA'] > threshold).astype(int)

        print(f"✓ Created target variables")

        return df

    def get_feature_columns(self, df):
        """Get list of feature columns"""
        # Exclude these columns
        exclude_cols = [
            'SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
            'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
            'PTS', 'PLUS_MINUS', 'PRA', 'DOUBLE_DOUBLE', 'TRIPLE_DOUBLE',
            'DK_POINTS', 'FD_POINTS', 'SEASON', 'SEASON_TYPE',
            'OPP_TEAM', 'OPP_PRA_allowed', 'above_avg', 'meets_avg_L10',
            'season_avg'
        ]

        # Also exclude targets
        exclude_cols.extend([col for col in df.columns if col.startswith('target_')])

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Filter to numeric columns only
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64', 'int32', 'float32']:
                numeric_features.append(col)

        return numeric_features

    def create_training_dataset(self, season='2023-24'):
        """Create complete training dataset"""
        print("\n" + "="*60)
        print(f"CREATING TRAINING DATASET FOR {season}")
        print("="*60)

        # Load data
        df = self.load_season_data(season)

        # Apply all feature engineering tiers
        df = self.tier1_core_performance(df)
        df = self.tier2_contextual_modulators(df)
        df = self.tier3_temporal_dynamics(df)
        df = self.create_target_variable(df)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)

        print(f"\nTotal features created: {len(feature_cols)}")

        # Filter to players with enough games
        min_games = 20
        player_games = df.groupby('PLAYER_NAME').size()
        active_players = player_games[player_games >= min_games].index
        df_filtered = df[df['PLAYER_NAME'].isin(active_players)]

        print(f"Filtered to {len(active_players)} players with {min_games}+ games")

        # Create final dataset
        final_cols = ['PLAYER_NAME', 'GAME_DATE'] + feature_cols + ['target_PRA']
        training_df = df_filtered[final_cols].dropna(subset=['target_PRA'])

        # Save dataset
        output_path = self.data_dir / 'processed' / f'training_data_real_{season}.csv'
        output_path.parent.mkdir(exist_ok=True)
        training_df.to_csv(output_path, index=False)

        print(f"\n✓ Training dataset saved to: {output_path}")
        print(f"  Samples: {len(training_df):,}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: target_PRA (next game)")

        # Show feature categories
        print("\nFeature Categories:")
        print(f"  Rolling averages: {len([c for c in feature_cols if '_L' in c])}")
        print(f"  EWMA features: {len([c for c in feature_cols if '_EMA' in c])}")
        print(f"  Trend features: {len([c for c in feature_cols if '_trend' in c])}")
        print(f"  Volatility features: {len([c for c in feature_cols if '_std' in c or '_cv' in c])}")
        print(f"  Context features: {len([c for c in feature_cols if any(x in c for x in ['rest', 'home', 'away', 'season'])])}")

        return training_df


def main():
    """Main execution"""
    print("\n" + "="*70)
    print(" FEATURE ENGINEERING FROM REAL NBA GAME LOGS ".center(70))
    print("="*70)

    try:
        # Create feature engineer
        engineer = RealGameFeatureEngineer()

        # Create training dataset
        df = engineer.create_training_dataset('2023-24')

        print("\n" + "="*70)
        print(" FEATURE ENGINEERING COMPLETE ".center(70))
        print("="*70)

        print("\n✅ Next steps:")
        print("1. Run model evaluation: uv run scripts/evaluate_models_real_data.py")
        print("2. The training data is in: data/processed/training_data_real_2023-24.csv")
        print("3. Use this for REAL predictions, not synthetic estimates")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()