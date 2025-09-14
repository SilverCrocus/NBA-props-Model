#!/usr/bin/env python3
"""
Analyze fetched NBA game logs and demonstrate real PRA prediction
Shows the difference between real game data and synthetic estimates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GameLogAnalyzer:
    """Analyze real NBA game logs for PRA prediction"""

    def __init__(self, data_dir: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_dir = Path(data_dir)
        self.game_logs_dir = self.data_dir / "game_logs"

    def load_game_logs(self, season: str = None) -> pd.DataFrame:
        """Load game logs for analysis"""
        if season:
            file_path = self.game_logs_dir / f"game_logs_{season}.csv"
            if not file_path.exists():
                print(f"Error: {file_path} not found. Run fetch_all_game_logs.py first.")
                return None
            return pd.read_csv(file_path)
        else:
            # Load combined file
            file_path = self.game_logs_dir / "all_game_logs_combined.csv"
            if not file_path.exists():
                print(f"Error: {file_path} not found. Run fetch_all_game_logs.py first.")
                return None
            return pd.read_csv(file_path)

    def compare_with_synthetic(self):
        """Compare real game data with synthetic PRA estimates"""
        print("\n" + "=" * 60)
        print("REAL vs SYNTHETIC PRA COMPARISON")
        print("=" * 60)

        # Load synthetic data
        synthetic_path = self.data_dir / "processed" / "player_features_2023_24.csv"
        if synthetic_path.exists():
            synthetic_df = pd.read_csv(synthetic_path)
            print(f"\nSynthetic PRA_estimate Statistics:")
            print(f"  Mean: {synthetic_df['PRA_estimate'].mean():.2f}")
            print(f"  Std: {synthetic_df['PRA_estimate'].std():.2f}")
            print(f"  Range: [{synthetic_df['PRA_estimate'].min():.2f}, "
                  f"{synthetic_df['PRA_estimate'].max():.2f}]")

        # Load real game data for 2023-24
        real_df = self.load_game_logs('2023-24')
        if real_df is not None:
            print(f"\nReal Game PRA Statistics (2023-24):")
            print(f"  Mean: {real_df['PRA'].mean():.2f}")
            print(f"  Std: {real_df['PRA'].std():.2f}")
            print(f"  Range: [{real_df['PRA'].min():.2f}, {real_df['PRA'].max():.2f}]")

            # Per-player averages
            player_avg = real_df.groupby('PLAYER_NAME')['PRA'].agg(['mean', 'std', 'count'])
            player_avg = player_avg[player_avg['count'] >= 20]  # Min 20 games

            print(f"\nPer-Player Statistics (min 20 games):")
            print(f"  Players: {len(player_avg)}")
            print(f"  Avg PRA Mean: {player_avg['mean'].mean():.2f}")
            print(f"  Avg PRA Std (within-player): {player_avg['std'].mean():.2f}")

            print("\n⚠️ KEY INSIGHT:")
            print("  Synthetic data is a SEASON AVERAGE")
            print("  Real data has GAME-BY-GAME VARIATION")
            print("  Real std dev is much higher due to game-to-game variance!")

    def calculate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling features for real prediction"""
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE'])

        # Calculate rolling averages
        for window in [3, 5, 10]:
            df[f'PRA_L{window}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Calculate trend
        df['PRA_trend'] = df.groupby('PLAYER_NAME')['PRA'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            )
        )

        # Days rest
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['days_rest'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days.fillna(3)

        return df

    def analyze_variance_components(self):
        """Analyze sources of variance in PRA"""
        print("\n" + "=" * 60)
        print("VARIANCE COMPONENTS ANALYSIS")
        print("=" * 60)

        df = self.load_game_logs('2023-24')
        if df is None:
            return

        # Filter to players with enough games
        player_games = df.groupby('PLAYER_NAME').size()
        regular_players = player_games[player_games >= 30].index
        df_regular = df[df['PLAYER_NAME'].isin(regular_players)]

        # Calculate variance components
        total_variance = df_regular['PRA'].var()
        between_player_var = df_regular.groupby('PLAYER_NAME')['PRA'].mean().var()
        within_player_var = df_regular.groupby('PLAYER_NAME')['PRA'].var().mean()

        print(f"\nTotal PRA Variance: {total_variance:.2f}")
        print(f"Between-Player Variance: {between_player_var:.2f} ({between_player_var/total_variance*100:.1f}%)")
        print(f"Within-Player Variance: {within_player_var:.2f} ({within_player_var/total_variance*100:.1f}%)")

        print("\n💡 INSIGHT: Large within-player variance means:")
        print("   - Single season averages are poor predictors")
        print("   - Recent form matters significantly")
        print("   - Context (opponent, rest, etc.) is crucial")

    def demonstrate_realistic_prediction(self):
        """Show realistic PRA prediction with real data"""
        print("\n" + "=" * 60)
        print("REALISTIC PRA PREDICTION DEMONSTRATION")
        print("=" * 60)

        df = self.load_game_logs('2023-24')
        if df is None:
            return

        # Add rolling features
        df = self.calculate_rolling_features(df)

        # Simple prediction: use last 5 games average
        df = df.dropna(subset=['PRA_L5'])

        from sklearn.metrics import mean_absolute_error, r2_score

        # Predict next game using L5 average
        mae = mean_absolute_error(df['PRA'], df['PRA_L5'])
        r2 = r2_score(df['PRA'], df['PRA_L5'])

        print(f"\nSimple Model (Last 5 Games Average):")
        print(f"  MAE: {mae:.2f} points")
        print(f"  R²: {r2:.3f}")
        print(f"  MAPE: {(np.abs(df['PRA'] - df['PRA_L5']) / df['PRA']).mean() * 100:.1f}%")

        # Compare with perfect season average prediction
        season_avg = df.groupby('PLAYER_NAME')['PRA'].transform('mean')
        mae_season = mean_absolute_error(df['PRA'], season_avg)
        r2_season = r2_score(df['PRA'], season_avg)

        print(f"\nSeason Average Model:")
        print(f"  MAE: {mae_season:.2f} points")
        print(f"  R²: {r2_season:.3f}")

        print("\n✅ REALISTIC EXPECTATIONS:")
        print(f"  MAE: 4-6 points")
        print(f"  R²: 0.35-0.50")
        print(f"  MAPE: 25-35%")

    def create_training_dataset(self, season: str = '2023-24'):
        """Create a proper training dataset from game logs"""
        print("\n" + "=" * 60)
        print("CREATING TRAINING DATASET")
        print("=" * 60)

        df = self.load_game_logs(season)
        if df is None:
            return None

        # Add features
        df = self.calculate_rolling_features(df)

        # Create features and target
        feature_cols = [
            'PRA_L3', 'PRA_L5', 'PRA_L10',
            'PRA_trend', 'days_rest', 'MIN'
        ]

        # Create lagged dataset
        df['target_PRA'] = df.groupby('PLAYER_NAME')['PRA'].shift(-1)

        # Remove rows with missing values
        training_df = df[feature_cols + ['target_PRA', 'PLAYER_NAME', 'GAME_DATE']].dropna()

        # Save training dataset
        output_path = self.data_dir / "processed" / f"game_training_data_{season}.csv"
        output_path.parent.mkdir(exist_ok=True)
        training_df.to_csv(output_path, index=False)

        print(f"\n✓ Training dataset created:")
        print(f"  Samples: {len(training_df)}")
        print(f"  Features: {feature_cols}")
        print(f"  Target: Next game PRA")
        print(f"  Saved to: {output_path}")

        return training_df


def main():
    """Main analysis function"""
    analyzer = GameLogAnalyzer()

    print("\n" + "=" * 60)
    print("NBA GAME LOGS ANALYSIS")
    print("=" * 60)

    # Check if data exists
    game_logs_dir = Path("/Users/diyagamah/Documents/nba_props_model/data/game_logs")
    if not game_logs_dir.exists() or not list(game_logs_dir.glob("*.csv")):
        print("\n⚠️ No game logs found!")
        print("Please run: uv run scripts/fetch_all_game_logs.py")
        print("This will fetch real game data from the NBA API")
        return

    # Run analyses
    analyzer.compare_with_synthetic()
    analyzer.analyze_variance_components()
    analyzer.demonstrate_realistic_prediction()
    analyzer.create_training_dataset()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Use the training dataset for model development")
    print("2. Add contextual features (opponent, home/away)")
    print("3. Implement proper time-series validation")
    print("4. Test on future games (not in training)")


if __name__ == "__main__":
    main()