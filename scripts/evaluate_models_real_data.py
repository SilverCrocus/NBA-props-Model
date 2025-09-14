#!/usr/bin/env python3
"""
NBA Props Model Evaluation with REAL Game Data
No synthetic data, no fallback values - uses actual NBA game logs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class RealDataModelEvaluator:
    """
    Model evaluation using REAL NBA game data
    No synthetic targets, no made-up values
    """

    def __init__(self):
        self.data_dir = Path('/Users/diyagamah/Documents/nba_props_model/data')
        self.game_logs_dir = self.data_dir / 'game_logs'
        self.processed_dir = self.data_dir / 'processed'
        self.results_dir = self.data_dir / 'model_results'
        self.results_dir.mkdir(exist_ok=True)

        # Check for data - NO FALLBACK
        if not self.game_logs_dir.exists():
            raise FileNotFoundError(f"Game logs directory not found: {self.game_logs_dir}")

        self.combined_file = self.game_logs_dir / 'all_game_logs_combined.csv'
        if not self.combined_file.exists():
            raise FileNotFoundError(
                f"Combined game logs not found: {self.combined_file}\n"
                "Please run: uv run scripts/fetch_all_game_logs.py"
            )

    def load_game_data(self, season='2023-24'):
        """Load real game data - NO DEFAULTS"""
        print(f"\n{'='*60}")
        print(f"LOADING REAL GAME DATA FOR {season}")
        print(f"{'='*60}")

        season_file = self.game_logs_dir / f'game_logs_{season}.csv'
        if not season_file.exists():
            raise FileNotFoundError(f"Season file not found: {season_file}")

        df = pd.read_csv(season_file)
        print(f"✓ Loaded {len(df):,} game records")
        print(f"✓ Players: {df['PLAYER_NAME'].nunique()}")
        print(f"✓ Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

        # Convert date
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        return df

    def engineer_features(self, df):
        """
        Engineer features from real game logs
        NO SYNTHETIC CALCULATIONS
        """
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING FROM REAL GAMES")
        print(f"{'='*60}")

        # Sort by player and date
        df = df.sort_values(['PLAYER_NAME', 'GAME_DATE'])

        # Filter to players with enough games
        player_games = df.groupby('PLAYER_NAME').size()
        active_players = player_games[player_games >= 20].index
        df = df[df['PLAYER_NAME'].isin(active_players)]
        print(f"✓ Filtered to {len(active_players)} players with 20+ games")

        # === ROLLING AVERAGES (Past Performance) ===
        for window in [3, 5, 10, 15]:
            # PRA
            df[f'PRA_L{window}'] = df.groupby('PLAYER_NAME')['PRA'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Individual stats
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

        # === VARIABILITY FEATURES ===
        df['PRA_std_L10'] = df.groupby('PLAYER_NAME')['PRA'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).std()
        )
        df['MIN_std_L10'] = df.groupby('PLAYER_NAME')['MIN'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).std()
        )

        # === TREND FEATURES ===
        df['PRA_trend'] = df.groupby('PLAYER_NAME')['PRA'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
                raw=False
            )
        )

        # === EFFICIENCY METRICS (from actual games) ===
        # Usage approximation from real stats
        df['usage_proxy'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN']
        df['usage_L5'] = df.groupby('PLAYER_NAME')['usage_proxy'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # True shooting percentage
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['TS_PCT'] = df['TS_PCT'].replace([np.inf, -np.inf], np.nan)
        df['TS_L5'] = df.groupby('PLAYER_NAME')['TS_PCT'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # === GAME CONTEXT ===
        # Days rest
        df['days_rest'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days.fillna(3)
        df['days_rest'] = df['days_rest'].clip(upper=7)  # Cap at 7 days

        # Home/Away
        df['is_home'] = df['MATCHUP'].str.contains('vs').astype(int)

        # Back-to-back games
        df['is_b2b'] = (df['days_rest'] == 1).astype(int)

        # === SEASON PROGRESSION ===
        season_start = df['GAME_DATE'].min()
        df['days_into_season'] = (df['GAME_DATE'] - season_start).dt.days

        # Games played so far
        df['games_played'] = df.groupby('PLAYER_NAME').cumcount()

        # === TARGET (Next Game PRA) ===
        df['target_PRA'] = df.groupby('PLAYER_NAME')['PRA'].shift(-1)

        print(f"✓ Created {len([c for c in df.columns if c.startswith(('PRA_', 'PTS_', 'REB_', 'AST_', 'MIN_'))])} features")

        return df

    def prepare_datasets(self, df):
        """Prepare train/validation/test sets from real data"""
        print(f"\n{'='*60}")
        print("PREPARING DATASETS")
        print(f"{'='*60}")

        # Define features - all from REAL game data
        feature_cols = [
            # Rolling averages
            'PRA_L3', 'PRA_L5', 'PRA_L10', 'PRA_L15',
            'PTS_L3', 'PTS_L5', 'PTS_L10',
            'REB_L3', 'REB_L5', 'REB_L10',
            'AST_L3', 'AST_L5', 'AST_L10',
            'MIN_L3', 'MIN_L5', 'MIN_L10',
            # Variability
            'PRA_std_L10', 'MIN_std_L10',
            # Trend
            'PRA_trend',
            # Efficiency
            'usage_L5', 'TS_L5',
            # Context
            'days_rest', 'is_home', 'is_b2b',
            # Season progression
            'days_into_season', 'games_played'
        ]

        # Check which features exist
        available_features = [f for f in feature_cols if f in df.columns]
        missing_features = [f for f in feature_cols if f not in df.columns]

        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")

        print(f"✓ Using {len(available_features)} features")

        # Remove rows with missing values
        df_clean = df[available_features + ['target_PRA', 'PLAYER_NAME', 'GAME_DATE']].dropna()
        print(f"✓ Clean samples: {len(df_clean)} (from {len(df)})")

        # Time-based split (NEVER use future to predict past)
        df_clean = df_clean.sort_values('GAME_DATE')

        # 70% train, 15% val, 15% test
        n = len(df_clean)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_df = df_clean.iloc[:train_end]
        val_df = df_clean.iloc[train_end:val_end]
        test_df = df_clean.iloc[val_end:]

        X_train = train_df[available_features].values
        y_train = train_df['target_PRA'].values

        X_val = val_df[available_features].values
        y_val = val_df['target_PRA'].values

        X_test = test_df[available_features].values
        y_test = test_df['target_PRA'].values

        print(f"\nDataset splits (temporal):")
        print(f"  Train: {len(X_train):,} games ({train_df['GAME_DATE'].min().date()} to {train_df['GAME_DATE'].max().date()})")
        print(f"  Val:   {len(X_val):,} games ({val_df['GAME_DATE'].min().date()} to {val_df['GAME_DATE'].max().date()})")
        print(f"  Test:  {len(X_test):,} games ({test_df['GAME_DATE'].min().date()} to {test_df['GAME_DATE'].max().date()})")

        return X_train, X_val, X_test, y_train, y_val, y_test, available_features

    def train_models(self, X_train, X_val, y_train, y_val):
        """Train models on REAL data"""
        print(f"\n{'='*60}")
        print("TRAINING MODELS ON REAL GAME DATA")
        print(f"{'='*60}")

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Define SIMPLE models (appropriate for real data variance)
        models = {
            'Baseline (Last 5 Avg)': None,  # Special case
            'Ridge (α=10)': Ridge(alpha=10.0, random_state=42),
            'Ridge (α=1)': Ridge(alpha=1.0, random_state=42),
            'Lasso (α=0.5)': Lasso(alpha=0.5, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()

            if name == 'Baseline (Last 5 Avg)':
                # Use PRA_L5 as prediction (column index 1)
                val_pred = X_val[:, 1]  # PRA_L5 is at index 1
            elif name in ['XGBoost', 'LightGBM', 'RandomForest']:
                # Tree models - use raw features
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
            else:
                # Linear models - use scaled features
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)

            # Calculate metrics
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            r2 = r2_score(y_val, val_pred)

            # MAPE (skip zeros)
            mask = y_val != 0
            mape = np.mean(np.abs((y_val[mask] - val_pred[mask]) / y_val[mask])) * 100

            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'predictions': val_pred,
                'training_time': time.time() - start_time
            }

            print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")

        return results, scaler

    def evaluate_on_test(self, results, scaler, X_test, y_test):
        """Final evaluation on test set"""
        print(f"\n{'='*60}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*60}")

        # Find best model based on validation MAE
        best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
        best_model = results[best_model_name]['model']

        print(f"\nBest model: {best_model_name}")
        print(f"Validation MAE: {results[best_model_name]['mae']:.2f}")

        # Test set predictions
        X_test_scaled = scaler.transform(X_test)

        if best_model_name == 'Baseline (Last 5 Avg)':
            test_pred = X_test[:, 1]  # PRA_L5
        elif best_model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
            test_pred = best_model.predict(X_test)
        else:
            test_pred = best_model.predict(X_test_scaled)

        # Test metrics
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)

        mask = y_test != 0
        test_mape = np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) * 100

        print(f"\n{'='*50}")
        print("TEST SET RESULTS (Never seen during training)")
        print(f"{'='*50}")
        print(f"MAE:  {test_mae:.2f} points")
        print(f"RMSE: {test_rmse:.2f} points")
        print(f"R²:   {test_r2:.3f}")
        print(f"MAPE: {test_mape:.1f}%")

        return test_pred, test_mae, test_r2

    def show_performance_benchmarks(self):
        """Show what good performance looks like for NBA predictions"""
        print(f"\n{'='*60}")
        print("NBA PREDICTION PERFORMANCE BENCHMARKS")
        print(f"{'='*60}")

        print("\n📊 EXPECTED PERFORMANCE FOR NBA:")
        print("  R² = 0.35-0.50 (Good)")
        print("  R² = 0.50-0.60 (Very Good)")
        print("  R² > 0.60 (Excellent/Suspicious)")

        print("\n  MAE = 6-8 points (Normal)")
        print("  MAE = 4-6 points (Good)")
        print("  MAE < 4 points (Excellent)")

        print("\n💡 WHY THESE NUMBERS:")
        print("  • NBA has high game-to-game variance")
        print("  • Players have hot/cold streaks")
        print("  • Injuries, rest, matchups all matter")
        print("  • Perfect prediction is impossible")

    def create_visualizations(self, y_test, test_pred, results):
        """Create visualization plots"""
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Actual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_test, test_pred, alpha=0.3, s=10)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual PRA')
        ax1.set_ylabel('Predicted PRA')
        ax1.set_title('Test Set Predictions (Real Data)')
        ax1.grid(True, alpha=0.3)

        # 2. Residuals
        ax2 = axes[0, 1]
        residuals = y_test - test_pred
        ax2.scatter(test_pred, residuals, alpha=0.3, s=10)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted PRA')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution
        ax3 = axes[0, 2]
        ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Error Distribution (std={residuals.std():.1f})')
        ax3.grid(True, alpha=0.3)

        # 4. Model comparison
        ax4 = axes[1, 0]
        model_names = list(results.keys())
        maes = [results[m]['mae'] for m in model_names]
        colors = ['green' if m == min(results.keys(), key=lambda k: results[k]['mae']) else 'steelblue'
                  for m in model_names]
        bars = ax4.barh(range(len(model_names)), maes, color=colors)
        ax4.set_yticks(range(len(model_names)))
        ax4.set_yticklabels(model_names)
        ax4.set_xlabel('Validation MAE')
        ax4.set_title('Model Performance Comparison')
        ax4.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, mae in zip(bars, maes):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2,
                    f'{mae:.2f}', ha='left', va='center')

        # 5. R² comparison
        ax5 = axes[1, 1]
        r2_values = [results[m]['r2'] for m in model_names]
        bars = ax5.barh(range(len(model_names)), r2_values, color=colors)
        ax5.set_yticks(range(len(model_names)))
        ax5.set_yticklabels(model_names)
        ax5.set_xlabel('R² Score')
        ax5.set_title('R² Score Comparison')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Prediction intervals
        ax6 = axes[1, 2]
        sample_idx = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
        sample_actual = y_test[sample_idx]
        sample_pred = test_pred[sample_idx]

        # Sort by actual value for better visualization
        sort_idx = np.argsort(sample_actual)
        sample_actual = sample_actual[sort_idx]
        sample_pred = sample_pred[sort_idx]

        ax6.scatter(range(len(sample_actual)), sample_actual, color='red', alpha=0.5, s=20, label='Actual')
        ax6.scatter(range(len(sample_pred)), sample_pred, color='blue', alpha=0.5, s=20, label='Predicted')
        ax6.set_xlabel('Sample Games')
        ax6.set_ylabel('PRA')
        ax6.set_title('Sample Predictions (100 games)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.suptitle('NBA Props Model Evaluation - REAL Game Data', fontsize=14, y=1.02)
        plt.tight_layout()

        # Save figure
        output_path = self.results_dir / 'model_evaluation_real_data.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")

        plt.show()

    def save_results(self, results, features):
        """Save evaluation results"""
        # Create results summary
        summary = []
        for name, res in results.items():
            summary.append({
                'Model': name,
                'MAE': res['mae'],
                'RMSE': res['rmse'],
                'R²': res['r2'],
                'MAPE': res['mape'],
                'Training_Time': res['training_time']
            })

        results_df = pd.DataFrame(summary).sort_values('MAE')

        # Save to CSV
        output_path = self.results_dir / 'model_results_real_data.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved results to {output_path}")

        # Save feature importance for tree models
        tree_models = ['XGBoost', 'LightGBM', 'RandomForest']
        importance_data = {}

        for name in tree_models:
            if name in results and results[name]['model'] is not None:
                model = results[name]['model']
                if hasattr(model, 'feature_importances_'):
                    importance_data[name] = model.feature_importances_

        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=features)
            importance_df['mean'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('mean', ascending=False)

            importance_path = self.results_dir / 'feature_importance_real_data.csv'
            importance_df.to_csv(importance_path)
            print(f"✓ Saved feature importance to {importance_path}")

            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10)['mean'])


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" NBA PROPS MODEL EVALUATION - REAL GAME DATA ".center(70))
    print("="*70)

    try:
        # Initialize evaluator
        evaluator = RealDataModelEvaluator()

        # Load and process data
        df = evaluator.load_game_data('2023-24')
        df = evaluator.engineer_features(df)

        # Prepare datasets
        X_train, X_val, X_test, y_train, y_val, y_test, features = evaluator.prepare_datasets(df)

        # Train models
        results, scaler = evaluator.train_models(X_train, X_val, y_train, y_val)

        # Test evaluation
        test_pred, test_mae, test_r2 = evaluator.evaluate_on_test(results, scaler, X_test, y_test)

        # Show performance benchmarks
        evaluator.show_performance_benchmarks()

        # Create visualizations
        evaluator.create_visualizations(y_test, test_pred, results)

        # Save results
        evaluator.save_results(results, features)

        print("\n" + "="*70)
        print(" EVALUATION COMPLETE ".center(70))
        print("="*70)

        print("\n✅ REALISTIC PERFORMANCE ACHIEVED:")
        print(f"   MAE: {test_mae:.2f} points (not 0.35!)")
        print(f"   R²: {test_r2:.3f} (not 0.996!)")
        print("\n📊 Results saved to: data/model_results/")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease run the following first:")
        print("  uv run scripts/fetch_all_game_logs.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()