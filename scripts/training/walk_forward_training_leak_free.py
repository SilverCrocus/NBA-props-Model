"""
Walk-Forward Training & Validation - Leak-Free Implementation

This script implements PROPER walk-forward methodology for BOTH training and validation:
- Features are calculated on-the-fly using ONLY past data
- No pre-calculated lag features that could contain future information
- Consistent feature calculation between training and validation
- Integrates CTG advanced stats, rest features, and recent form

Expected Results:
- MAE: 9-10 points (realistic, no leakage)
- Win Rate: ~52% (slightly above breakeven)
- This establishes TRUE baseline performance

Author: NBA Props Model - Week 1 Day 2
Date: October 14, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from datetime import datetime, timedelta
import sys
import logging

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from ctg_feature_builder import CTGFeatureBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MLflow tracking
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.mlflow_integration.tracker import NBAPropsTracker, enable_autologging


def calculate_lag_features(player_history: pd.DataFrame, lags=[1, 3, 5, 7]) -> dict:
    """
    Calculate lag features using ONLY historical games (no current game).

    Args:
        player_history: DataFrame of player's games BEFORE current date
        lags: List of lag periods to calculate

    Returns:
        Dictionary of lag features
    """
    features = {}

    if len(player_history) == 0:
        for lag in lags:
            features[f'PRA_lag{lag}'] = 0
            features[f'MIN_lag{lag}'] = 0
        return features

    # Sort by date descending (most recent first)
    history = player_history.sort_values('GAME_DATE', ascending=False)

    for lag in lags:
        if len(history) >= lag:
            features[f'PRA_lag{lag}'] = history.iloc[lag-1]['PRA']
            features[f'MIN_lag{lag}'] = history.iloc[lag-1].get('MIN', 0)
        else:
            features[f'PRA_lag{lag}'] = 0
            features[f'MIN_lag{lag}'] = 0

    return features


def calculate_rolling_features(player_history: pd.DataFrame, windows=[5, 10, 20]) -> dict:
    """
    Calculate rolling average features using ONLY historical games.

    Args:
        player_history: DataFrame of player's games BEFORE current date
        windows: List of rolling windows

    Returns:
        Dictionary of rolling features
    """
    features = {}

    if len(player_history) == 0:
        for window in windows:
            features[f'PRA_L{window}_mean'] = 0
            features[f'PRA_L{window}_std'] = 0
            features[f'MIN_L{window}_mean'] = 0
        return features

    history = player_history.sort_values('GAME_DATE', ascending=False)

    for window in windows:
        if len(history) >= window:
            recent_games = history.iloc[:window]
            features[f'PRA_L{window}_mean'] = recent_games['PRA'].mean()
            features[f'PRA_L{window}_std'] = recent_games['PRA'].std()
            features[f'MIN_L{window}_mean'] = recent_games.get('MIN', pd.Series([0])).mean()
        elif len(history) >= 3:  # At least 3 games for partial calculation
            features[f'PRA_L{window}_mean'] = history['PRA'].mean()
            features[f'PRA_L{window}_std'] = history['PRA'].std()
            features[f'MIN_L{window}_mean'] = history.get('MIN', pd.Series([0])).mean()
        else:
            features[f'PRA_L{window}_mean'] = 0
            features[f'PRA_L{window}_std'] = 0
            features[f'MIN_L{window}_mean'] = 0

    return features


def calculate_ewma_features(player_history: pd.DataFrame, spans=[5, 10]) -> dict:
    """
    Calculate EWMA (Exponentially Weighted Moving Average) features.

    Args:
        player_history: DataFrame of player's games BEFORE current date
        spans: List of span parameters

    Returns:
        Dictionary of EWMA features
    """
    features = {}

    if len(player_history) < 3:
        for span in spans:
            features[f'PRA_ewma{span}'] = 0
        return features

    history = player_history.sort_values('GAME_DATE', ascending=True)

    for span in spans:
        ewma_value = history['PRA'].ewm(span=span, min_periods=1).mean().iloc[-1]
        features[f'PRA_ewma{span}'] = ewma_value

    return features


def calculate_rest_features(player_history: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    """
    Calculate rest and schedule fatigue features.

    Args:
        player_history: DataFrame of player's games BEFORE current date
        current_date: Current game date

    Returns:
        Dictionary of rest features
    """
    features = {}

    if len(player_history) == 0:
        features['days_rest'] = 7
        features['is_b2b'] = 0
        features['games_last_7d'] = 0
        return features

    # Get last game date
    last_game = player_history.sort_values('GAME_DATE', ascending=False).iloc[0]
    last_game_date = last_game['GAME_DATE']

    # Days of rest
    days_rest = (current_date - last_game_date).days
    features['days_rest'] = min(days_rest, 7)  # Cap at 7 days

    # Back-to-back indicator
    features['is_b2b'] = 1 if days_rest <= 1 else 0

    # Games in last 7 days
    week_ago = current_date - timedelta(days=7)
    recent_games = player_history[player_history['GAME_DATE'] >= week_ago]
    features['games_last_7d'] = len(recent_games)

    return features


def calculate_trend_features(player_history: pd.DataFrame) -> dict:
    """
    Calculate trend features (recent vs longer-term performance).

    Args:
        player_history: DataFrame of player's games BEFORE current date

    Returns:
        Dictionary of trend features
    """
    features = {}

    if len(player_history) < 10:
        features['PRA_trend'] = 0
        return features

    history = player_history.sort_values('GAME_DATE', ascending=False)

    # Compare last 5 games to games 6-15
    if len(history) >= 15:
        l5_mean = history.iloc[:5]['PRA'].mean()
        l10_mean = history.iloc[5:15]['PRA'].mean()
        features['PRA_trend'] = l5_mean - l10_mean
    else:
        features['PRA_trend'] = 0

    return features


def calculate_all_features(
    player_history: pd.DataFrame,
    current_date: pd.Timestamp,
    player_name: str,
    season: str,
    ctg_builder: CTGFeatureBuilder,
    current_game_stats: dict = None
) -> dict:
    """
    Calculate ALL features for a single prediction.

    Args:
        player_history: Player's games before current date
        current_date: Current game date
        player_name: Player name for CTG lookup
        season: Current season (e.g., "2024-25")
        ctg_builder: CTG feature builder instance
        current_game_stats: Current game stats (MIN, FGA, etc.) - use last game as proxy

    Returns:
        Dictionary of all features
    """
    features = {}

    # Calculate temporal features
    features.update(calculate_lag_features(player_history))
    features.update(calculate_rolling_features(player_history))
    features.update(calculate_ewma_features(player_history))
    features.update(calculate_rest_features(player_history, current_date))
    features.update(calculate_trend_features(player_history))

    # Get CTG season stats
    ctg_feats = ctg_builder.get_player_ctg_features(player_name, season)
    features.update(ctg_feats)

    # Current game stats (use last game as proxy)
    if current_game_stats is None and len(player_history) > 0:
        last_game = player_history.sort_values('GAME_DATE', ascending=False).iloc[0]
        current_game_stats = {
            'MIN': last_game.get('MIN', 0),
            'FGA': last_game.get('FGA', 0),
            'FG_PCT': last_game.get('FG_PCT', 0),
            'FG3A': last_game.get('FG3A', 0),
            'FTA': last_game.get('FTA', 0)
        }

    if current_game_stats:
        features.update(current_game_stats)

    return features


def walk_forward_train_and_validate(
    train_season: str = "2023-24",
    val_season: str = "2024-25",
    min_train_games: int = 10,
    min_history_for_prediction: int = 5
):
    """
    Perform walk-forward training and validation with leak-free features.

    Args:
        train_season: Season to train on
        val_season: Season to validate on
        min_train_games: Minimum games for training
        min_history_for_prediction: Minimum games for making predictions

    Returns:
        Dictionary with results
    """
    logger.info("=" * 80)
    logger.info("WALK-FORWARD TRAINING & VALIDATION - LEAK-FREE")
    logger.info("=" * 80)

    # Initialize MLflow
    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
    tracker.start_run(
        run_name=f"walk_forward_leak_free_{val_season}",
        tags={
            'model_type': 'xgboost',
            'validation_type': 'walk_forward_leak_free',
            'train_season': train_season,
            'val_season': val_season,
            'description': 'Leak-free walk-forward with on-the-fly feature calculation'
        }
    )

    try:
        # Initialize CTG feature builder
        logger.info("\n1. Initializing CTG feature builder...")
        ctg_builder = CTGFeatureBuilder()

        # Load raw game logs
        logger.info("\n2. Loading raw game logs...")

        # Try to load preprocessed game logs first
        game_logs_paths = [
            'data/game_logs/game_logs_2023_24_preprocessed.csv',
            'data/game_logs/game_logs_2024_25_preprocessed.csv',
            'data/game_logs/all_game_logs_combined.csv'
        ]

        all_games = []
        for path in game_logs_paths:
            if Path(path).exists():
                df = pd.read_csv(path)
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

                # Add PRA if not present
                if 'PRA' not in df.columns and all(col in df.columns for col in ['PTS', 'REB', 'AST']):
                    df['PRA'] = df['PTS'] + df['REB'] + df['AST']

                all_games.append(df)
                logger.info(f"   Loaded {path}: {len(df):,} games")

        if not all_games:
            raise FileNotFoundError("No game log files found!")

        # Combine all games
        all_games_df = pd.concat(all_games, ignore_index=True)
        all_games_df = all_games_df.sort_values('GAME_DATE').reset_index(drop=True)

        logger.info(f"\n✅ Total games loaded: {len(all_games_df):,}")
        logger.info(f"   Date range: {all_games_df['GAME_DATE'].min()} to {all_games_df['GAME_DATE'].max()}")

        # Split into training and validation periods
        train_start_date = pd.to_datetime('2023-10-01')
        train_end_date = pd.to_datetime('2024-06-30')
        val_start_date = pd.to_datetime('2024-10-01')
        val_end_date = all_games_df['GAME_DATE'].max()

        train_games = all_games_df[
            (all_games_df['GAME_DATE'] >= train_start_date) &
            (all_games_df['GAME_DATE'] <= train_end_date)
        ].copy()

        val_games = all_games_df[
            (all_games_df['GAME_DATE'] >= val_start_date) &
            (all_games_df['GAME_DATE'] <= val_end_date)
        ].copy()

        logger.info(f"\n3. Data split:")
        logger.info(f"   Training: {len(train_games):,} games ({train_games['GAME_DATE'].min()} to {train_games['GAME_DATE'].max()})")
        logger.info(f"   Validation: {len(val_games):,} games ({val_games['GAME_DATE'].min()} to {val_games['GAME_DATE'].max()})")

        # Build training dataset with walk-forward feature calculation
        logger.info("\n4. Building training dataset with leak-free features...")

        training_samples = []
        train_dates = sorted(train_games['GAME_DATE'].unique())

        logger.info(f"   Processing {len(train_dates)} training dates...")

        for pred_date in tqdm(train_dates[:100], desc="Building training data (sample)"):  # Sample first 100 dates
            games_today = train_games[train_games['GAME_DATE'] == pred_date]
            past_games = train_games[train_games['GAME_DATE'] < pred_date]

            for _, row in games_today.iterrows():
                player_id = row['PLAYER_ID']
                player_name = row.get('PLAYER_NAME', '')

                # Get player history
                player_history = past_games[past_games['PLAYER_ID'] == player_id]

                if len(player_history) < min_train_games:
                    continue

                # Calculate features
                features = calculate_all_features(
                    player_history,
                    pred_date,
                    player_name,
                    train_season,
                    ctg_builder
                )

                # Add target
                features['PRA'] = row['PRA']
                features['GAME_DATE'] = pred_date
                features['PLAYER_ID'] = player_id

                training_samples.append(features)

        train_df = pd.DataFrame(training_samples)
        logger.info(f"\n✅ Training dataset built: {len(train_df):,} samples")

        # Define feature columns
        exclude_cols = ['PRA', 'GAME_DATE', 'PLAYER_ID', 'CTG_Available']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        # Fill NaN with 0
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['PRA']

        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Samples: {len(X_train):,}")

        # Log training config
        training_config = {
            'n_samples': len(X_train),
            'n_features': len(feature_cols),
            'train_period': f"{train_start_date} to {train_end_date}",
            'validation_type': 'walk_forward_leak_free',
            'min_train_games': min_train_games
        }
        tracker.log_training_config(training_config)

        # Train model
        logger.info("\n5. Training XGBoost model...")

        hyperparams = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        tracker.log_params(hyperparams)

        enable_autologging('xgboost')

        model = xgb.XGBRegressor(**hyperparams)
        model.fit(X_train, y_train, verbose=False)

        train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)

        logger.info(f"✅ Model trained - Train MAE: {train_mae:.2f}")

        # Walk-forward validation on 2024-25
        logger.info("\n6. Running walk-forward validation on 2024-25...")

        val_predictions = []
        val_dates = sorted(val_games['GAME_DATE'].unique())

        for pred_date in tqdm(val_dates, desc="Walk-forward validation"):
            games_today = val_games[val_games['GAME_DATE'] == pred_date]

            # Use ALL past games (training + validation up to this date)
            past_games = all_games_df[all_games_df['GAME_DATE'] < pred_date]

            for _, row in games_today.iterrows():
                player_id = row['PLAYER_ID']
                player_name = row.get('PLAYER_NAME', '')

                player_history = past_games[past_games['PLAYER_ID'] == player_id]

                if len(player_history) < min_history_for_prediction:
                    continue

                # Calculate features
                features = calculate_all_features(
                    player_history,
                    pred_date,
                    player_name,
                    val_season,
                    ctg_builder
                )

                # Create feature vector
                feature_vector = [features.get(col, 0) for col in feature_cols]

                # Predict
                pred_pra = model.predict([feature_vector])[0]

                val_predictions.append({
                    'PLAYER_NAME': player_name,
                    'PLAYER_ID': player_id,
                    'GAME_DATE': pred_date,
                    'PRA': row['PRA'],
                    'predicted_PRA': pred_pra,
                    'error': pred_pra - row['PRA'],
                    'abs_error': abs(pred_pra - row['PRA']),
                    'CTG_Available': features.get('CTG_Available', 0)
                })

        val_df = pd.DataFrame(val_predictions)

        logger.info(f"\n✅ Validation complete: {len(val_df):,} predictions")

        # Calculate metrics
        val_mae = mean_absolute_error(val_df['PRA'], val_df['predicted_PRA'])
        val_rmse = np.sqrt(mean_squared_error(val_df['PRA'], val_df['predicted_PRA']))
        val_r2 = r2_score(val_df['PRA'], val_df['predicted_PRA'])

        # Log metrics
        val_metrics = {
            'mae': val_mae,
            'rmse': val_rmse,
            'r2': val_r2,
            'within_3pts_pct': (val_df['abs_error'] <= 3).mean() * 100,
            'within_5pts_pct': (val_df['abs_error'] <= 5).mean() * 100,
            'within_10pts_pct': (val_df['abs_error'] <= 10).mean() * 100,
            'ctg_coverage': (val_df['CTG_Available'] == 1).mean() * 100
        }
        tracker.log_validation_metrics(val_metrics)

        # Save predictions
        output_path = Path('data/results/walk_forward_leak_free_2024_25.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(output_path, index=False)

        logger.info("\n" + "=" * 80)
        logger.info("RESULTS - LEAK-FREE WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        logger.info(f"\nValidation Metrics (2024-25 Season):")
        logger.info(f"  MAE: {val_mae:.2f} points")
        logger.info(f"  RMSE: {val_rmse:.2f} points")
        logger.info(f"  R²: {val_r2:.3f}")
        logger.info(f"  Within ±3 pts: {val_metrics['within_3pts_pct']:.1f}%")
        logger.info(f"  Within ±5 pts: {val_metrics['within_5pts_pct']:.1f}%")
        logger.info(f"  Within ±10 pts: {val_metrics['within_10pts_pct']:.1f}%")
        logger.info(f"\nFeature Coverage:")
        logger.info(f"  CTG data available: {val_metrics['ctg_coverage']:.1f}%")
        logger.info(f"\n✅ Results saved to {output_path}")

        # Log feature importance
        importance = model.feature_importances_
        tracker.log_feature_importance(feature_cols, importance)

        # Log model
        tracker.log_model(model, model_type='xgboost')

        tracker.end_run(status="FINISHED")

        return {
            'model': model,
            'features': feature_cols,
            'val_df': val_df,
            'metrics': val_metrics,
            'run_id': tracker.run_id
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    results = walk_forward_train_and_validate(
        train_season="2023-24",
        val_season="2024-25",
        min_train_games=10,
        min_history_for_prediction=5
    )

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nRun ID: {results['run_id']}")
    logger.info("\nView in MLflow: uv run mlflow ui")
