"""
Simple training script for NBA props model
Direct implementation with opponent features and proper validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from datetime import datetime
import xgboost as xgb
from lightgbm import LGBMRegressor

from features.opponent_features import OpponentFeatures
from models.validation import NBATimeSeriesValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPropsTrainer:
    """
    Simple trainer for PRA predictions
    Focus on what works without over-engineering
    """

    def __init__(self, data_path: str = "/Users/diyagamah/Documents/nba_props_model/data"):
        self.data_path = Path(data_path)
        self.opponent_features = OpponentFeatures(data_path)
        self.validator = NBATimeSeriesValidator(n_splits=5, test_size=30)

    def prepare_training_data(self, player_stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with all features including opponent adjustments
        """
        # Calculate basic rolling features
        df = self.add_rolling_features(player_stats_df)

        # Add opponent features
        df = self.opponent_features.get_opponent_features(df)

        # Create matchup interaction features
        matchup_features = self.opponent_features.create_matchup_features(df, df)
        df = pd.concat([df, matchup_features], axis=1)

        # Create target (PRA)
        df['PRA'] = df['Points'] + df['Rebounds'] + df['Assists']

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple but effective rolling features
        """
        # Sort by player and date
        df = df.sort_values(['Player', 'Date'])

        # Group by player for rolling calculations
        player_groups = df.groupby('Player')

        # Last 5 games average
        for stat in ['Points', 'Rebounds', 'Assists', 'Minutes']:
            if stat in df.columns:
                df[f'{stat}_L5'] = player_groups[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )

        # Last 10 games average
        for stat in ['Points', 'Rebounds', 'Assists']:
            if stat in df.columns:
                df[f'{stat}_L10'] = player_groups[stat].transform(
                    lambda x: x.rolling(10, min_periods=2).mean()
                )

        # Volatility (standard deviation over last 5 games)
        for stat in ['Points', 'Rebounds', 'Assists']:
            if stat in df.columns:
                df[f'{stat}_volatility'] = player_groups[stat].transform(
                    lambda x: x.rolling(5, min_periods=2).std()
                )

        # Trend (difference between L5 and L10 averages)
        for stat in ['Points', 'Rebounds', 'Assists']:
            if f'{stat}_L5' in df.columns and f'{stat}_L10' in df.columns:
                df[f'{stat}_trend'] = df[f'{stat}_L5'] - df[f'{stat}_L10']

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for training
        """
        # Exclude non-feature columns
        exclude_cols = ['Player', 'Date', 'Season', 'Team', 'Opponent',
                       'PRA', 'Points', 'Rebounds', 'Assists']

        # Include these patterns
        include_patterns = ['_L5', '_L10', '_volatility', '_trend',
                          'opp_', 'vs_', 'pace_factor', 'def_difficulty',
                          'Usage', 'PSA', 'AST%', 'MIN', 'fgDR%', 'fgOR%']

        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if any(pattern in col for pattern in include_patterns):
                    feature_cols.append(col)

        return feature_cols

    def train_model(self, df: pd.DataFrame, model_type: str = 'xgboost') -> Dict:
        """
        Train model with proper time series validation
        """
        # Prepare features
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols + ['Date', 'Player', 'Season']].copy()
        y = df['PRA'].copy()

        # Remove rows with NaN in features or target
        valid_idx = ~(X[feature_cols].isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]

        logger.info(f"Training with {len(X)} samples and {len(feature_cols)} features")

        # Initialize model
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:  # lightgbm
            model = LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

        # Perform time series validation
        validation_results = self.validator.validate_model(model, X, y)

        # Train final model on all data
        model.fit(X[feature_cols], y)

        # Get feature importance
        if model_type == 'xgboost':
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_

        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return {
            'model': model,
            'features': feature_cols,
            'validation_results': validation_results,
            'feature_importance': feature_importance,
            'training_samples': len(X)
        }

    def save_model(self, model_dict: Dict, output_path: str = None):
        """
        Save trained model and metadata
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.data_path / 'models' / f'nba_props_model_{timestamp}.pkl'

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(model_dict, f)

        logger.info(f"Model saved to {output_path}")

        # Save feature importance as CSV
        importance_path = output_path.parent / f"{output_path.stem}_importance.csv"
        model_dict['feature_importance'].to_csv(importance_path, index=False)

        # Save validation results
        import json
        validation_path = output_path.parent / f"{output_path.stem}_validation.json"
        with open(validation_path, 'w') as f:
            # Convert numpy types for JSON serialization
            val_results = model_dict['validation_results'].copy()
            val_results['predictions'] = [float(x) for x in val_results['predictions']]
            val_results['actuals'] = [float(x) for x in val_results['actuals']]
            json.dump(val_results, f, indent=2)

        return output_path


# Usage example
if __name__ == "__main__":
    # This is how you would use it
    trainer = NBAPropsTrainer()

    # Load your player stats data
    # df = pd.read_csv('your_player_stats.csv')

    # Prepare data with all features
    # df_prepared = trainer.prepare_training_data(df)

    # Train model
    # model_results = trainer.train_model(df_prepared, model_type='xgboost')

    # Save model
    # trainer.save_model(model_results)

    print("Trainer initialized. Load your data and call prepare_training_data() and train_model()")