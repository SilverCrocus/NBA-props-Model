"""
NBA Props Model Training Script with MLflow Integration
Integrates experiment tracking into the existing training pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from features.opponent_features import OpponentFeatures
from models.validation import NBATimeSeriesValidator

# Import MLflow integration
from mlflow_integration.tracker import NBAPropsTracker, enable_autologging
from mlflow_integration.registry import ModelRegistry, DEFAULT_PRODUCTION_CRITERIA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPropsMLflowTrainer:
    """
    NBA Props trainer with full MLflow experiment tracking
    Extends existing trainer with comprehensive logging
    """

    def __init__(
        self,
        data_path: str = "/Users/diyagamah/Documents/nba_props_model/data",
        experiment_name: str = "Phase1_Foundation"
    ):
        self.data_path = Path(data_path)
        self.opponent_features = OpponentFeatures(str(data_path))
        self.validator = NBATimeSeriesValidator(n_splits=5, test_size=30)

        # Initialize MLflow tracker
        self.tracker = NBAPropsTracker(experiment_name=experiment_name)
        self.registry = ModelRegistry()

        logger.info(f"Initialized trainer for experiment: {experiment_name}")

    def prepare_training_data(self, player_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all features"""
        logger.info("Preparing training data...")

        # Calculate basic rolling features
        df = self.add_rolling_features(player_stats_df)

        # Add opponent features
        df = self.opponent_features.get_opponent_features(df)

        # Create matchup interaction features
        matchup_features = self.opponent_features.create_matchup_features(df, df)
        df = pd.concat([df, matchup_features], axis=1)

        # Create target (PRA)
        if 'PRA' not in df.columns:
            df['PRA'] = df['Points'] + df['Rebounds'] + df['Assists']

        logger.info(f"Prepared {len(df)} samples with {len(df.columns)} features")
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        df = df.sort_values(['Player', 'Date'])
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

        # Volatility
        for stat in ['Points', 'Rebounds', 'Assists']:
            if stat in df.columns:
                df[f'{stat}_volatility'] = player_groups[stat].transform(
                    lambda x: x.rolling(5, min_periods=2).std()
                )

        # Trend
        for stat in ['Points', 'Rebounds', 'Assists']:
            if f'{stat}_L5' in df.columns and f'{stat}_L10' in df.columns:
                df[f'{stat}_trend'] = df[f'{stat}_L5'] - df[f'{stat}_L10']

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns"""
        exclude_cols = [
            'Player', 'Date', 'Season', 'Team', 'Opponent',
            'PRA', 'Points', 'Rebounds', 'Assists'
        ]

        include_patterns = [
            '_L5', '_L10', '_volatility', '_trend',
            'opp_', 'vs_', 'pace_factor', 'def_difficulty',
            'Usage', 'PSA', 'AST%', 'MIN', 'fgDR%', 'fgOR%'
        ]

        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if any(pattern in col for pattern in include_patterns):
                    feature_cols.append(col)

        return feature_cols

    def train_model(
        self,
        df: pd.DataFrame,
        model_type: str = 'xgboost',
        run_name: str = None,
        hyperparams: Dict = None,
        register_model: bool = False,
        tags: Dict[str, str] = None
    ) -> Dict:
        """
        Train model with full MLflow tracking

        Args:
            df: Training DataFrame
            model_type: 'xgboost' or 'lightgbm'
            run_name: Name for this MLflow run
            hyperparams: Model hyperparameters
            register_model: Whether to register model in registry
            tags: Additional tags for the run

        Returns:
            Dictionary with training results
        """
        # Generate run name if not provided
        if run_name is None:
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}"

        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        # Default tags
        if tags is None:
            tags = {}

        tags['model_type'] = model_type
        tags['feature_version'] = 'v1.0'

        # Start MLflow run
        self.tracker.start_run(run_name=run_name, tags=tags)

        try:
            # Enable autologging
            enable_autologging(model_type)

            # Prepare features
            feature_cols = self.get_feature_columns(df)
            X = df[feature_cols + ['Date', 'Player', 'Season']].copy()
            y = df['PRA'].copy()

            # Remove rows with NaN
            valid_idx = ~(X[feature_cols].isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]

            logger.info(f"Training with {len(X)} samples and {len(feature_cols)} features")

            # Log training configuration
            training_config = {
                'n_samples': len(X),
                'n_features': len(feature_cols),
                'train_seasons': df['Season'].unique().tolist() if 'Season' in df.columns else [],
                'validation_type': 'time_series_cv',
                'n_cv_splits': self.validator.n_splits,
            }
            self.tracker.log_training_config(training_config)

            # Log feature configuration
            feature_config = {
                'feature_set_version': 'v1.0',
                'n_features': len(feature_cols),
                'feature_names': feature_cols,
                'feature_categories': self._categorize_features(feature_cols)
            }
            self.tracker.log_feature_config(feature_config)

            # Log hyperparameters
            self.tracker.log_params(hyperparams)

            # Initialize model
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(**hyperparams)
            elif model_type == 'lightgbm':
                model = LGBMRegressor(**hyperparams)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Perform time series validation
            logger.info("Starting time series cross-validation...")
            validation_results = self.validator.validate_model(model, X, y)

            # Log validation metrics
            val_metrics = {
                'mae': validation_results['mean_mae'],
                'rmse': validation_results['mean_rmse'],
                'r2': validation_results['mean_r2'],
                'mae_std': validation_results['std_mae'],
            }
            self.tracker.log_validation_metrics(val_metrics)

            # Train final model on all data
            logger.info("Training final model...")
            start_time = datetime.now()
            model.fit(X[feature_cols], y)
            training_time = (datetime.now() - start_time).total_seconds()

            # Log training time
            self.tracker.log_training_metrics({'training_time': training_time})

            # Get and log feature importance
            importance = model.feature_importances_
            self.tracker.log_feature_importance(feature_cols, importance)

            # Generate predictions for analysis
            y_pred = model.predict(X[feature_cols])

            # Calculate final metrics
            final_metrics = {
                'train_mae': mean_absolute_error(y, y_pred),
                'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'train_r2': r2_score(y, y_pred),
            }
            self.tracker.log_training_metrics(final_metrics)

            # Log residuals plot
            self.tracker.log_residuals_plot(y.values, y_pred)

            # Log predictions
            predictions_df = pd.DataFrame({
                'actual': y.values,
                'predicted': y_pred,
                'error': y.values - y_pred,
                'abs_error': np.abs(y.values - y_pred)
            })
            self.tracker.log_predictions(predictions_df, "train_predictions.csv")

            # Log model
            self.tracker.log_model(
                model,
                model_type=model_type,
                registered_model_name="NBAPropsModel" if register_model else None
            )

            # Compile results
            results = {
                'model': model,
                'features': feature_cols,
                'validation_results': validation_results,
                'final_metrics': final_metrics,
                'training_samples': len(X),
                'run_id': self.tracker.run_id,
            }

            # End run successfully
            self.tracker.end_run(status="FINISHED")
            logger.info(f"Training completed successfully! Run ID: {self.tracker.run_id}")

            return results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.tracker.end_run(status="FAILED")
            raise

    def train_and_evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        model_type: str = 'xgboost',
        run_name: str = None,
        hyperparams: Dict = None,
        betting_thresholds: Dict = None,
        register_model: bool = True,
        tags: Dict[str, str] = None
    ) -> Dict:
        """
        Train model with walk-forward validation and betting metrics

        Args:
            df: Full dataset with multiple seasons
            model_type: 'xgboost' or 'lightgbm'
            run_name: Name for this MLflow run
            hyperparams: Model hyperparameters
            betting_thresholds: Thresholds for betting metrics
            register_model: Whether to register model
            tags: Additional tags

        Returns:
            Dictionary with results including betting metrics
        """
        # Generate run name
        if run_name is None:
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_type}_walkforward"

        if tags is None:
            tags = {}
        tags['validation_type'] = 'walk_forward'

        # Default betting thresholds
        if betting_thresholds is None:
            betting_thresholds = {
                'min_edge': 0.02,  # 2% minimum edge
                'confidence_threshold': 0.6,
            }

        # Start MLflow run
        self.tracker.start_run(run_name=run_name, tags=tags)

        try:
            # Train basic model first
            base_results = self.train_model(
                df=df,
                model_type=model_type,
                run_name=None,  # We already started the run
                hyperparams=hyperparams,
                register_model=False,
                tags=None
            )

            model = base_results['model']
            feature_cols = base_results['features']

            # Perform walk-forward validation by season
            logger.info("Starting walk-forward validation...")

            seasons = sorted(df['Season'].unique()) if 'Season' in df.columns else []

            wf_results = []
            for i in range(2, len(seasons)):  # Need at least 2 seasons for training
                train_seasons = seasons[:i]
                test_season = seasons[i]

                logger.info(f"Training on {train_seasons}, testing on {test_season}")

                # Split data
                train_mask = df['Season'].isin(train_seasons)
                test_mask = df['Season'] == test_season

                X_train = df[train_mask][feature_cols]
                y_train = df[train_mask]['PRA']
                X_test = df[test_mask][feature_cols]
                y_test = df[test_mask]['PRA']

                # Remove NaN
                train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
                test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

                X_train = X_train[train_valid]
                y_train = y_train[train_valid]
                X_test = X_test[test_valid]
                y_test = y_test[test_valid]

                # Train model
                if model_type == 'xgboost':
                    wf_model = xgb.XGBRegressor(**(hyperparams or {}))
                else:
                    wf_model = LGBMRegressor(**(hyperparams or {}))

                wf_model.fit(X_train, y_train)

                # Predict
                y_pred = wf_model.predict(X_test)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                wf_results.append({
                    'test_season': test_season,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                })

                # Log season-specific metrics
                self.tracker.log_metrics({
                    f'wf_{test_season}_mae': mae,
                    f'wf_{test_season}_rmse': rmse,
                    f'wf_{test_season}_r2': r2,
                })

            # Calculate overall walk-forward metrics
            wf_mae_mean = np.mean([r['mae'] for r in wf_results])
            wf_rmse_mean = np.mean([r['rmse'] for r in wf_results])
            wf_r2_mean = np.mean([r['r2'] for r in wf_results])

            # Log overall walk-forward metrics
            wf_metrics = {
                'mae': wf_mae_mean,
                'rmse': wf_rmse_mean,
                'r2': wf_r2_mean,
            }
            self.tracker.log_validation_metrics(wf_metrics)

            # Simulate betting metrics (placeholder - replace with actual betting logic)
            betting_metrics = self._calculate_betting_metrics(
                df, model, feature_cols, betting_thresholds
            )
            self.tracker.log_betting_metrics(betting_metrics)

            # Log walk-forward results as artifact
            wf_df = pd.DataFrame(wf_results)
            self.tracker.log_predictions(wf_df, "walk_forward_results.csv")

            # Register model if it meets production criteria
            if register_model:
                # Check if meets criteria
                meets_criteria = self.registry.evaluate_for_production(
                    model_name="NBAPropsModel",
                    version=1,  # This would be dynamically set
                    criteria=DEFAULT_PRODUCTION_CRITERIA
                )

                if meets_criteria:
                    logger.info("Model meets production criteria, promoting to Staging")
                    # Register and promote (this would be done via registry)
                else:
                    logger.info("Model does not meet production criteria")

            # Compile final results
            results = {
                **base_results,
                'walk_forward_results': wf_results,
                'walk_forward_metrics': wf_metrics,
                'betting_metrics': betting_metrics,
            }

            self.tracker.end_run(status="FINISHED")
            logger.info("Walk-forward validation completed!")

            return results

        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            self.tracker.end_run(status="FAILED")
            raise

    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features by tier"""
        tiers = {
            'tier1_core': [],
            'tier2_contextual': [],
            'tier3_temporal': []
        }

        tier1_keywords = ['USG', 'PSA', 'AST', 'PER', 'efficiency', 'OR%', 'DR%']
        tier2_keywords = ['Minutes', 'Rest', 'B2B', 'opp_', 'Pos_vs']
        tier3_keywords = ['L5', 'L10', 'L15', 'ewma', 'volatility', 'trend']

        for feature in features:
            categorized = False

            # Check Tier 3 first
            for keyword in tier3_keywords:
                if keyword in feature:
                    tiers['tier3_temporal'].append(feature)
                    categorized = True
                    break

            if not categorized:
                # Check Tier 2
                for keyword in tier2_keywords:
                    if keyword in feature:
                        tiers['tier2_contextual'].append(feature)
                        categorized = True
                        break

            if not categorized:
                # Check Tier 1
                for keyword in tier1_keywords:
                    if keyword in feature:
                        tiers['tier1_core'].append(feature)
                        categorized = True
                        break

            # Default to Tier 1
            if not categorized:
                tiers['tier1_core'].append(feature)

        return tiers

    def _calculate_betting_metrics(
        self,
        df: pd.DataFrame,
        model,
        feature_cols: List[str],
        thresholds: Dict
    ) -> Dict[str, float]:
        """
        Calculate betting metrics (placeholder - implement actual betting logic)

        This is a simplified version. Replace with actual betting simulation.
        """
        # TODO: Implement actual betting logic with odds data
        # For now, return placeholder metrics

        return {
            'roi': 0.065,  # 6.5% ROI
            'win_rate': 0.58,  # 58% win rate
            'clv': 0.023,  # 2.3% closing line value
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15,
            'brier_score': 0.18,
            'calibration_error': 0.04,
            'n_bets': 850,
        }


# Usage example
if __name__ == "__main__":
    # Initialize trainer for Phase 1
    trainer = NBAPropsMLflowTrainer(
        experiment_name="Phase1_Foundation"
    )

    # Load your data
    # df = pd.read_csv('your_player_stats.csv')

    # Prepare data
    # df_prepared = trainer.prepare_training_data(df)

    # Train baseline XGBoost model
    # results = trainer.train_model(
    #     df=df_prepared,
    #     model_type='xgboost',
    #     run_name='baseline_xgb_v1',
    #     hyperparams={
    #         'n_estimators': 200,
    #         'max_depth': 6,
    #         'learning_rate': 0.05,
    #         'subsample': 0.8,
    #         'colsample_bytree': 0.8,
    #     },
    #     register_model=True,
    #     tags={'model_version': 'v1.0.0', 'description': 'Baseline model'}
    # )

    # Or train with walk-forward validation
    # results = trainer.train_and_evaluate_walk_forward(
    #     df=df_prepared,
    #     model_type='xgboost',
    #     run_name='baseline_xgb_walkforward',
    #     register_model=True
    # )

    print("Trainer initialized. Load data and call train_model() or train_and_evaluate_walk_forward()")
