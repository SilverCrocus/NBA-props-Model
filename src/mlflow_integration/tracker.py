"""
MLflow Experiment Tracker for NBA Props Model
Handles experiment tracking, parameter/metric logging, and artifact management
"""

import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPropsTracker:
    """
    Wrapper for MLflow experiment tracking optimized for NBA props model
    Handles all experiment logging, artifact management, and metric tracking
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = None,
        artifact_location: str = None
    ):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of the experiment (e.g., "Phase1_Foundation")
            tracking_uri: MLflow tracking URI (defaults to local)
            artifact_location: Where to store artifacts (defaults to mlflow_artifacts/)
        """
        if tracking_uri is None:
            tracking_uri = "file:///Users/diyagamah/Documents/nba_props_model/mlruns"

        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if artifact_location is None:
                    artifact_location = f"mlflow_artifacts/{experiment_name}"

                experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")

            self.experiment_name = experiment_name
            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise

        self.run_id = None
        self.run_name = None

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """
        Start a new MLflow run

        Args:
            run_name: Name for this run (e.g., "20251014_xgb_baseline")
            tags: Additional tags for the run
        """
        if run_name is None:
            run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run"

        self.run_name = run_name

        mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = mlflow.active_run().info.run_id

        logger.info(f"Started run: {run_name} (ID: {self.run_id})")

        # Log standard tags
        mlflow.set_tag("project", "nba_props_model")
        mlflow.set_tag("run_name", run_name)
        mlflow.set_tag("start_time", datetime.now().isoformat())

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

    def log_params(self, params: Dict[str, Any]):
        """Log model hyperparameters and configuration"""
        try:
            for key, value in params.items():
                # MLflow params must be strings
                mlflow.log_param(key, str(value))
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")

    def log_feature_config(self, feature_config: Dict[str, Any]):
        """
        Log feature configuration as artifact

        Args:
            feature_config: Dictionary with feature set information
        """
        try:
            # Log as JSON artifact
            mlflow.log_dict(feature_config, "features.json")

            # Log key metrics as params for easy filtering
            mlflow.log_param("n_features", feature_config.get("n_features", 0))
            mlflow.log_param(
                "feature_set_version",
                feature_config.get("feature_set_version", "unknown")
            )

            logger.info("Logged feature configuration")
        except Exception as e:
            logger.error(f"Error logging feature config: {e}")

    def log_training_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log training metrics (called each epoch/iteration)

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging training metrics: {e}")

    def log_validation_metrics(self, metrics: Dict[str, float]):
        """
        Log validation metrics from walk-forward or cross-validation

        Args:
            metrics: Dictionary of metric name -> value
        """
        try:
            # Add prefix for clarity
            val_metrics = {f"val_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(val_metrics)
            logger.info(f"Logged {len(val_metrics)} validation metrics")
        except Exception as e:
            logger.error(f"Error logging validation metrics: {e}")

    def log_betting_metrics(self, metrics: Dict[str, float]):
        """
        Log betting-specific metrics (ROI, win rate, CLV, etc.)

        Args:
            metrics: Dictionary of metric name -> value
        """
        try:
            # Add prefix for clarity
            betting_metrics = {f"betting_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(betting_metrics)
            logger.info(f"Logged {len(betting_metrics)} betting metrics")
        except Exception as e:
            logger.error(f"Error logging betting metrics: {e}")

    def log_segmented_metrics(self, segments: Dict[str, Dict[str, float]]):
        """
        Log metrics segmented by different categories
        (e.g., by player tier, edge size, season)

        Args:
            segments: Nested dict like {"tier1": {"mae": 3.2, "roi": 0.08}, ...}
        """
        try:
            for segment_name, metrics in segments.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{segment_name}_{metric_name}", value)

            logger.info(f"Logged metrics for {len(segments)} segments")
        except Exception as e:
            logger.error(f"Error logging segmented metrics: {e}")

    def log_model(
        self,
        model,
        model_type: str,
        signature=None,
        input_example=None,
        registered_model_name: str = None
    ):
        """
        Log trained model with MLflow

        Args:
            model: Trained model object
            model_type: Type of model ('xgboost', 'lightgbm', 'sklearn')
            signature: MLflow model signature
            input_example: Example input for model
            registered_model_name: Name to register model under
        """
        try:
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif model_type == 'lightgbm':
                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:  # sklearn or generic
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )

            logger.info(f"Logged {model_type} model")

            if registered_model_name:
                logger.info(f"Registered model as: {registered_model_name}")

        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 30
    ):
        """
        Log and plot feature importance

        Args:
            feature_names: List of feature names
            importance_values: Array of importance values
            top_n: Number of top features to plot
        """
        try:
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)

            # Save as CSV
            importance_df.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 12))
            top_features = importance_df.head(top_n)

            ax.barh(
                range(len(top_features)),
                top_features['importance'],
                align='center'
            )
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importances')
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close()

            logger.info(f"Logged feature importance for {len(feature_names)} features")

        except Exception as e:
            logger.error(f"Error logging feature importance: {e}")

    def log_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ):
        """
        Log calibration curve plot

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration plot
        """
        try:
            from sklearn.calibration import calibration_curve

            prob_true, prob_pred = calibration_curve(
                y_true,
                y_pred_proba,
                n_bins=n_bins,
                strategy='uniform'
            )

            fig, ax = plt.subplots(figsize=(10, 10))

            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.plot(prob_pred, prob_true, 's-', label='Model calibration')

            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Curve')
            ax.legend()
            ax.grid(alpha=0.3)

            plt.tight_layout()
            mlflow.log_figure(fig, "calibration_curve.png")
            plt.close()

            logger.info("Logged calibration curve")

        except Exception as e:
            logger.error(f"Error logging calibration curve: {e}")

    def log_predictions(
        self,
        predictions_df: pd.DataFrame,
        filename: str = "predictions.csv"
    ):
        """
        Log predictions as CSV artifact

        Args:
            predictions_df: DataFrame with predictions
            filename: Name for predictions file
        """
        try:
            predictions_df.to_csv(filename, index=False)
            mlflow.log_artifact(filename)
            logger.info(f"Logged predictions: {filename}")
        except Exception as e:
            logger.error(f"Error logging predictions: {e}")

    def log_residuals_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """
        Log residuals plot

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        try:
            residuals = y_true - y_pred

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Residuals vs Predicted
            axes[0, 0].scatter(y_pred, residuals, alpha=0.3)
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            axes[0, 0].grid(alpha=0.3)

            # Histogram of residuals
            axes[0, 1].hist(residuals, bins=50, edgecolor='black')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Residuals')
            axes[0, 1].grid(alpha=0.3)

            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(alpha=0.3)

            # Predicted vs Actual
            axes[1, 1].scatter(y_true, y_pred, alpha=0.3)
            axes[1, 1].plot([y_true.min(), y_true.max()],
                           [y_true.min(), y_true.max()],
                           'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Predicted vs Actual')
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            mlflow.log_figure(fig, "residuals_analysis.png")
            plt.close()

            logger.info("Logged residuals analysis")

        except Exception as e:
            logger.error(f"Error logging residuals plot: {e}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = None
    ):
        """
        Log confusion matrix for over/under predictions

        Args:
            y_true: True values (actual PRA)
            y_pred: Predicted values
            threshold: Line value to classify over/under (if None, uses median)
        """
        try:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns

            if threshold is None:
                threshold = np.median(y_true)

            y_true_binary = (y_true > threshold).astype(int)
            y_pred_binary = (y_pred > threshold).astype(int)

            cm = confusion_matrix(y_true_binary, y_pred_binary)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Under', 'Over'],
                yticklabels=['Under', 'Over'],
                ax=ax
            )
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            ax.set_title(f'Confusion Matrix (Threshold: {threshold:.1f})')

            plt.tight_layout()
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()

            # Log accuracy metrics
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

            mlflow.log_metrics({
                "classification_accuracy": accuracy,
                "classification_precision": precision,
                "classification_recall": recall,
                "classification_threshold": threshold
            })

            logger.info("Logged confusion matrix and classification metrics")

        except Exception as e:
            logger.error(f"Error logging confusion matrix: {e}")

    def log_error_analysis(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        y_pred_col: str,
        group_cols: List[str] = None
    ):
        """
        Log error analysis report

        Args:
            df: DataFrame with predictions and actual values
            y_true_col: Column name for true values
            y_pred_col: Column name for predicted values
            group_cols: Columns to group by for analysis (e.g., ['Player', 'Season'])
        """
        try:
            df = df.copy()
            df['error'] = df[y_true_col] - df[y_pred_col]
            df['abs_error'] = df['error'].abs()
            df['squared_error'] = df['error'] ** 2

            # Overall statistics
            overall_stats = {
                'mean_error': df['error'].mean(),
                'mae': df['abs_error'].mean(),
                'rmse': np.sqrt(df['squared_error'].mean()),
                'median_abs_error': df['abs_error'].median(),
                'std_error': df['error'].std(),
            }

            # Group-wise analysis
            if group_cols:
                group_stats = df.groupby(group_cols).agg({
                    'error': 'mean',
                    'abs_error': 'mean',
                    'squared_error': lambda x: np.sqrt(x.mean())
                }).reset_index()

                group_stats.columns = group_cols + ['mean_error', 'mae', 'rmse']
                group_stats = group_stats.sort_values('mae')

                # Save group analysis
                group_stats.to_csv("error_analysis_by_group.csv", index=False)
                mlflow.log_artifact("error_analysis_by_group.csv")

            # Save overall analysis
            with open("error_analysis.json", 'w') as f:
                json.dump(overall_stats, f, indent=2)
            mlflow.log_artifact("error_analysis.json")

            logger.info("Logged error analysis")

        except Exception as e:
            logger.error(f"Error logging error analysis: {e}")

    def log_training_config(self, config: Dict[str, Any]):
        """
        Log training configuration

        Args:
            config: Dictionary with training configuration
        """
        try:
            mlflow.log_dict(config, "training_config.json")

            # Log key config items as params
            for key in ['train_seasons', 'validation_type', 'test_season']:
                if key in config:
                    value = config[key]
                    if isinstance(value, list):
                        value = ','.join(map(str, value))
                    mlflow.log_param(key, value)

            logger.info("Logged training configuration")

        except Exception as e:
            logger.error(f"Error logging training config: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run

        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')
        """
        try:
            mlflow.set_tag("end_time", datetime.now().isoformat())
            mlflow.set_tag("status", status)
            mlflow.end_run()
            logger.info(f"Ended run: {self.run_name} (Status: {status})")
        except Exception as e:
            logger.error(f"Error ending run: {e}")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        if exc_type is not None:
            self.end_run(status="FAILED")
            logger.error(f"Run failed with error: {exc_val}")
        else:
            self.end_run(status="FINISHED")


def enable_autologging(model_type: str = 'xgboost'):
    """
    Enable MLflow autologging for XGBoost or LightGBM

    Args:
        model_type: 'xgboost' or 'lightgbm'
    """
    try:
        if model_type == 'xgboost':
            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                silent=False
            )
            logger.info("Enabled XGBoost autologging")

        elif model_type == 'lightgbm':
            mlflow.lightgbm.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                disable=False,
                exclusive=False,
                silent=False
            )
            logger.info("Enabled LightGBM autologging")

        else:
            logger.warning(f"Autologging not available for {model_type}")

    except Exception as e:
        logger.error(f"Error enabling autologging: {e}")
