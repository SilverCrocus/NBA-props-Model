"""
MLflow Model Registry Manager for NBA Props Model
Handles model registration, versioning, promotion, and lifecycle management
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages MLflow Model Registry for NBA props models
    Handles registration, promotion, and lifecycle management
    """

    def __init__(self, tracking_uri: str = None):
        """
        Initialize Model Registry manager

        Args:
            tracking_uri: MLflow tracking URI (defaults to local)
        """
        if tracking_uri is None:
            tracking_uri = "file:///Users/diyagamah/Documents/nba_props_model/mlruns"

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        logger.info(f"Initialized Model Registry with URI: {tracking_uri}")

    def register_model(
        self,
        run_id: str,
        model_name: str = "NBAPropsModel",
        tags: Dict[str, str] = None
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Register a model from a completed run

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name to register model under
            tags: Additional tags for the model version

        Returns:
            ModelVersion object
        """
        try:
            model_uri = f"runs:/{run_id}/model"

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            logger.info(
                f"Registered model '{model_name}' version {model_version.version} "
                f"from run {run_id}"
            )

            # Add tags
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=str(value)
                    )

            # Add registration timestamp
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="registered_at",
                value=datetime.now().isoformat()
            )

            return model_version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def get_model_version(
        self,
        model_name: str,
        version: int = None,
        stage: str = None
    ) -> mlflow.entities.model_registry.ModelVersion:
        """
        Get a specific model version

        Args:
            model_name: Name of the registered model
            version: Specific version number (if None, gets latest)
            stage: Get latest version in specific stage (e.g., 'Production')

        Returns:
            ModelVersion object
        """
        try:
            if version is not None:
                return self.client.get_model_version(model_name, version)

            elif stage is not None:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if versions:
                    return versions[0]
                else:
                    raise ValueError(f"No model in stage '{stage}'")

            else:
                # Get latest version
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if versions:
                    return max(versions, key=lambda v: int(v.version))
                else:
                    raise ValueError(f"No versions found for model '{model_name}'")

        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            raise

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Promote a model version to a specific stage

        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing: Archive existing models in target stage
        """
        try:
            # Archive existing models in target stage
            if archive_existing and stage in ['Staging', 'Production']:
                existing = self.client.get_latest_versions(model_name, stages=[stage])
                for model_version in existing:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Archived"
                    )
                    logger.info(
                        f"Archived model version {model_version.version} "
                        f"from {stage}"
                    )

            # Promote new version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )

            # Add promotion timestamp
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"promoted_to_{stage.lower()}_at",
                value=datetime.now().isoformat()
            )

            logger.info(f"Promoted model version {version} to {stage}")

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            raise

    def evaluate_for_production(
        self,
        model_name: str,
        version: int,
        criteria: Dict[str, float]
    ) -> bool:
        """
        Evaluate if model meets production criteria

        Args:
            model_name: Name of the registered model
            version: Version to evaluate
            criteria: Dictionary of metric thresholds

        Returns:
            True if model meets all criteria, False otherwise
        """
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)

            metrics = run.data.metrics
            params = run.data.params

            results = {}
            all_passed = True

            for metric_name, threshold in criteria.items():
                if metric_name.startswith('min_'):
                    # Minimum threshold (e.g., min_win_rate > 0.55)
                    actual_metric = metric_name.replace('min_', '')
                    actual_value = metrics.get(actual_metric)

                    if actual_value is None:
                        logger.warning(f"Metric '{actual_metric}' not found in run")
                        results[metric_name] = False
                        all_passed = False
                    elif actual_value < threshold:
                        logger.info(
                            f"FAIL: {actual_metric} ({actual_value:.4f}) "
                            f"< {threshold}"
                        )
                        results[metric_name] = False
                        all_passed = False
                    else:
                        logger.info(
                            f"PASS: {actual_metric} ({actual_value:.4f}) "
                            f">= {threshold}"
                        )
                        results[metric_name] = True

                elif metric_name.startswith('max_'):
                    # Maximum threshold (e.g., max_mae < 3.5)
                    actual_metric = metric_name.replace('max_', '')
                    actual_value = metrics.get(actual_metric)

                    if actual_value is None:
                        logger.warning(f"Metric '{actual_metric}' not found in run")
                        results[metric_name] = False
                        all_passed = False
                    elif actual_value > threshold:
                        logger.info(
                            f"FAIL: {actual_metric} ({actual_value:.4f}) "
                            f"> {threshold}"
                        )
                        results[metric_name] = False
                        all_passed = False
                    else:
                        logger.info(
                            f"PASS: {actual_metric} ({actual_value:.4f}) "
                            f"<= {threshold}"
                        )
                        results[metric_name] = True

            # Add evaluation results as tags
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="production_evaluation",
                value="PASSED" if all_passed else "FAILED"
            )

            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="evaluation_timestamp",
                value=datetime.now().isoformat()
            )

            return all_passed

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return False

    def rollback_model(
        self,
        model_name: str,
        reason: str,
        target_version: int = None
    ):
        """
        Rollback to previous production model

        Args:
            model_name: Name of the registered model
            reason: Reason for rollback
            target_version: Specific version to rollback to (if None, uses previous)
        """
        try:
            # Get current production model
            current_prod = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )

            if not current_prod:
                logger.warning("No production model to rollback from")
                return

            current_version = current_prod[0]

            # Archive current production model
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_version.version,
                stage="Archived"
            )

            # Add rollback reason
            self.client.set_model_version_tag(
                name=model_name,
                version=current_version.version,
                key="rollback_reason",
                value=reason
            )

            logger.info(f"Archived production model v{current_version.version}")

            # Determine target version
            if target_version is None:
                # Get previous version
                all_versions = self.client.search_model_versions(
                    f"name='{model_name}'"
                )
                sorted_versions = sorted(
                    all_versions,
                    key=lambda v: int(v.version),
                    reverse=True
                )

                # Find first version that was previously in production
                target_version = None
                for v in sorted_versions:
                    if v.version != current_version.version:
                        tags = {t.key: t.value for t in v.tags}
                        if 'promoted_to_production_at' in tags:
                            target_version = int(v.version)
                            break

                if target_version is None:
                    logger.error("No previous production version found")
                    return

            # Promote target version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )

            # Add rollback timestamp
            self.client.set_model_version_tag(
                name=model_name,
                version=target_version,
                key="rolled_back_at",
                value=datetime.now().isoformat()
            )

            logger.info(
                f"Rolled back from v{current_version.version} to v{target_version}"
            )
            logger.info(f"Reason: {reason}")

        except Exception as e:
            logger.error(f"Error rolling back model: {e}")
            raise

    def list_models(
        self,
        model_name: str = None,
        stage: str = None
    ) -> List[mlflow.entities.model_registry.ModelVersion]:
        """
        List registered models

        Args:
            model_name: Filter by model name (if None, lists all)
            stage: Filter by stage (e.g., 'Production', 'Staging')

        Returns:
            List of ModelVersion objects
        """
        try:
            if model_name and stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            elif model_name:
                versions = self.client.search_model_versions(f"name='{model_name}'")
            else:
                # List all registered models
                versions = []
                for rm in self.client.search_registered_models():
                    versions.extend(
                        self.client.search_model_versions(f"name='{rm.name}'")
                    )

            return versions

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_model_info(
        self,
        model_name: str,
        version: int = None
    ) -> Dict:
        """
        Get detailed information about a model version

        Args:
            model_name: Name of the registered model
            version: Version number (if None, gets production version)

        Returns:
            Dictionary with model information
        """
        try:
            if version is None:
                model_version = self.get_model_version(
                    model_name,
                    stage="Production"
                )
            else:
                model_version = self.client.get_model_version(model_name, version)

            run = self.client.get_run(model_version.run_id)

            info = {
                "model_name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "tags": {tag.key: tag.value for tag in model_version.tags},
                "metrics": run.data.metrics,
                "params": run.data.params,
            }

            return info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    def compare_models(
        self,
        model_name: str,
        version1: int,
        version2: int,
        metrics: List[str] = None
    ) -> Dict:
        """
        Compare two model versions

        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            metrics: List of metrics to compare (if None, compares all)

        Returns:
            Dictionary with comparison results
        """
        try:
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)

            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)

            if metrics is None:
                # Get all common metrics
                metrics = set(run1.data.metrics.keys()) & set(run2.data.metrics.keys())

            comparison = {
                "version1": version1,
                "version2": version2,
                "metrics": {}
            }

            for metric in metrics:
                val1 = run1.data.metrics.get(metric)
                val2 = run2.data.metrics.get(metric)

                if val1 is not None and val2 is not None:
                    comparison["metrics"][metric] = {
                        "version1": val1,
                        "version2": val2,
                        "difference": val2 - val1,
                        "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else None
                    }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}

    def delete_model_version(
        self,
        model_name: str,
        version: int
    ):
        """
        Delete a specific model version

        Args:
            model_name: Name of the registered model
            version: Version to delete
        """
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model version {version}")
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            raise


# Production criteria for NBA props model
DEFAULT_PRODUCTION_CRITERIA = {
    "max_val_mae": 3.5,                # MAE must be < 3.5
    "min_betting_roi": 0.05,           # ROI must be > 5%
    "min_betting_win_rate": 0.55,      # Win rate must be > 55%
    "max_betting_calibration_error": 0.05,  # Calibration error < 5%
    "min_betting_sharpe_ratio": 1.0,   # Sharpe ratio > 1.0
}
