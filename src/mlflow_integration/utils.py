"""
Utility functions for MLflow experiment tracking
Experiment setup, comparison, reporting, and analysis
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow_experiments(tracking_uri: str = None) -> Dict[str, str]:
    """
    Set up experiment structure for NBA props model development

    Args:
        tracking_uri: MLflow tracking URI (defaults to local)

    Returns:
        Dictionary mapping phase names to experiment IDs
    """
    if tracking_uri is None:
        tracking_uri = "file:///Users/diyagamah/Documents/nba_props_model/mlruns"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Define experiment structure
    experiments = [
        {
            "name": "Phase1_Foundation",
            "description": "Weeks 1-3: Baseline models, feature selection, initial validation",
            "tags": {"phase": "1", "weeks": "1-3", "focus": "foundation"}
        },
        {
            "name": "Phase2_Advanced",
            "description": "Weeks 4-7: Advanced features, opponent features, ensemble methods",
            "tags": {"phase": "2", "weeks": "4-7", "focus": "advanced_features"}
        },
        {
            "name": "Phase3_Calibration",
            "description": "Weeks 8-9: Model calibration, probability adjustment, betting optimization",
            "tags": {"phase": "3", "weeks": "8-9", "focus": "calibration"}
        },
        {
            "name": "Phase4_Production",
            "description": "Weeks 10-12: Walk-forward validation, production deployment, monitoring",
            "tags": {"phase": "4", "weeks": "10-12", "focus": "production"}
        },
    ]

    experiment_ids = {}

    for exp_config in experiments:
        try:
            # Check if experiment already exists
            experiment = client.get_experiment_by_name(exp_config["name"])

            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(
                    name=exp_config["name"],
                    artifact_location=f"mlflow_artifacts/{exp_config['name']}",
                    tags=exp_config["tags"]
                )
                logger.info(f"Created experiment: {exp_config['name']}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {exp_config['name']}")

            experiment_ids[exp_config["name"]] = experiment_id

        except Exception as e:
            logger.error(f"Error setting up experiment {exp_config['name']}: {e}")

    return experiment_ids


def compare_runs(
    run_ids: List[str],
    metrics: List[str] = None,
    params: List[str] = None
) -> pd.DataFrame:
    """
    Compare multiple MLflow runs

    Args:
        run_ids: List of run IDs to compare
        metrics: List of metrics to compare (if None, includes all)
        params: List of parameters to compare (if None, includes all)

    Returns:
        DataFrame with comparison
    """
    client = MlflowClient()
    results = []

    for run_id in run_ids:
        try:
            run = client.get_run(run_id)

            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "experiment_id": run.info.experiment_id,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "status": run.info.status,
            }

            # Add metrics
            if metrics is None:
                metrics_to_add = run.data.metrics.keys()
            else:
                metrics_to_add = metrics

            for metric in metrics_to_add:
                run_data[f"metric_{metric}"] = run.data.metrics.get(metric)

            # Add params
            if params is None:
                params_to_add = run.data.params.keys()
            else:
                params_to_add = params

            for param in params_to_add:
                run_data[f"param_{param}"] = run.data.params.get(param)

            results.append(run_data)

        except Exception as e:
            logger.error(f"Error getting run {run_id}: {e}")

    return pd.DataFrame(results)


def get_best_model(
    experiment_name: str,
    metric: str = "val_mae",
    ascending: bool = True,
    filter_string: str = None,
    top_n: int = 1
) -> List[Dict]:
    """
    Get best model(s) from an experiment

    Args:
        experiment_name: Name of experiment
        metric: Metric to optimize (e.g., 'val_mae', 'betting_roi')
        ascending: True if lower is better, False if higher is better
        filter_string: MLflow filter string (e.g., "metrics.val_mae < 3.5")
        top_n: Number of top models to return

    Returns:
        List of dictionaries with model information
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return []

        # Search runs
        order_by = [f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by,
            max_results=top_n
        )

        if runs_df.empty:
            logger.warning(f"No runs found in experiment '{experiment_name}'")
            return []

        results = []
        for _, row in runs_df.iterrows():
            result = {
                "run_id": row.get("run_id"),
                "run_name": row.get("tags.mlflow.runName", ""),
                "metric_value": row.get(f"metrics.{metric}"),
                "start_time": row.get("start_time"),
            }

            # Add other relevant metrics
            for col in row.index:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    result[metric_name] = row[col]
                elif col.startswith("params."):
                    param_name = col.replace("params.", "")
                    result[param_name] = row[col]

            results.append(result)

        return results

    except Exception as e:
        logger.error(f"Error getting best model: {e}")
        return []


def generate_experiment_report(
    experiment_name: str,
    output_path: str = None
) -> str:
    """
    Generate comprehensive report for an experiment

    Args:
        experiment_name: Name of experiment
        output_path: Path to save report (if None, returns string)

    Returns:
        Report as string
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return f"Experiment '{experiment_name}' not found"

        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        report = []
        report.append("=" * 80)
        report.append(f"MLFLOW EXPERIMENT REPORT: {experiment_name}")
        report.append("=" * 80)
        report.append("")

        # Experiment overview
        report.append("EXPERIMENT OVERVIEW")
        report.append("-" * 40)
        report.append(f"Experiment ID: {experiment.experiment_id}")
        report.append(f"Total Runs: {len(runs_df)}")
        report.append(f"Artifact Location: {experiment.artifact_location}")
        report.append("")

        if not runs_df.empty:
            # Status summary
            status_counts = runs_df['status'].value_counts()
            report.append("RUN STATUS")
            report.append("-" * 40)
            for status, count in status_counts.items():
                report.append(f"{status}: {count}")
            report.append("")

            # Best models by metric
            key_metrics = [
                ("val_mae", True, "Validation MAE"),
                ("betting_roi", False, "Betting ROI"),
                ("betting_win_rate", False, "Win Rate"),
            ]

            for metric, ascending, display_name in key_metrics:
                metric_col = f"metrics.{metric}"
                if metric_col in runs_df.columns:
                    runs_with_metric = runs_df[runs_df[metric_col].notna()]
                    if not runs_with_metric.empty:
                        if ascending:
                            best = runs_with_metric.nsmallest(1, metric_col).iloc[0]
                        else:
                            best = runs_with_metric.nlargest(1, metric_col).iloc[0]

                        report.append(f"BEST {display_name.upper()}")
                        report.append("-" * 40)
                        report.append(f"Run ID: {best['run_id']}")
                        report.append(f"Run Name: {best.get('tags.mlflow.runName', 'N/A')}")
                        report.append(f"{display_name}: {best[metric_col]:.4f}")
                        report.append("")

            # Metric statistics
            metric_cols = [col for col in runs_df.columns if col.startswith("metrics.")]
            if metric_cols:
                report.append("METRIC STATISTICS")
                report.append("-" * 40)

                for col in metric_cols[:10]:  # Limit to top 10 metrics
                    metric_name = col.replace("metrics.", "")
                    values = runs_df[col].dropna()

                    if len(values) > 0:
                        report.append(f"{metric_name}:")
                        report.append(f"  Mean: {values.mean():.4f}")
                        report.append(f"  Std: {values.std():.4f}")
                        report.append(f"  Min: {values.min():.4f}")
                        report.append(f"  Max: {values.max():.4f}")

                report.append("")

            # Recent runs
            report.append("RECENT RUNS (Last 5)")
            report.append("-" * 40)

            recent = runs_df.nlargest(5, 'start_time')
            for _, run in recent.iterrows():
                run_name = run.get('tags.mlflow.runName', 'N/A')
                start_time = run['start_time'].strftime('%Y-%m-%d %H:%M:%S')
                status = run['status']

                report.append(f"- {run_name} ({start_time}) - {status}")

                # Show key metrics if available
                if 'metrics.val_mae' in run:
                    report.append(f"  MAE: {run['metrics.val_mae']:.4f}")
                if 'metrics.betting_roi' in run:
                    report.append(f"  ROI: {run['metrics.betting_roi']:.4f}")

            report.append("")

        report.append("=" * 80)

        report_text = "\n".join(report)

        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return f"Error: {e}"


def get_weekly_summary(
    week_number: int,
    experiment_names: List[str] = None
) -> Dict:
    """
    Generate weekly progress summary

    Args:
        week_number: Week number (1-12)
        experiment_names: List of experiment names to include (if None, includes all)

    Returns:
        Dictionary with weekly summary
    """
    try:
        # Calculate date range for this week
        # Assuming project started on 2025-10-14
        project_start = datetime(2025, 10, 14)
        week_start = project_start + timedelta(weeks=week_number - 1)
        week_end = week_start + timedelta(days=7)

        client = MlflowClient()

        # Get experiments
        if experiment_names is None:
            experiments = client.search_experiments()
        else:
            experiments = [
                client.get_experiment_by_name(name)
                for name in experiment_names
                if client.get_experiment_by_name(name) is not None
            ]

        summary = {
            "week": week_number,
            "date_range": f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
            "experiments": {},
            "totals": {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "models_registered": 0,
            }
        }

        for exp in experiments:
            # Get runs for this week
            filter_string = (
                f"attributes.start_time >= {int(week_start.timestamp() * 1000)} "
                f"AND attributes.start_time < {int(week_end.timestamp() * 1000)}"
            )

            runs_df = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=filter_string
            )

            exp_summary = {
                "total_runs": len(runs_df),
                "successful": len(runs_df[runs_df['status'] == 'FINISHED']),
                "failed": len(runs_df[runs_df['status'] == 'FAILED']),
            }

            # Get best metrics
            if not runs_df.empty:
                if 'metrics.val_mae' in runs_df.columns:
                    exp_summary['best_mae'] = runs_df['metrics.val_mae'].min()
                if 'metrics.betting_roi' in runs_df.columns:
                    exp_summary['best_roi'] = runs_df['metrics.betting_roi'].max()

            summary["experiments"][exp.name] = exp_summary

            # Update totals
            summary["totals"]["total_runs"] += exp_summary["total_runs"]
            summary["totals"]["successful_runs"] += exp_summary["successful"]
            summary["totals"]["failed_runs"] += exp_summary["failed"]

        return summary

    except Exception as e:
        logger.error(f"Error generating weekly summary: {e}")
        return {}


def compare_phases() -> pd.DataFrame:
    """
    Compare best models from each development phase

    Returns:
        DataFrame with phase comparison
    """
    phases = [
        "Phase1_Foundation",
        "Phase2_Advanced",
        "Phase3_Calibration",
        "Phase4_Production"
    ]

    results = []

    for phase_name in phases:
        best_models = get_best_model(
            experiment_name=phase_name,
            metric="val_mae",
            ascending=True,
            top_n=1
        )

        if best_models:
            model = best_models[0]
            result = {
                "phase": phase_name,
                "run_id": model.get("run_id"),
                "run_name": model.get("run_name"),
                "val_mae": model.get("val_mae"),
                "betting_roi": model.get("betting_roi"),
                "betting_win_rate": model.get("betting_win_rate"),
            }
            results.append(result)

    return pd.DataFrame(results)


def export_runs_to_csv(
    experiment_name: str,
    output_path: str,
    filter_string: str = None
):
    """
    Export experiment runs to CSV

    Args:
        experiment_name: Name of experiment
        output_path: Path to save CSV
        filter_string: MLflow filter string
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment '{experiment_name}' not found")
            return

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string
        )

        runs_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(runs_df)} runs to {output_path}")

    except Exception as e:
        logger.error(f"Error exporting runs: {e}")
