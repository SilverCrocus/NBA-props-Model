"""
Setup MLflow experiment structure for NBA props model
Run this once at the start of the project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mlflow_integration.utils import setup_mlflow_experiments
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Set up all experiments for the 12-week development"""

    print("=" * 80)
    print("MLFLOW EXPERIMENT SETUP - NBA PROPS MODEL")
    print("=" * 80)
    print()

    tracking_uri = "file:///Users/diyagamah/Documents/nba_props_model/mlruns"

    logger.info(f"Setting up experiments with tracking URI: {tracking_uri}")

    # Create experiment structure
    experiment_ids = setup_mlflow_experiments(tracking_uri)

    print("\nExperiment structure created successfully!\n")
    print("Experiments:")
    print("-" * 80)

    for exp_name, exp_id in experiment_ids.items():
        print(f"  {exp_name}: {exp_id}")

    print()
    print("Next steps:")
    print("  1. Start MLflow UI: uv run mlflow ui")
    print("  2. Navigate to http://localhost:5000")
    print("  3. Start training models!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
