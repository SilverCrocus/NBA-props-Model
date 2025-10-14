"""
Test MLflow integration with a simple training run
Run this to verify MLflow is working correctly
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mlflow_integration.tracker import NBAPropsTracker, enable_autologging
from mlflow_integration.registry import ModelRegistry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_tracking():
    """Test basic MLflow tracking"""
    print("=" * 80)
    print("TEST 1: Basic Tracking")
    print("=" * 80)
    print()

    # Create synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Initialize tracker
    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")

    # Start run
    tracker.start_run(
        run_name="test_basic_tracking",
        tags={"test": "true", "model_type": "xgboost"}
    )

    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1
    }
    tracker.log_params(params)

    # Train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    train_metrics = {
        'mae': mean_absolute_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'r2': r2_score(y_train, y_pred_train)
    }

    test_metrics = {
        'mae': mean_absolute_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2': r2_score(y_test, y_pred_test)
    }

    # Log metrics
    tracker.log_training_metrics(train_metrics)
    tracker.log_validation_metrics(test_metrics)

    # Log feature importance
    tracker.log_feature_importance(
        feature_names=[f'feature_{i}' for i in range(20)],
        importance_values=model.feature_importances_
    )

    # Log residuals
    tracker.log_residuals_plot(y_test, y_pred_test)

    # Log model
    tracker.log_model(model, model_type='xgboost')

    # End run
    tracker.end_run()

    print(f"✓ Test passed! Run ID: {tracker.run_id}")
    print()


def test_autologging():
    """Test XGBoost autologging"""
    print("=" * 80)
    print("TEST 2: Autologging")
    print("=" * 80)
    print()

    # Create synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    # Initialize tracker
    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")

    # Enable autologging
    enable_autologging('xgboost')

    # Start run
    tracker.start_run(
        run_name="test_autologging",
        tags={"test": "true", "autolog": "true"}
    )

    # Train model (autologging handles everything)
    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    # End run
    tracker.end_run()

    print(f"✓ Test passed! Run ID: {tracker.run_id}")
    print("  Check MLflow UI to see autologged parameters and metrics")
    print()


def test_model_registry():
    """Test model registry functionality"""
    print("=" * 80)
    print("TEST 3: Model Registry")
    print("=" * 80)
    print()

    # Create synthetic data and train model
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

    tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
    tracker.start_run(
        run_name="test_registry",
        tags={"test": "true"}
    )

    # Train model
    model = XGBRegressor(n_estimators=50, max_depth=3)
    model.fit(X, y)

    # Log model
    tracker.log_model(
        model,
        model_type='xgboost',
        registered_model_name="NBAPropsModelTest"
    )

    run_id = tracker.run_id
    tracker.end_run()

    # Test registry operations
    registry = ModelRegistry()

    # Get model info
    try:
        model_info = registry.get_model_info("NBAPropsModelTest")
        print(f"✓ Model registered successfully!")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Version: {model_info['version']}")
        print(f"  Stage: {model_info['stage']}")
        print()
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        print()


def test_comparison():
    """Test experiment comparison"""
    print("=" * 80)
    print("TEST 4: Experiment Comparison")
    print("=" * 80)
    print()

    from mlflow_integration.utils import get_best_model, generate_experiment_report

    # Get best model
    best_models = get_best_model(
        experiment_name="Phase1_Foundation",
        metric="val_mae",
        ascending=True,
        top_n=3
    )

    if best_models:
        print("✓ Found best models:")
        for i, model in enumerate(best_models, 1):
            print(f"  {i}. {model.get('run_name')} - {model.get('run_id')[:8]}")
        print()
    else:
        print("  No models found yet (this is expected for first run)")
        print()


def run_all_tests():
    """Run all tests"""
    print()
    print("=" * 80)
    print("MLFLOW INTEGRATION TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_basic_tracking()
        test_autologging()
        test_model_registry()
        test_comparison()

        print("=" * 80)
        print("ALL TESTS COMPLETED!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("  1. Start MLflow UI: uv run mlflow ui")
        print("  2. Navigate to http://localhost:5000")
        print("  3. Explore your test runs in the Phase1_Foundation experiment")
        print()

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
