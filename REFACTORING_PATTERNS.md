# NBA Props Model: Refactoring Patterns & Best Practices

**Date:** October 15, 2025
**Purpose:** Practical patterns for refactoring the ML pipeline
**Audience:** Developers implementing the refactoring

---

## Table of Contents

1. [Feature Engineering Patterns](#1-feature-engineering-patterns)
2. [Pipeline Stage Patterns](#2-pipeline-stage-patterns)
3. [Model Wrapper Patterns](#3-model-wrapper-patterns)
4. [Configuration Patterns](#4-configuration-patterns)
5. [Testing Patterns](#5-testing-patterns)
6. [MLflow Patterns](#6-mlflow-patterns)
7. [Migration Patterns](#7-migration-patterns)

---

## 1. Feature Engineering Patterns

### Pattern 1.1: Feature Definition as Class

**Problem:** Feature logic duplicated across multiple files.

**Solution:** Define each feature as a reusable class.

```python
# ❌ BEFORE: Inline feature calculation
def calculate_lag_features(player_history, lags=[1, 3, 5, 7]):
    features = {}
    history = player_history.sort_values("GAME_DATE", ascending=False)
    for lag in lags:
        if len(history) >= lag:
            features[f"PRA_lag{lag}"] = history.iloc[lag - 1]["PRA"]
    return features

# ✅ AFTER: Reusable feature class
class LagFeature(BaseFeature):
    def __init__(self, stat: str, lag: int, name: str = None):
        if name is None:
            name = f"{stat}_lag{lag}"
        super().__init__(name, version="1.0")
        self.stat = stat
        self.lag = lag

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute with .shift() to prevent leakage."""
        return df.groupby('PLAYER_ID')[self.stat].shift(self.lag)
```

**Benefits:**
- Single implementation (vs 4 copies)
- Built-in validation
- Versioning support
- Testable in isolation

---

### Pattern 1.2: Feature Composition

**Problem:** Complex features require combining multiple calculations.

**Solution:** Compose features from simpler features.

```python
# ❌ BEFORE: Monolithic calculation
def calculate_per_36_stats(player_history):
    min_total = player_history.get("MIN").sum()
    pra_total = player_history["PRA"].sum()
    pts_total = player_history["PTS"].sum()
    return {
        "PRA_per_36": (pra_total / min_total) * 36,
        "PTS_per_36": (pts_total / min_total) * 36,
    }

# ✅ AFTER: Composable features
class PerMinuteFeature(BaseFeature):
    """Base class for per-minute normalization."""
    def __init__(self, stat: str, minutes: int = 36):
        name = f"{stat}_per_{minutes}"
        super().__init__(name, version="1.0")
        self.stat = stat
        self.minutes = minutes
        self.dependencies = [stat, 'MIN']

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute stat per N minutes."""
        stat_sum = df.groupby('PLAYER_ID')[self.stat].transform('sum')
        min_sum = df.groupby('PLAYER_ID')['MIN'].transform('sum')
        return (stat_sum / min_sum) * self.minutes

# Usage
store.register(PerMinuteFeature('PRA', minutes=36))
store.register(PerMinuteFeature('PTS', minutes=36))
store.register(PerMinuteFeature('REB', minutes=36))
```

**Benefits:**
- DRY (Don't Repeat Yourself)
- Easy to add new per-minute features
- Dependencies tracked automatically

---

### Pattern 1.3: Feature Group Registry

**Problem:** Hard to manage which features to use in experiments.

**Solution:** Organize features into named groups.

```python
# ❌ BEFORE: Hardcoded feature lists
feature_cols = [
    'PRA_lag1', 'PRA_lag3', 'PRA_lag5',
    'PRA_L5_mean', 'PRA_L10_mean',
    'PRA_ewma5', 'PRA_ewma10',
    # ... 50+ features
]

# ✅ AFTER: Named feature groups
store = FeatureStore()

# Register temporal features
for lag in [1, 3, 5, 7]:
    store.register(LagFeature('PRA', lag), group='temporal')

for window in [5, 10, 20]:
    store.register(RollingFeature('PRA', window, 'mean'), group='temporal')

for span in [5, 10]:
    store.register(EWMAFeature('PRA', span), group='temporal')

# Register efficiency features
store.register(TSPctFeature(), group='efficiency')
store.register(PERFeature(), group='efficiency')

# Register opponent features
store.register(OpponentDRtgFeature(), group='contextual')

# Use in experiments
temporal_features = store.compute_features(
    df,
    feature_names=store.get_feature_group('temporal')
)

all_features = store.compute_features(
    df,
    feature_names=(
        store.get_feature_group('temporal') +
        store.get_feature_group('efficiency') +
        store.get_feature_group('contextual')
    )
)
```

**Benefits:**
- Easy to experiment with feature combinations
- Clear organization
- Config-driven feature selection

---

### Pattern 1.4: Feature Validation

**Problem:** No automated checks for feature quality.

**Solution:** Built-in validation in feature classes.

```python
class BaseFeature(ABC):
    def validate(self, series: pd.Series) -> bool:
        """Validate computed feature."""
        # Check for inf values
        if np.isinf(series).any():
            logger.warning(f"Feature {self.name} contains inf values")
            return False

        # Check for all NaN
        if series.isna().all():
            logger.warning(f"Feature {self.name} is all NaN")
            return False

        # Check variance
        if series.std() < 1e-7:
            logger.warning(f"Feature {self.name} has zero variance")
            return False

        return True

class LagFeature(BaseFeature):
    def validate(self, series: pd.Series) -> bool:
        """Additional validation for lag features."""
        if not super().validate(series):
            return False

        # First lag values should be NaN (no previous game)
        # This ensures .shift() was used correctly
        if self.lag == 1:
            first_values = series.groupby(level=0).first()
            if not first_values.isna().any():
                logger.warning(
                    f"Lag feature {self.name} has no NaN in first positions. "
                    "Possible data leakage!"
                )
                return False

        return True
```

**Benefits:**
- Catch data leakage automatically
- Ensure feature quality
- Early error detection

---

## 2. Pipeline Stage Patterns

### Pattern 2.1: Stage Input/Output Validation

**Problem:** Stages fail silently or with unclear errors.

**Solution:** Explicit validation at stage boundaries.

```python
# ❌ BEFORE: No validation
def build_training_dataset():
    df = load_game_logs()
    df = add_features(df)
    df = split_train_val(df)
    return df

# ✅ AFTER: Validated stages
class DataLoadingStage(PipelineStage):
    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate stage can run."""
        # Check config
        if 'game_logs_path' not in self.config:
            logger.error("Missing game_logs_path in config")
            return False
        return True

    def validate_outputs(self, context: PipelineContext) -> bool:
        """Validate stage produced correct output."""
        df = context.artifacts.get('game_logs')

        # Check data loaded
        if df is None or len(df) == 0:
            logger.error("No game logs loaded")
            return False

        # Check required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE', 'PRA']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False

        # Check date range
        date_range = df['GAME_DATE'].max() - df['GAME_DATE'].min()
        if date_range.days < 30:
            logger.error(f"Date range too short: {date_range.days} days")
            return False

        return True

    def run(self, context: PipelineContext) -> PipelineContext:
        """Load data with error handling."""
        try:
            df = pd.read_csv(self.config['game_logs_path'])
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            context.artifacts['game_logs'] = df
            return context
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
```

**Benefits:**
- Clear error messages
- Fail fast (don't waste time on invalid data)
- Self-documenting (validation shows requirements)

---

### Pattern 2.2: Stage Configuration

**Problem:** Hard to configure stages for different experiments.

**Solution:** Config-driven stage behavior.

```python
# ❌ BEFORE: Hardcoded parameters
class FeatureEngineeringStage:
    def run(self, context):
        # Hardcoded feature list
        features = calculate_lag_features(df, lags=[1, 3, 5])
        features.update(calculate_rolling_features(df, windows=[5, 10]))
        return features

# ✅ AFTER: Config-driven
class FeatureEngineeringStage(PipelineStage):
    def run(self, context: PipelineContext) -> PipelineContext:
        """Compute features based on config."""
        df = context.artifacts['game_logs']

        # Get feature groups from config
        feature_groups = self.config.get('feature_groups', ['temporal'])

        # Compute features
        feature_names = []
        for group in feature_groups:
            feature_names.extend(self.feature_store.get_feature_group(group))

        features_df = self.feature_store.compute_features(df, feature_names)

        context.artifacts['features'] = pd.concat([df, features_df], axis=1)
        return context

# Usage with different configs
# Experiment 1: Only temporal features
config1 = {'feature_groups': ['temporal']}

# Experiment 2: All features
config2 = {'feature_groups': ['temporal', 'efficiency', 'contextual']}

# Experiment 3: Custom feature combination
config3 = {'feature_groups': ['temporal', 'position_defense']}
```

**Benefits:**
- Easy to experiment
- Config defines behavior
- Same code, different experiments

---

### Pattern 2.3: Stage Metrics Tracking

**Problem:** Hard to debug which stage is slow or failing.

**Solution:** Automatic timing and metrics at each stage.

```python
class PipelineStage(ABC):
    def run_with_metrics(self, context: PipelineContext) -> PipelineContext:
        """Wrapper that adds automatic metrics tracking."""
        import time

        logger.info(f"Starting stage: {self.name}")
        start_time = time.time()

        # Validate inputs
        if not self.validate_inputs(context):
            raise ValueError(f"Input validation failed: {self.name}")

        # Run stage
        try:
            result_context = self.run(context)
        except Exception as e:
            logger.error(f"Stage failed: {self.name}")
            logger.error(f"Error: {e}")
            raise

        # Validate outputs
        if not self.validate_outputs(result_context):
            raise ValueError(f"Output validation failed: {self.name}")

        # Record metrics
        elapsed_time = time.time() - start_time
        result_context.metrics[f'{self.name}_duration_seconds'] = elapsed_time

        logger.info(f"Completed stage: {self.name} ({elapsed_time:.2f}s)")

        # Log to MLflow if available
        if result_context.mlflow_tracker:
            result_context.mlflow_tracker.log_metric(
                f'{self.name}_duration_seconds',
                elapsed_time
            )

        return result_context

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Actual stage logic (implemented by subclass)."""
        pass

# Usage
class Pipeline:
    def run(self, config: Dict[str, Any]) -> PipelineContext:
        """Execute pipeline with automatic metrics."""
        context = PipelineContext(config=config, artifacts={}, metrics={})

        for stage in self.stages:
            context = stage.run_with_metrics(context)  # Automatic metrics

        return context
```

**Benefits:**
- Automatic performance tracking
- Easy to identify bottlenecks
- Logged to MLflow for analysis

---

## 3. Model Wrapper Patterns

### Pattern 3.1: Unified Interface

**Problem:** Different model types have incompatible APIs.

**Solution:** Abstract base class with consistent interface.

```python
# ❌ BEFORE: Inconsistent interfaces
xgb_model.fit(X, y, verbose=False)
two_stage.fit(X, y_pra, y_minutes)
ensemble.fit(X.values, y.values)  # Requires numpy

# ✅ AFTER: Consistent interface
class BasePredictor(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BasePredictor':
        """Train model. Always takes DataFrame/Series."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions. Always takes DataFrame."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BasePredictor':
        """Load model from disk."""
        pass

# All models follow same interface
xgb_model = XGBoostPredictor(hyperparams)
two_stage_model = TwoStagePredictor(hyperparams)
ensemble_model = EnsemblePredictor(hyperparams)

# Use polymorphically
for model in [xgb_model, two_stage_model, ensemble_model]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    model.save(f'models/{model.name}')
```

**Benefits:**
- Treat all models the same
- Easy to swap models
- Polymorphism enables powerful abstractions

---

### Pattern 3.2: Model Factory

**Problem:** Hard to create models from configuration.

**Solution:** Factory pattern with registry.

```python
# ❌ BEFORE: Manual model creation
if model_type == 'xgboost':
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, ...)
elif model_type == 'lightgbm':
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, ...)
elif model_type == 'two_stage':
    minutes_model = cat.CatBoostRegressor(...)
    pra_model = cat.CatBoostRegressor(...)
    model = TwoStagePredictor(minutes_model, pra_model)
# ... more elif ...

# ✅ AFTER: Factory pattern
class ModelFactory:
    _registry = {
        'xgboost': XGBoostPredictor,
        'lightgbm': LightGBMPredictor,
        'catboost': CatBoostPredictor,
        'two_stage': TwoStagePredictor,
        'ensemble': EnsemblePredictor
    }

    @classmethod
    def create(cls, model_type: str, hyperparams: Dict) -> BasePredictor:
        """Create model from type and config."""
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls._registry[model_type]
        return model_class(hyperparams=hyperparams)

# Usage
model = ModelFactory.create(
    model_type=config['model']['type'],
    hyperparams=config['model']['hyperparams']
)

# Easy to add custom models
class CustomPredictor(BasePredictor):
    ...

ModelFactory.register('custom', CustomPredictor)
```

**Benefits:**
- Config-driven model creation
- Easy to add new model types
- Centralized model instantiation logic

---

## 4. Configuration Patterns

### Pattern 4.1: Hierarchical Configs

**Problem:** Duplicating config values across experiments.

**Solution:** Base configs + experiment-specific overrides.

```yaml
# configs/base/model.yaml (shared defaults)
xgboost:
  type: xgboost
  hyperparams:
    n_estimators: 300
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8

# configs/experiments/baseline.yaml
experiment:
  name: baseline_xgboost

model: !include configs/base/model.yaml#xgboost  # Inherit defaults

# configs/experiments/deeper_trees.yaml
experiment:
  name: xgboost_depth_8

model: !include configs/base/model.yaml#xgboost
model:
  hyperparams:
    max_depth: 8  # Override one parameter
```

**Benefits:**
- DRY (Don't Repeat Yourself)
- Easy to see what changed
- Maintain consistency across experiments

---

### Pattern 4.2: Environment-Specific Configs

**Problem:** Production configs mixed with development configs.

**Solution:** Environment overrides.

```yaml
# configs/environments/development.yaml
mlflow:
  tracking_uri: file:///Users/me/mlruns
  experiment_name: Development

data:
  game_logs_path: data/game_logs/sample_1000.csv  # Small sample

validation:
  n_folds: 2  # Fast validation

# configs/environments/production.yaml
mlflow:
  tracking_uri: https://mlflow.company.com
  experiment_name: Production

data:
  game_logs_path: s3://company/nba_props/game_logs/all.csv  # Full dataset

validation:
  n_folds: 5  # Thorough validation

# Usage
loader = ConfigLoader()
config = loader.load(
    'experiments/baseline.yaml',
    environment='production'  # Merges production overrides
)
```

**Benefits:**
- Separate dev/staging/prod configs
- Easy to switch environments
- Prevent accidents (using prod data in dev)

---

## 5. Testing Patterns

### Pattern 5.1: Feature Testing

**Problem:** No tests for feature correctness.

**Solution:** Systematic feature testing.

```python
import pytest
import pandas as pd
import numpy as np

class TestLagFeature:
    @pytest.fixture
    def sample_data(self):
        """Create sample game logs."""
        return pd.DataFrame({
            'PLAYER_ID': [1, 1, 1, 2, 2],
            'GAME_DATE': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-03',
                '2024-01-01', '2024-01-02'
            ]),
            'PRA': [30, 35, 40, 25, 28]
        })

    def test_lag_feature_prevents_leakage(self, sample_data):
        """Test that lag feature doesn't leak future data."""
        feature = LagFeature('PRA', lag=1)
        result = feature.compute(sample_data)

        # First game should be NaN (no previous game)
        assert pd.isna(result.iloc[0])  # Player 1, game 1
        assert pd.isna(result.iloc[3])  # Player 2, game 1

        # Second game should have first game's value
        assert result.iloc[1] == 30  # Player 1, game 2 = game 1's PRA
        assert result.iloc[4] == 25  # Player 2, game 2 = game 1's PRA

        # Third game should have second game's value
        assert result.iloc[2] == 35  # Player 1, game 3 = game 2's PRA

    def test_lag_feature_respects_player_grouping(self, sample_data):
        """Test that lag doesn't cross player boundaries."""
        feature = LagFeature('PRA', lag=1)
        result = feature.compute(sample_data)

        # Player 2's first game should NOT get Player 1's last game
        assert pd.isna(result.iloc[3])  # Should be NaN, not 40

    def test_lag_feature_handles_missing_data(self):
        """Test lag feature with insufficient history."""
        df = pd.DataFrame({
            'PLAYER_ID': [1],
            'GAME_DATE': pd.to_datetime(['2024-01-01']),
            'PRA': [30]
        })

        feature = LagFeature('PRA', lag=3)
        result = feature.compute(df)

        # Should be NaN (not enough history)
        assert pd.isna(result.iloc[0])
```

**Benefits:**
- Catch data leakage bugs
- Document expected behavior
- Prevent regressions

---

### Pattern 5.2: Pipeline Integration Testing

**Problem:** Components work individually but fail together.

**Solution:** End-to-end pipeline tests.

```python
class TestTrainingPipeline:
    @pytest.fixture
    def sample_config(self):
        return {
            'data': {
                'game_logs_path': 'tests/fixtures/sample_game_logs.csv',
                'train_end_date': '2024-06-30'
            },
            'features': {
                'groups': ['temporal']
            },
            'model': {
                'type': 'xgboost',
                'hyperparams': {
                    'n_estimators': 10,  # Small for testing
                    'max_depth': 3
                }
            }
        }

    def test_full_pipeline_execution(self, sample_config):
        """Test complete pipeline runs without errors."""
        # Create pipeline
        pipeline = Pipeline([
            DataLoadingStage(sample_config['data']),
            FeatureEngineeringStage(feature_store, sample_config['features']),
            ModelTrainingStage(sample_config['model']),
            ValidationStage()
        ])

        # Run pipeline
        context = pipeline.run(sample_config)

        # Verify outputs
        assert 'model' in context.artifacts
        assert 'predictions' in context.artifacts
        assert 'val_mae' in context.metrics

    def test_pipeline_validation_catches_bad_data(self):
        """Test that validation catches corrupted data."""
        bad_config = {
            'data': {
                'game_logs_path': 'tests/fixtures/corrupted_logs.csv'
            }
        }

        pipeline = Pipeline([
            DataLoadingStage(bad_config['data'])
        ])

        # Should raise validation error
        with pytest.raises(ValueError, match="No game logs loaded"):
            pipeline.run(bad_config)

    def test_pipeline_produces_reproducible_results(self, sample_config):
        """Test that pipeline is deterministic."""
        # Run twice
        context1 = pipeline.run(sample_config)
        context2 = pipeline.run(sample_config)

        # Predictions should be identical
        pred1 = context1.artifacts['predictions']
        pred2 = context2.artifacts['predictions']

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)
```

---

## 6. MLflow Patterns

### Pattern 6.1: Experiment Comparison

**Problem:** Hard to compare experiments.

**Solution:** Structured experiment tagging and comparison.

```python
# When starting run, use structured tags
tracker.start_run(
    run_name="experiment_001",
    tags={
        'phase': 'phase2',
        'model_type': 'two_stage',
        'feature_set': 'temporal_efficiency_position',
        'dataset_version': 'v2.1',
        'author': 'data_scientist_1'
    }
)

# Search experiments by tags
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Find all two-stage experiments
two_stage_runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.model_type = 'two_stage'"
)

# Compare metrics
for run in two_stage_runs:
    print(f"Run: {run.info.run_name}")
    print(f"MAE: {run.data.metrics['val_mae']}")
    print(f"Features: {run.data.tags['feature_set']}")
```

---

### Pattern 6.2: Feature Lineage Tracking

**Problem:** Don't know which features were used in old experiments.

**Solution:** Log complete feature schema.

```python
# Log feature schema with every experiment
tracker.log_dict(feature_store.export_schema(), "feature_schema.json")

# Schema includes:
{
    'features': {
        'PRA_lag1': {
            'class': 'LagFeature',
            'version': '1.0',
            'dependencies': ['PRA'],
            'parameters': {'stat': 'PRA', 'lag': 1}
        },
        # ... all features ...
    },
    'groups': {
        'temporal': ['PRA_lag1', 'PRA_lag3', ...],
        'efficiency': ['TS_pct', 'PER', ...]
    }
}

# Later, reconstruct exact feature set
with open('feature_schema.json') as f:
    schema = json.load(f)

# Recreate features
for feature_name, feature_def in schema['features'].items():
    feature_class = globals()[feature_def['class']]
    feature = feature_class(**feature_def['parameters'])
    store.register(feature, group=...)
```

---

## 7. Migration Patterns

### Pattern 7.1: Gradual Migration

**Problem:** Big bang rewrite is risky.

**Solution:** Migrate one component at a time.

```python
# Step 1: Keep old code, add feature flag
USE_NEW_FEATURE_STORE = os.getenv('USE_NEW_FEATURE_STORE', 'false') == 'true'

if USE_NEW_FEATURE_STORE:
    # New implementation
    features = feature_store.compute_features(df, feature_names)
else:
    # Old implementation (keep working)
    features = calculate_lag_features(df)
    features.update(calculate_rolling_features(df))

# Step 2: Test new implementation
# pytest tests/features/test_feature_store.py

# Step 3: Run A/B comparison
old_features = calculate_lag_features(df)
new_features = feature_store.compute_features(df, feature_names)

# Verify identical
pd.testing.assert_frame_equal(old_features, new_features)

# Step 4: Flip flag to new implementation
USE_NEW_FEATURE_STORE = True

# Step 5: After 1 week, delete old code
```

---

### Pattern 7.2: Validation Scripts

**Problem:** Need to verify refactoring doesn't change results.

**Solution:** Comparison scripts.

```python
# scripts/validate_refactoring.py
"""
Validate that refactored code produces identical results.
"""

import pandas as pd
import numpy as np
from old_code import old_walk_forward_training
from new_code import Pipeline, build_pipeline_from_config

def validate_predictions_match():
    """Verify predictions are identical."""
    # Run old code
    old_results = old_walk_forward_training(
        train_season='2023-24',
        val_season='2024-25'
    )
    old_predictions = old_results['val_df']['predicted_PRA']

    # Run new code
    config = load_config('configs/validation/baseline.yaml')
    pipeline = build_pipeline_from_config(config)
    new_context = pipeline.run(config)
    new_predictions = new_context.artifacts['predictions']['predicted_PRA']

    # Compare (allow small floating-point differences)
    np.testing.assert_allclose(
        old_predictions.values,
        new_predictions.values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Predictions don't match! Refactoring changed results."
    )

    print("✅ Validation passed: Predictions match!")

def validate_metrics_match():
    """Verify metrics are identical."""
    old_mae = 8.83
    new_mae = 8.83

    assert abs(old_mae - new_mae) < 0.01, "MAE changed!"

    print("✅ Validation passed: Metrics match!")

if __name__ == "__main__":
    validate_predictions_match()
    validate_metrics_match()
```

---

## Summary

### Key Takeaways

1. **Feature Engineering:** Use classes, not functions. Organize into groups.
2. **Pipelines:** Compose from validated stages with clear inputs/outputs.
3. **Models:** Unified interface + factory pattern for easy swapping.
4. **Configuration:** YAML-based with hierarchical inheritance.
5. **Testing:** Test features, stages, and full pipeline.
6. **MLflow:** Log everything (features, config, metrics, artifacts).
7. **Migration:** Gradual, with validation scripts.

### Anti-Patterns to Avoid

❌ **Don't:**
- Duplicate feature logic
- Mix concerns in one function
- Hardcode configurations
- Skip validation
- Change behavior without tests

✅ **Do:**
- Centralize feature definitions
- Separate stages cleanly
- Config-driven experiments
- Validate at boundaries
- Test everything

---

## Next Steps

1. Review patterns with team
2. Start with feature store (Pattern 1.1-1.4)
3. Add configuration system (Pattern 4.1-4.2)
4. Refactor pipeline stages (Pattern 2.1-2.3)
5. Standardize model interfaces (Pattern 3.1-3.2)
6. Enhance MLflow tracking (Pattern 6.1-6.2)

**Questions?** Refer to `ML_PIPELINE_ANALYSIS.md` for detailed architecture.
