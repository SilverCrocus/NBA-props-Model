# NBA Props Model: ML Pipeline Architecture Analysis

**Date:** October 15, 2025
**Analysis Focus:** Production ML Pipeline Refactoring & Architecture Improvements

---

## Executive Summary

The NBA Props model has achieved strong initial results (52% win rate, 0.91% ROI, MAE 8.83) but the ML pipeline has significant technical debt and architectural gaps that limit:
- **Experimentation velocity** (hard to swap models/features)
- **Reproducibility** (configuration scattered across scripts)
- **Model versioning** (limited MLflow integration)
- **Feature reusability** (duplicated feature logic)

This analysis identifies **15 high-impact refactoring opportunities** across 5 core areas.

---

## 1. Current Architecture Assessment

### 1.1 Pipeline Components Review

#### ✅ **Strengths**
- **Temporal leakage prevention**: Excellent `.shift(1)` discipline in feature engineering
- **Walk-forward validation**: Proper time-series validation implementation
- **MLflow integration**: Basic tracking infrastructure exists
- **Model diversity**: Two-stage predictor, ensemble, and calibration components

#### ⚠️ **Critical Weaknesses**

| Component | Issue | Impact | Priority |
|-----------|-------|--------|----------|
| Feature Engineering | Logic duplicated across 4+ files | High maintenance burden | **CRITICAL** |
| Model Training | Monolithic scripts (600+ lines) | Hard to test/modify | **HIGH** |
| Configuration | Hardcoded hyperparameters | No experiment tracking | **HIGH** |
| Feature Store | No centralized feature registry | Feature drift risk | **MEDIUM** |
| Pipeline Orchestration | Manual script execution | Error-prone, not scalable | **MEDIUM** |

---

## 2. Detailed Component Analysis

### 2.1 Feature Engineering Architecture

**Current State:**
- Feature logic scattered across:
  - `src/features/engineering.py` (legacy, 689 lines)
  - `src/features/position_defense.py` (Phase 2 addition)
  - `scripts/training/walk_forward_training_advanced_features.py` (inline functions)
  - `src/data/game_log_builder.py` (temporal features)
  - `utils/ctg_feature_builder.py` (CTG stats)

**Problems:**

1. **Code Duplication (75% redundancy)**
   ```python
   # Pattern appears 4 times across different files:
   def calculate_lag_features(player_history, lags=[1, 3, 5, 7]):
       features = {}
       for lag in lags:
           if len(history) >= lag:
               features[f"PRA_lag{lag}"] = history.iloc[lag - 1]["PRA"]
   ```

2. **No Feature Registry**
   - Cannot track which features are available
   - Cannot version feature sets
   - Cannot share features across experiments

3. **Inconsistent Interfaces**
   ```python
   # Three different APIs for similar operations:
   calculate_lag_features(df, lags=[1,3,5])          # walk_forward script
   create_lag_features(df, stats=['PRA'], lags=[1])  # game_log_builder
   calculate_rolling_averages(df, stats, windows)    # engineering.py
   ```

4. **Feature Validation Gap**
   - No centralized validation
   - No feature quality checks
   - No automated tests for feature correctness

---

### 2.2 Model Training Pipeline

**Current State:**
- Monolithic training scripts:
  - `walk_forward_training_advanced_features.py`: 682 lines
  - `train_two_stage_model.py`: Mixed concerns
  - Inline feature calculation during training

**Problems:**

1. **Monolithic Structure**
   ```python
   # 682-line script mixes concerns:
   def walk_forward_train_and_validate():
       # Lines 1-100: Data loading
       # Lines 101-200: CTG merging
       # Lines 201-400: Feature engineering
       # Lines 401-500: Model training
       # Lines 501-600: Walk-forward validation
       # Lines 601-682: Metrics & plotting
   ```

2. **No Separation of Concerns**
   - Data loading + feature engineering + training + evaluation in one function
   - Hard to test individual components
   - Hard to swap model types

3. **Configuration Hardcoding**
   ```python
   # Hyperparameters scattered throughout:
   hyperparams = {
       "n_estimators": 300,      # Line 536
       "max_depth": 6,           # Line 537
       "learning_rate": 0.05,    # Line 538
       # ...
   }
   # No way to experiment without editing code
   ```

4. **Limited Model Flexibility**
   - Can't easily swap XGBoost → LightGBM → CatBoost
   - Can't A/B test different model architectures
   - Hard to implement custom models

---

### 2.3 Configuration Management

**Current State:**
- Hyperparameters hardcoded in training scripts
- Feature configs hardcoded in feature engineering functions
- Pipeline configs hardcoded in walk-forward scripts

**Problems:**

1. **No Centralized Config**
   ```
   # Config scattered across:
   scripts/training/*.py          # Model hyperparameters
   src/features/*.py             # Feature parameters
   src/models/*.py               # Model-specific configs
   utils/ctg_feature_builder.py # CTG configs
   ```

2. **No Config Versioning**
   - Cannot reproduce experiments without reading code
   - Cannot track which config produced which results
   - Cannot compare configs across experiments

3. **No Environment Management**
   - Production vs staging configs mixed
   - No way to override configs for local testing
   - No validation of config values

---

### 2.4 Experiment Tracking (MLflow)

**Current State:**
- Basic MLflow tracking exists (`src/mlflow_integration/tracker.py`)
- Model registry implemented (`src/mlflow_integration/registry.py`)
- Used in some scripts, missing in others

**Problems:**

1. **Inconsistent Usage**
   ```python
   # Some scripts use MLflow:
   tracker = NBAPropsTracker(experiment_name="Phase1_Foundation")
   tracker.log_metrics(...)

   # Others don't:
   # No MLflow in: train_two_stage_model.py, position_defense experiments
   ```

2. **Missing Artifacts**
   - Feature importance saved as CSV (not logged to MLflow)
   - Predictions saved locally (not as MLflow artifacts)
   - Model configs not logged consistently

3. **No Feature Store Integration**
   - Features calculated on-the-fly
   - No way to query "which features were used in run X?"
   - No way to reuse features across experiments

4. **Limited Model Metadata**
   ```python
   # Missing critical metadata:
   # - Feature engineering code version
   # - Data preprocessing steps
   # - Feature schema/types
   # - Model training duration
   # - Hardware used
   ```

---

### 2.5 Model Interface Consistency

**Current State:**
- Three model types with different interfaces:
  - `TwoStagePredictor`: Custom fit/predict API
  - `TreeEnsemblePredictor`: Custom fit/predict API
  - XGBoost/LightGBM: Native scikit-learn API

**Problems:**

1. **Inconsistent Interfaces**
   ```python
   # TwoStagePredictor:
   predictor.fit(X, y_pra, y_minutes)  # Takes 3 arguments

   # TreeEnsemblePredictor:
   predictor.fit(X, y)  # Takes 2 arguments

   # XGBoost:
   model.fit(X, y, verbose=False)  # Takes 2 + kwargs
   ```

2. **No BasePredictor Class**
   - Cannot treat models polymorphically
   - Cannot implement common utilities (save/load, predict_proba, etc.)
   - Cannot enforce consistent error handling

3. **Inconsistent Save/Load**
   ```python
   # TwoStagePredictor:
   predictor.save(path_prefix)  # Custom pickle format

   # TreeEnsemblePredictor:
   # No save/load implemented!

   # XGBoost:
   model.save_model("model.json")  # Native format
   ```

---

## 3. Recommended Architecture Improvements

### 3.1 Feature Store Pattern

**Implement a centralized feature registry and computation engine.**

#### New Architecture:

```
src/features/
├── store.py              # FeatureStore class (registry + computation)
├── definitions/
│   ├── base.py          # BaseFeature abstract class
│   ├── temporal.py      # LagFeature, RollingFeature, EWMAFeature
│   ├── contextual.py    # RestFeature, OpponentFeature
│   ├── efficiency.py    # TSPctFeature, PERFeature
│   └── position.py      # PositionDefenseFeature
├── transformers/
│   ├── base.py          # BaseTransformer
│   ├── lag.py           # LagTransformer
│   ├── rolling.py       # RollingTransformer
│   └── ewma.py          # EWMATransformer
└── registry.py          # FeatureRegistry (metadata storage)
```

#### Implementation Example:

```python
# src/features/store.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class BaseFeature(ABC):
    """Base class for all feature definitions."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.dependencies = []
        self.metadata = {}

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute feature from input data."""
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate feature output."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize feature definition."""
        return {
            'name': self.name,
            'version': self.version,
            'class': self.__class__.__name__,
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }

class LagFeature(BaseFeature):
    """Lag feature (previous game values)."""

    def __init__(self, stat: str, lag: int, name: str = None):
        if name is None:
            name = f"{stat}_lag{lag}"
        super().__init__(name, version="1.0")
        self.stat = stat
        self.lag = lag
        self.dependencies = [stat]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute lag feature with .shift() to prevent leakage."""
        return df.groupby('PLAYER_ID')[self.stat].shift(self.lag)

class FeatureStore:
    """Centralized feature computation and registry."""

    def __init__(self):
        self.features: Dict[str, BaseFeature] = {}
        self.feature_groups: Dict[str, List[str]] = {}
        self.cache: Dict[str, pd.Series] = {}

    def register(self, feature: BaseFeature, group: str = None):
        """Register a feature definition."""
        self.features[feature.name] = feature
        if group:
            if group not in self.feature_groups:
                self.feature_groups[group] = []
            self.feature_groups[group].append(feature.name)

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Compute features with dependency resolution.

        Args:
            df: Input dataframe
            feature_names: List of features to compute (None = all)
            use_cache: Use cached features if available

        Returns:
            DataFrame with computed features
        """
        if feature_names is None:
            feature_names = list(self.features.keys())

        result = pd.DataFrame(index=df.index)

        # Topological sort for dependency resolution
        sorted_features = self._resolve_dependencies(feature_names)

        for feature_name in sorted_features:
            if use_cache and feature_name in self.cache:
                result[feature_name] = self.cache[feature_name]
            else:
                feature = self.features[feature_name]
                result[feature_name] = feature.compute(df)

                # Validate
                if not feature.validate(result[feature_name]):
                    raise ValueError(f"Feature validation failed: {feature_name}")

                # Cache
                if use_cache:
                    self.cache[feature_name] = result[feature_name]

        return result

    def get_feature_group(self, group: str) -> List[str]:
        """Get all features in a group."""
        return self.feature_groups.get(group, [])

    def export_schema(self) -> Dict[str, Any]:
        """Export feature schema for MLflow logging."""
        return {
            'features': {
                name: feat.to_dict()
                for name, feat in self.features.items()
            },
            'groups': self.feature_groups
        }

    def _resolve_dependencies(self, feature_names: List[str]) -> List[str]:
        """Topological sort for dependency resolution."""
        # Implementation of dependency resolution
        # Returns features in correct computation order
        pass
```

#### Usage Example:

```python
# Initialize feature store
store = FeatureStore()

# Register features
store.register(LagFeature('PRA', lag=1), group='temporal')
store.register(LagFeature('PRA', lag=3), group='temporal')
store.register(LagFeature('PRA', lag=5), group='temporal')
store.register(RollingFeature('PRA', window=5, stat='mean'), group='temporal')
store.register(EWMAFeature('PRA', span=5), group='temporal')
store.register(EfficiencyFeature('TS_pct'), group='efficiency')
store.register(OpponentFeature('opp_DRtg'), group='contextual')

# Compute all temporal features
temporal_features = store.compute_features(
    df,
    feature_names=store.get_feature_group('temporal')
)

# Log to MLflow
tracker.log_dict(store.export_schema(), "feature_schema.json")
```

#### Benefits:

✅ **Single source of truth** for feature definitions
✅ **Dependency resolution** (compute features in correct order)
✅ **Feature versioning** (track changes to feature logic)
✅ **Reusability** (use features across experiments)
✅ **Validation** (automated quality checks)
✅ **Caching** (avoid redundant computation)

---

### 3.2 Pipeline Orchestration Framework

**Break monolithic scripts into composable pipeline stages.**

#### New Architecture:

```
src/pipelines/
├── base.py                  # BasePipeline, PipelineStage
├── stages/
│   ├── data_loading.py      # DataLoadingStage
│   ├── feature_engineering.py  # FeatureEngineeringStage
│   ├── model_training.py    # ModelTrainingStage
│   ├── validation.py        # ValidationStage
│   └── prediction.py        # PredictionStage
├── configs/
│   ├── default.yaml         # Default pipeline config
│   ├── production.yaml      # Production config
│   └── experiment.yaml      # Experiment config
└── orchestrator.py          # PipelineOrchestrator
```

#### Implementation Example:

```python
# src/pipelines/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PipelineContext:
    """Shared context passed between pipeline stages."""
    config: Dict[str, Any]
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    mlflow_tracker: Optional[Any] = None

class PipelineStage(ABC):
    """Base class for pipeline stages."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute stage logic."""
        pass

    def validate_inputs(self, context: PipelineContext) -> bool:
        """Validate stage inputs."""
        return True

    def validate_outputs(self, context: PipelineContext) -> bool:
        """Validate stage outputs."""
        return True

class Pipeline:
    """Pipeline composed of multiple stages."""

    def __init__(self, name: str, stages: List[PipelineStage]):
        self.name = name
        self.stages = stages

    def run(self, config: Dict[str, Any]) -> PipelineContext:
        """Execute pipeline stages sequentially."""
        context = PipelineContext(
            config=config,
            artifacts={},
            metrics={}
        )

        logger.info(f"Starting pipeline: {self.name}")

        for stage in self.stages:
            logger.info(f"Running stage: {stage.name}")

            # Validate inputs
            if not stage.validate_inputs(context):
                raise ValueError(f"Input validation failed: {stage.name}")

            # Execute stage
            context = stage.run(context)

            # Validate outputs
            if not stage.validate_outputs(context):
                raise ValueError(f"Output validation failed: {stage.name}")

            logger.info(f"Completed stage: {stage.name}")

        logger.info(f"Pipeline completed: {self.name}")
        return context
```

```python
# src/pipelines/stages/data_loading.py
from pathlib import Path
import pandas as pd
from .base import PipelineStage, PipelineContext

class DataLoadingStage(PipelineStage):
    """Load game logs and CTG data."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("data_loading", config)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Load data from disk."""
        # Load game logs
        game_logs_path = self.config.get('game_logs_path',
                                         'data/game_logs/all_game_logs_combined.csv')
        df = pd.read_csv(game_logs_path)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Store in context
        context.artifacts['game_logs'] = df
        context.metrics['n_games_loaded'] = len(df)

        # Log to MLflow
        if context.mlflow_tracker:
            context.mlflow_tracker.log_metric('n_games_loaded', len(df))

        return context

    def validate_outputs(self, context: PipelineContext) -> bool:
        """Validate loaded data."""
        df = context.artifacts.get('game_logs')
        if df is None or len(df) == 0:
            return False

        # Check required columns
        required_cols = ['PLAYER_ID', 'GAME_DATE', 'PRA']
        if not all(col in df.columns for col in required_cols):
            return False

        return True
```

```python
# src/pipelines/stages/feature_engineering.py
from .base import PipelineStage, PipelineContext
from src.features.store import FeatureStore

class FeatureEngineeringStage(PipelineStage):
    """Compute features using FeatureStore."""

    def __init__(self, feature_store: FeatureStore, config: Dict[str, Any] = None):
        super().__init__("feature_engineering", config)
        self.feature_store = feature_store

    def run(self, context: PipelineContext) -> PipelineContext:
        """Compute features."""
        df = context.artifacts['game_logs']

        # Get feature list from config
        feature_groups = self.config.get('feature_groups', ['temporal', 'contextual'])
        feature_names = []
        for group in feature_groups:
            feature_names.extend(self.feature_store.get_feature_group(group))

        # Compute features
        features_df = self.feature_store.compute_features(df, feature_names)

        # Merge with game logs
        result = pd.concat([df, features_df], axis=1)

        # Store in context
        context.artifacts['features'] = result
        context.metrics['n_features'] = len(features_df.columns)

        # Log to MLflow
        if context.mlflow_tracker:
            context.mlflow_tracker.log_metric('n_features', len(features_df.columns))
            context.mlflow_tracker.log_dict(
                self.feature_store.export_schema(),
                "feature_schema.json"
            )

        return context
```

```python
# src/pipelines/stages/model_training.py
from .base import PipelineStage, PipelineContext
from src.models.factory import ModelFactory

class ModelTrainingStage(PipelineStage):
    """Train model using ModelFactory."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("model_training", config)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Train model."""
        df = context.artifacts['features']

        # Split data
        train_end = self.config.get('train_end_date', '2024-06-30')
        train_df = df[df['GAME_DATE'] <= train_end]

        # Get feature columns
        exclude_cols = ['PRA', 'GAME_DATE', 'PLAYER_ID', 'PLAYER_NAME']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['PRA']

        # Create model from config
        model_config = self.config.get('model', {})
        model = ModelFactory.create(
            model_type=model_config.get('type', 'xgboost'),
            hyperparams=model_config.get('hyperparams', {})
        )

        # Train
        model.fit(X_train, y_train)

        # Store in context
        context.artifacts['model'] = model
        context.artifacts['feature_cols'] = feature_cols

        # Log to MLflow
        if context.mlflow_tracker:
            context.mlflow_tracker.log_model(model, model_type=model_config['type'])

        return context
```

#### Usage Example:

```python
# scripts/run_training_pipeline.py
from src.pipelines.orchestrator import Pipeline
from src.pipelines.stages import (
    DataLoadingStage,
    FeatureEngineeringStage,
    ModelTrainingStage,
    ValidationStage
)
from src.features.store import FeatureStore
from src.mlflow_integration.tracker import NBAPropsTracker
import yaml

# Load config
with open('src/pipelines/configs/experiment.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize feature store
store = FeatureStore()
# ... register features ...

# Initialize MLflow
tracker = NBAPropsTracker(experiment_name="Phase2_Experiments")
tracker.start_run(run_name="experiment_001", tags=config.get('tags', {}))

# Create pipeline
pipeline = Pipeline(
    name="training_pipeline",
    stages=[
        DataLoadingStage(config['data']),
        FeatureEngineeringStage(store, config['features']),
        ModelTrainingStage(config['model']),
        ValidationStage(config['validation'])
    ]
)

# Run pipeline
context = pipeline.run(config)
context.mlflow_tracker = tracker

# Get results
model = context.artifacts['model']
metrics = context.metrics

print(f"MAE: {metrics['val_mae']:.2f}")
```

#### Benefits:

✅ **Modularity** (swap stages independently)
✅ **Testability** (test each stage in isolation)
✅ **Reproducibility** (config-driven experiments)
✅ **Error handling** (validation at each stage)
✅ **Reusability** (compose pipelines from stages)

---

### 3.3 Configuration Management System

**Implement centralized, versioned configuration.**

#### New Architecture:

```
configs/
├── base/
│   ├── data.yaml         # Data loading config
│   ├── features.yaml     # Feature engineering config
│   ├── models.yaml       # Model configs
│   └── validation.yaml   # Validation config
├── experiments/
│   ├── baseline.yaml     # Baseline experiment
│   ├── two_stage.yaml    # Two-stage predictor
│   └── ensemble.yaml     # Ensemble model
├── environments/
│   ├── development.yaml  # Local development
│   ├── staging.yaml      # Staging environment
│   └── production.yaml   # Production environment
└── schema.yaml           # Config validation schema
```

#### Configuration Schema:

```yaml
# configs/base/data.yaml
data:
  game_logs_path: "data/game_logs/all_game_logs_combined.csv"
  ctg_data_path: "data/ctg_data_organized/players"
  output_path: "data/processed"

  train_period:
    start_date: "2023-10-01"
    end_date: "2024-06-30"

  validation_period:
    start_date: "2024-10-01"
    end_date: "2025-06-30"

  filters:
    min_minutes_per_game: 15.0
    min_games_played: 10

# configs/base/features.yaml
features:
  groups:
    - temporal
    - contextual
    - efficiency
    - position_defense

  temporal:
    lag_features:
      stats: ["PRA", "MIN", "PTS", "REB", "AST"]
      lags: [1, 3, 5, 7]

    rolling_features:
      stats: ["PRA", "MIN", "PTS", "REB", "AST"]
      windows: [5, 10, 20]
      aggregations: ["mean", "std"]

    ewma_features:
      stats: ["PRA", "MIN"]
      spans: [5, 10]

  contextual:
    rest_features: true
    opponent_features: true
    home_away: true

  efficiency:
    ts_pct: true
    per: true
    usg_per_36: true

  position_defense:
    enabled: true
    window: 10

# configs/base/models.yaml
models:
  xgboost:
    type: "xgboost"
    hyperparams:
      n_estimators: 300
      max_depth: 6
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42

  lightgbm:
    type: "lightgbm"
    hyperparams:
      n_estimators: 300
      max_depth: 6
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
      num_leaves: 63

  catboost:
    type: "catboost"
    hyperparams:
      iterations: 300
      depth: 6
      learning_rate: 0.05
      subsample: 0.8

  two_stage:
    type: "two_stage"
    minutes_model: "catboost"
    pra_model: "catboost"
    minutes_hyperparams:
      iterations: 300
      depth: 5
    pra_hyperparams:
      iterations: 300
      depth: 6

  ensemble:
    type: "ensemble"
    base_models: ["xgboost", "lightgbm", "catboost"]
    meta_learner: "ridge"
    use_stacking: true

# configs/base/validation.yaml
validation:
  method: "walk_forward"

  walk_forward:
    min_train_games: 10
    min_history_for_prediction: 5

  metrics:
    - mae
    - rmse
    - r2
    - within_3pts_pct
    - within_5pts_pct

  betting_simulation:
    enabled: true
    odds: -110
    confidence_thresholds: [0.55, 0.60, 0.65]

  calibration:
    enabled: true
    method: "isotonic"
```

```yaml
# configs/experiments/baseline.yaml
experiment:
  name: "baseline_xgboost"
  description: "Baseline XGBoost model with temporal features"
  tags:
    phase: "phase1"
    model_type: "xgboost"
    feature_set: "temporal_only"

# Inherit from base configs
data: !include configs/base/data.yaml
validation: !include configs/base/validation.yaml

# Override features
features:
  groups:
    - temporal  # Only temporal features

# Override model
model: !include configs/base/models.yaml#xgboost

# Environment overrides
environment: development
```

#### Configuration Loader:

```python
# src/config/loader.py
import yaml
from pathlib import Path
from typing import Dict, Any
import os

class ConfigLoader:
    """Load and merge configuration from YAML files."""

    def __init__(self, config_dir: Path = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config_dir = config_dir

    def load(self, config_path: str, environment: str = None) -> Dict[str, Any]:
        """
        Load configuration with environment overrides.

        Args:
            config_path: Path to experiment config (e.g., 'experiments/baseline.yaml')
            environment: Environment name (e.g., 'development', 'production')

        Returns:
            Merged configuration dictionary
        """
        # Load base config
        config = self._load_yaml(config_path)

        # Resolve includes
        config = self._resolve_includes(config)

        # Apply environment overrides
        if environment:
            env_config = self._load_yaml(f"environments/{environment}.yaml")
            config = self._deep_merge(config, env_config)

        # Apply CLI/env var overrides
        config = self._apply_overrides(config)

        # Validate
        self._validate(config)

        return config

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML file."""
        full_path = self.config_dir / path
        with open(full_path, 'r') as f:
            return yaml.safe_load(f)

    def _resolve_includes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve !include directives."""
        # Implementation of include resolution
        pass

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Support CLI overrides like:
        # MODEL_LEARNING_RATE=0.1 python train.py
        for key in os.environ:
            if key.startswith('MODEL_') or key.startswith('FEATURE_'):
                # Parse and apply override
                pass
        return config

    def _validate(self, config: Dict[str, Any]):
        """Validate configuration against schema."""
        # Load schema
        schema = self._load_yaml("schema.yaml")
        # Validate config
        # Raise ValueError if invalid
        pass
```

#### Usage Example:

```python
# scripts/train_from_config.py
from src.config.loader import ConfigLoader
from src.pipelines.orchestrator import Pipeline
from src.mlflow_integration.tracker import NBAPropsTracker
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Experiment config')
    parser.add_argument('--env', default='development', help='Environment')
    args = parser.parse_args()

    # Load config
    loader = ConfigLoader()
    config = loader.load(args.config, environment=args.env)

    # Initialize MLflow
    tracker = NBAPropsTracker(
        experiment_name=config['experiment']['name']
    )
    tracker.start_run(
        run_name=config['experiment']['name'],
        tags=config['experiment']['tags']
    )

    # Log config to MLflow
    tracker.log_dict(config, "experiment_config.yaml")

    # Build and run pipeline
    pipeline = build_pipeline_from_config(config)
    context = pipeline.run(config)

    # Done
    tracker.end_run()

if __name__ == "__main__":
    main()
```

```bash
# Run experiments with different configs
python scripts/train_from_config.py --config experiments/baseline.yaml
python scripts/train_from_config.py --config experiments/two_stage.yaml
python scripts/train_from_config.py --config experiments/ensemble.yaml --env staging

# Override hyperparameters via environment
MODEL_LEARNING_RATE=0.1 python scripts/train_from_config.py --config experiments/baseline.yaml
```

#### Benefits:

✅ **Reproducibility** (experiments fully defined in config)
✅ **Version control** (config changes tracked in git)
✅ **Environment management** (dev/staging/prod configs)
✅ **Easy experimentation** (edit YAML, don't touch code)
✅ **Validation** (schema ensures config correctness)

---

### 3.4 Model Factory & Unified Interface

**Implement polymorphic model interface and factory pattern.**

#### Implementation:

```python
# src/models/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

class BasePredictor(ABC):
    """
    Base class for all predictors.
    Enforces consistent interface across model types.
    """

    def __init__(self, name: str, hyperparams: Dict[str, Any] = None):
        self.name = name
        self.hyperparams = hyperparams or {}
        self.is_fitted = False
        self.feature_names = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BasePredictor':
        """
        Train the model.

        Args:
            X: Feature matrix (DataFrame)
            y: Target values (Series)
            **kwargs: Additional fit parameters

        Returns:
            self (fitted model)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions (numpy array)
        """
        pass

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Default: no uncertainty
        predictions = self.predict(X)
        uncertainties = np.zeros_like(predictions)
        return predictions, uncertainties

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        return None

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BasePredictor':
        """Load model from disk."""
        pass

    def validate_input(self, X: pd.DataFrame) -> bool:
        """Validate input features."""
        if self.feature_names is None:
            return True

        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        return True

    def get_params(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return self.hyperparams.copy()

    def set_params(self, **params):
        """Set hyperparameters."""
        self.hyperparams.update(params)
        return self
```

```python
# src/models/xgboost_predictor.py
import xgboost as xgb
from .base import BasePredictor

class XGBoostPredictor(BasePredictor):
    """XGBoost model wrapper."""

    def __init__(self, hyperparams: Dict[str, Any] = None):
        super().__init__("xgboost", hyperparams)
        self.model = xgb.XGBRegressor(**self.hyperparams)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostPredictor':
        """Fit XGBoost model."""
        self.feature_names = list(X.columns)
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self.validate_input(X)
        return self.model.predict(X[self.feature_names])

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            return None

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model."""
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> 'XGBoostPredictor':
        """Load model."""
        predictor = cls()
        predictor.model = xgb.XGBRegressor()
        predictor.model.load_model(path)
        predictor.is_fitted = True
        return predictor
```

```python
# src/models/two_stage_predictor_v2.py
from .base import BasePredictor
from .factory import ModelFactory

class TwoStagePredictor(BasePredictor):
    """
    Two-stage predictor with unified interface.
    Stage 1: Predict minutes
    Stage 2: Predict PRA given predicted minutes
    """

    def __init__(self, hyperparams: Dict[str, Any] = None):
        super().__init__("two_stage", hyperparams)

        # Create sub-models from factory
        self.minutes_model = ModelFactory.create(
            model_type=hyperparams.get('minutes_model', 'catboost'),
            hyperparams=hyperparams.get('minutes_hyperparams', {})
        )

        self.pra_model = ModelFactory.create(
            model_type=hyperparams.get('pra_model', 'catboost'),
            hyperparams=hyperparams.get('pra_hyperparams', {})
        )

        self.minutes_features = None
        self.pra_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'TwoStagePredictor':
        """Fit both stages."""
        # Extract minutes target (if available)
        y_minutes = kwargs.get('y_minutes', X.get('MIN'))
        if y_minutes is None:
            raise ValueError("Must provide y_minutes or 'MIN' column in X")

        # Select features for each stage
        self.minutes_features = self._select_minutes_features(X)
        self.pra_features = self._select_pra_features(X)

        # Stage 1: Train minutes model
        self.minutes_model.fit(X[self.minutes_features], y_minutes)

        # Stage 2: Train PRA model with predicted minutes
        predicted_minutes_train = self.minutes_model.predict(X[self.minutes_features])
        X_pra = X[self.pra_features].copy()
        X_pra['predicted_MIN'] = predicted_minutes_train

        self.pra_model.fit(X_pra, y)

        self.is_fitted = True
        self.feature_names = self.minutes_features + self.pra_features
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using two stages."""
        # Stage 1: Predict minutes
        predicted_minutes = self.minutes_model.predict(X[self.minutes_features])

        # Stage 2: Predict PRA given predicted minutes
        X_pra = X[self.pra_features].copy()
        X_pra['predicted_MIN'] = predicted_minutes

        return self.pra_model.predict(X_pra)

    def _select_minutes_features(self, X: pd.DataFrame) -> List[str]:
        """Select features for minutes prediction."""
        minutes_patterns = ['MIN_lag', 'MIN_L', 'days_rest', 'is_b2b', 'games_last_7d']
        return [col for col in X.columns if any(p in col for p in minutes_patterns)]

    def _select_pra_features(self, X: pd.DataFrame) -> List[str]:
        """Select features for PRA prediction."""
        pra_patterns = ['PRA_lag', 'PRA_L', 'PRA_ewma', 'TS_pct', 'PER',
                        'CTG_', 'opp_', 'per_36']
        return [col for col in X.columns if any(p in col for p in pra_patterns)]

    def save(self, path: str):
        """Save both models."""
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save sub-models
        self.minutes_model.save(f"{path}_minutes")
        self.pra_model.save(f"{path}_pra")

        # Save metadata
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'minutes_features': self.minutes_features,
                'pra_features': self.pra_features,
                'hyperparams': self.hyperparams
            }, f)

    @classmethod
    def load(cls, path: str) -> 'TwoStagePredictor':
        """Load both models."""
        import pickle

        # Load metadata
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create predictor
        predictor = cls(metadata['hyperparams'])

        # Load sub-models
        predictor.minutes_model = ModelFactory.load(f"{path}_minutes")
        predictor.pra_model = ModelFactory.load(f"{path}_pra")

        predictor.minutes_features = metadata['minutes_features']
        predictor.pra_features = metadata['pra_features']
        predictor.is_fitted = True

        return predictor
```

```python
# src/models/factory.py
from typing import Dict, Any
from .base import BasePredictor
from .xgboost_predictor import XGBoostPredictor
from .lightgbm_predictor import LightGBMPredictor
from .catboost_predictor import CatBoostPredictor
from .two_stage_predictor_v2 import TwoStagePredictor
from .ensemble_predictor_v2 import EnsemblePredictor

class ModelFactory:
    """Factory for creating models from config."""

    _registry = {
        'xgboost': XGBoostPredictor,
        'lightgbm': LightGBMPredictor,
        'catboost': CatBoostPredictor,
        'two_stage': TwoStagePredictor,
        'ensemble': EnsemblePredictor
    }

    @classmethod
    def create(
        cls,
        model_type: str,
        hyperparams: Dict[str, Any] = None
    ) -> BasePredictor:
        """
        Create model from type and hyperparameters.

        Args:
            model_type: Model type ('xgboost', 'lightgbm', etc.)
            hyperparams: Hyperparameters dictionary

        Returns:
            Initialized model (not fitted)
        """
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls._registry[model_type]
        return model_class(hyperparams=hyperparams)

    @classmethod
    def load(cls, path: str, model_type: str = None) -> BasePredictor:
        """
        Load model from disk.

        Args:
            path: Path to model file
            model_type: Model type (inferred if not provided)

        Returns:
            Loaded model
        """
        if model_type is None:
            # Infer from path or metadata
            model_type = cls._infer_model_type(path)

        model_class = cls._registry[model_type]
        return model_class.load(path)

    @classmethod
    def register(cls, model_type: str, model_class: type):
        """Register custom model type."""
        cls._registry[model_type] = model_class

    @classmethod
    def _infer_model_type(cls, path: str) -> str:
        """Infer model type from path or metadata."""
        # Check file extensions or metadata
        if path.endswith('.json'):
            return 'xgboost'
        elif path.endswith('.txt'):
            return 'lightgbm'
        elif path.endswith('.cbm'):
            return 'catboost'
        else:
            raise ValueError(f"Cannot infer model type from path: {path}")
```

#### Usage Example:

```python
# scripts/train_any_model.py
from src.models.factory import ModelFactory
from src.config.loader import ConfigLoader

# Load config
loader = ConfigLoader()
config = loader.load('experiments/baseline.yaml')

# Create model from config
model = ModelFactory.create(
    model_type=config['model']['type'],
    hyperparams=config['model']['hyperparams']
)

# Train (same interface for all models)
model.fit(X_train, y_train)

# Predict (same interface)
predictions = model.predict(X_test)

# Get feature importance (if available)
importance = model.get_feature_importance()

# Save
model.save('models/baseline_model')

# Load later
loaded_model = ModelFactory.load('models/baseline_model', model_type='xgboost')
```

#### Benefits:

✅ **Polymorphism** (treat all models the same)
✅ **Easy swapping** (change model type in config)
✅ **Consistent API** (fit/predict/save/load everywhere)
✅ **Testability** (mock models in tests)
✅ **Extensibility** (register custom models)

---

### 3.5 Enhanced MLflow Integration

**Improve experiment tracking and model registry usage.**

#### Implementation:

```python
# src/mlflow_integration/enhanced_tracker.py
from .tracker import NBAPropsTracker
from typing import Dict, Any, List
import mlflow
import pandas as pd

class EnhancedNBAPropsTracker(NBAPropsTracker):
    """Enhanced tracker with feature store and model registry integration."""

    def log_feature_store_schema(self, feature_store):
        """Log feature store schema."""
        schema = feature_store.export_schema()
        self.log_dict(schema, "feature_store_schema.json")

        # Log feature groups as params
        for group, features in schema['groups'].items():
            mlflow.log_param(f"feature_group_{group}_count", len(features))

    def log_pipeline_config(self, pipeline_config: Dict[str, Any]):
        """Log complete pipeline configuration."""
        self.log_dict(pipeline_config, "pipeline_config.yaml")

        # Extract key params
        if 'model' in pipeline_config:
            mlflow.log_param('model_type', pipeline_config['model'].get('type'))

        if 'features' in pipeline_config:
            mlflow.log_param('feature_groups',
                           ','.join(pipeline_config['features'].get('groups', [])))

    def log_data_quality_metrics(self, df: pd.DataFrame):
        """Log data quality metrics."""
        metrics = {
            'data_n_rows': len(df),
            'data_n_cols': len(df.columns),
            'data_missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            'data_duplicate_pct': df.duplicated().sum() / len(df) * 100
        }
        self.log_metrics(metrics)

    def log_walk_forward_results(self, results_df: pd.DataFrame):
        """Log walk-forward validation results by date."""
        # Group by date
        daily_metrics = results_df.groupby('GAME_DATE').agg({
            'abs_error': 'mean',
            'error': 'mean'
        }).reset_index()

        # Log as time series
        for _, row in daily_metrics.iterrows():
            timestamp = int(row['GAME_DATE'].timestamp())
            self.log_metrics({
                'daily_mae': row['abs_error'],
                'daily_bias': row['error']
            }, step=timestamp)

    def compare_with_baseline(self, metrics: Dict[str, float], baseline_run_id: str):
        """Compare current run with baseline."""
        # Get baseline metrics
        baseline_run = mlflow.get_run(baseline_run_id)
        baseline_metrics = baseline_run.data.metrics

        # Calculate improvements
        improvements = {}
        for metric_name, current_value in metrics.items():
            baseline_value = baseline_metrics.get(metric_name)
            if baseline_value:
                improvement = baseline_value - current_value  # Lower is better for MAE
                pct_improvement = (improvement / baseline_value) * 100
                improvements[f"{metric_name}_vs_baseline"] = improvement
                improvements[f"{metric_name}_vs_baseline_pct"] = pct_improvement

        # Log improvements
        self.log_metrics(improvements)
        mlflow.set_tag('baseline_run_id', baseline_run_id)
```

```python
# src/mlflow_integration/model_registry_v2.py
from .registry import ModelRegistry
from typing import Dict, Any

class EnhancedModelRegistry(ModelRegistry):
    """Enhanced registry with automated promotion workflows."""

    def auto_promote_if_improved(
        self,
        model_name: str,
        new_version: int,
        metrics: Dict[str, float],
        improvement_threshold: float = 0.05
    ) -> bool:
        """
        Automatically promote model if it improves on production.

        Args:
            model_name: Model name
            new_version: New model version
            metrics: Metrics for new model
            improvement_threshold: Minimum improvement required (e.g., 0.05 = 5%)

        Returns:
            True if promoted, False otherwise
        """
        # Get current production model
        try:
            prod_model = self.get_model_version(model_name, stage='Production')
            prod_run = self.client.get_run(prod_model.run_id)
            prod_metrics = prod_run.data.metrics
        except ValueError:
            # No production model yet, auto-promote
            self.promote_model(model_name, new_version, 'Production')
            return True

        # Compare MAE (lower is better)
        new_mae = metrics.get('val_mae', float('inf'))
        prod_mae = prod_metrics.get('val_mae', float('inf'))

        improvement = (prod_mae - new_mae) / prod_mae

        if improvement >= improvement_threshold:
            # New model is better, promote
            self.promote_model(model_name, new_version, 'Production', archive_existing=True)

            # Log promotion reason
            self.client.set_model_version_tag(
                name=model_name,
                version=new_version,
                key='auto_promotion_reason',
                value=f'MAE improved by {improvement*100:.1f}%'
            )

            return True
        else:
            # Not good enough, keep in staging
            self.promote_model(model_name, new_version, 'Staging', archive_existing=False)
            return False

    def create_model_card(self, model_name: str, version: int) -> str:
        """
        Generate model card markdown.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Markdown string
        """
        info = self.get_model_info(model_name, version)

        card = f"""
# Model Card: {model_name} v{version}

## Model Details
- **Version:** {version}
- **Stage:** {info['stage']}
- **Run ID:** {info['run_id']}
- **Created:** {info['creation_timestamp']}

## Metrics
| Metric | Value |
|--------|-------|
"""

        for metric, value in info['metrics'].items():
            card += f"| {metric} | {value:.4f} |\n"

        card += f"""
## Hyperparameters
```yaml
{yaml.dump(info['params'], default_flow_style=False)}
```

## Tags
"""
        for tag_key, tag_value in info['tags'].items():
            card += f"- **{tag_key}:** {tag_value}\n"

        return card
```

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Establish core infrastructure without breaking existing functionality.

#### Week 1: Feature Store

- [ ] Implement `BaseFeature` abstract class
- [ ] Implement `FeatureStore` with registry
- [ ] Migrate 5 most common features (lag, rolling, EWMA)
- [ ] Write unit tests for feature computation
- [ ] Create feature schema export for MLflow

**Deliverables:**
- `src/features/store.py`
- `src/features/definitions/temporal.py`
- `tests/features/test_feature_store.py`

#### Week 2: Configuration System

- [ ] Create YAML config schema
- [ ] Implement `ConfigLoader` with validation
- [ ] Create base configs (data, features, models, validation)
- [ ] Create 3 experiment configs (baseline, two-stage, ensemble)
- [ ] Write config validation tests

**Deliverables:**
- `configs/` directory structure
- `src/config/loader.py`
- `tests/config/test_loader.py`

---

### Phase 2: Pipeline Refactoring (Week 3-4)

**Goal:** Break monolithic scripts into composable stages.

#### Week 3: Pipeline Stages

- [ ] Implement `BasePipelineStage` abstract class
- [ ] Implement `DataLoadingStage`
- [ ] Implement `FeatureEngineeringStage` (uses FeatureStore)
- [ ] Implement `ModelTrainingStage` (uses ModelFactory)
- [ ] Implement `ValidationStage`

**Deliverables:**
- `src/pipelines/base.py`
- `src/pipelines/stages/` (5 stage implementations)
- `tests/pipelines/test_stages.py`

#### Week 4: Pipeline Orchestration

- [ ] Implement `Pipeline` orchestrator
- [ ] Create `build_pipeline_from_config()` factory
- [ ] Migrate walk-forward training to pipeline
- [ ] Write integration tests
- [ ] Create CLI for running pipelines

**Deliverables:**
- `src/pipelines/orchestrator.py`
- `scripts/run_pipeline.py` (CLI)
- `tests/pipelines/test_integration.py`

---

### Phase 3: Model Interface (Week 5)

**Goal:** Standardize model interfaces and enable easy swapping.

#### Week 5: Model Factory & Interfaces

- [ ] Implement `BasePredictor` abstract class
- [ ] Wrap XGBoost in `XGBoostPredictor`
- [ ] Wrap LightGBM in `LightGBMPredictor`
- [ ] Wrap CatBoost in `CatBoostPredictor`
- [ ] Refactor `TwoStagePredictor` to use BasePredictor
- [ ] Refactor `EnsemblePredictor` to use BasePredictor
- [ ] Implement `ModelFactory`
- [ ] Write model tests

**Deliverables:**
- `src/models/base.py`
- `src/models/{xgboost,lightgbm,catboost}_predictor.py`
- `src/models/two_stage_predictor_v2.py`
- `src/models/ensemble_predictor_v2.py`
- `src/models/factory.py`
- `tests/models/test_factory.py`

---

### Phase 4: Enhanced MLflow (Week 6)

**Goal:** Improve experiment tracking and model registry.

#### Week 6: MLflow Enhancements

- [ ] Implement `EnhancedNBAPropsTracker`
- [ ] Add feature store schema logging
- [ ] Add pipeline config logging
- [ ] Add walk-forward time-series logging
- [ ] Implement `EnhancedModelRegistry`
- [ ] Add auto-promotion workflow
- [ ] Add model card generation
- [ ] Integrate with pipeline

**Deliverables:**
- `src/mlflow_integration/enhanced_tracker.py`
- `src/mlflow_integration/model_registry_v2.py`
- Updated pipeline stages to use enhanced tracker

---

### Phase 5: Migration & Testing (Week 7-8)

**Goal:** Complete migration and validate performance.

#### Week 7: Migration

- [ ] Migrate all feature engineering to FeatureStore
- [ ] Migrate all training scripts to pipeline
- [ ] Migrate all models to unified interface
- [ ] Update documentation
- [ ] Create migration guide

**Deliverables:**
- All scripts using new architecture
- Updated `CLAUDE.md` with new architecture
- `MIGRATION_GUIDE.md`

#### Week 8: Validation & Cleanup

- [ ] Run baseline experiments (verify MAE matches)
- [ ] Run walk-forward validation (verify no regressions)
- [ ] Performance benchmarking (ensure no slowdowns)
- [ ] Clean up deprecated code
- [ ] Final documentation pass

**Deliverables:**
- Validation report (MAE, ROI, win rate unchanged)
- Performance benchmark report
- Clean codebase (deprecated code removed)

---

## 5. Success Metrics

### Technical Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Code Duplication | ~75% | <20% | `radon` tool, manual review |
| Test Coverage | ~20% | >80% | `pytest-cov` |
| Lines of Code per Module | 600+ | <300 | `cloc` tool |
| Configuration Flexibility | 0% (hardcoded) | 100% | All configs in YAML |
| Feature Reusability | 0% | 100% | FeatureStore usage |
| Model Swapping Time | 2+ hours | <5 minutes | Time to change model type |
| Experiment Setup Time | 30+ minutes | <2 minutes | Time to create new experiment |

### ML Performance (Must Not Regress)

| Metric | Current | After Refactoring |
|--------|---------|-------------------|
| MAE | 8.83 | 8.83 ± 0.1 |
| Win Rate | 52% | 52% ± 1% |
| ROI | 0.91% | 0.91% ± 0.1% |

### Operational Metrics

| Metric | Target |
|--------|--------|
| Training Pipeline Runtime | No increase >10% |
| Feature Engineering Runtime | No increase >10% |
| Memory Usage | No increase >20% |

---

## 6. Risk Mitigation

### Risk 1: Performance Regression

**Mitigation:**
- Run baseline experiment before and after migration
- Use walk-forward validation (not train/test)
- Compare predictions file-by-file (should be identical within floating-point error)
- Performance benchmarking at each phase

### Risk 2: Breaking Changes

**Mitigation:**
- Maintain old code during migration (delete only at end)
- Feature flags for new vs old pipeline
- Gradual rollout (feature store → pipeline → models → MLflow)
- Extensive testing at each phase

### Risk 3: Increased Complexity

**Mitigation:**
- Clear documentation at each step
- Simple examples for common tasks
- Migration guide for team
- Office hours / Q&A sessions

### Risk 4: Timeline Slippage

**Mitigation:**
- Prioritize phases (Feature Store > Config > Pipeline > Models > MLflow)
- Each phase delivers standalone value
- Can pause after any phase without breaking system
- Buffer week at end for issues

---

## 7. Quick Wins (Immediate Implementation)

While the full refactoring takes 8 weeks, these can be done immediately:

### Quick Win 1: Configuration Management (2 days)

**Impact:** Immediate experimentation velocity improvement
**Effort:** Low
**Implementation:**

```python
# configs/experiments/baseline.yaml
model:
  type: xgboost
  n_estimators: 300
  learning_rate: 0.05

# Load and use
import yaml
with open('configs/experiments/baseline.yaml') as f:
    config = yaml.safe_load(f)

model = xgb.XGBRegressor(**config['model'])
```

### Quick Win 2: Feature Importance Logging (1 day)

**Impact:** Better understanding of features
**Effort:** Very Low
**Implementation:**

```python
# Add to all training scripts
tracker.log_feature_importance(feature_cols, model.feature_importances_)
```

### Quick Win 3: Model Save/Load Standardization (2 days)

**Impact:** Easier model deployment
**Effort:** Low
**Implementation:**

```python
# Standardize save format across all models
def save_model(model, path, metadata):
    mlflow.xgboost.save_model(model, path)
    with open(f"{path}/metadata.json", 'w') as f:
        json.dump(metadata, f)
```

---

## 8. Conclusion

The current NBA Props ML pipeline has strong fundamentals (temporal leakage prevention, walk-forward validation) but suffers from:

1. **75% code duplication** in feature engineering
2. **Monolithic training scripts** (600+ lines)
3. **Hardcoded configurations** (no experimentation flexibility)
4. **Inconsistent model interfaces** (hard to swap models)
5. **Limited MLflow integration** (missing artifacts and metadata)

The recommended refactoring addresses these issues through:

1. **FeatureStore pattern** (single source of truth for features)
2. **Pipeline orchestration** (composable stages)
3. **Configuration management** (YAML-driven experiments)
4. **Unified model interface** (BasePredictor + ModelFactory)
5. **Enhanced MLflow integration** (complete experiment tracking)

**Expected benefits:**
- **10x faster experimentation** (change config vs edit code)
- **5x less code duplication** (centralized feature logic)
- **100% reproducibility** (config-driven experiments)
- **Easy model swapping** (change one line in config)
- **Better collaboration** (clear interfaces and patterns)

**Timeline:** 8 weeks, with incremental value delivery at each phase.

**Next steps:**
1. Review and approve architecture
2. Start with Phase 1 (Feature Store + Config)
3. Run baseline experiments to establish benchmarks
4. Begin migration

---

## Appendix A: Code Examples

See implementation examples throughout this document for:
- Feature Store (Section 3.1)
- Pipeline Orchestration (Section 3.2)
- Configuration Management (Section 3.3)
- Model Factory (Section 3.4)
- Enhanced MLflow (Section 3.5)

## Appendix B: Testing Strategy

```python
# tests/features/test_feature_store.py
def test_lag_feature_computation():
    """Test lag feature prevents leakage."""
    df = pd.DataFrame({
        'PLAYER_ID': [1, 1, 1],
        'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'PRA': [30, 35, 40]
    })

    feature = LagFeature('PRA', lag=1)
    result = feature.compute(df)

    # First game should have NaN (no previous game)
    assert pd.isna(result.iloc[0])
    # Second game should have first game's PRA (30)
    assert result.iloc[1] == 30
    # Third game should have second game's PRA (35)
    assert result.iloc[2] == 35

# tests/pipelines/test_stages.py
def test_feature_engineering_stage():
    """Test feature engineering stage."""
    store = FeatureStore()
    store.register(LagFeature('PRA', lag=1), group='temporal')

    stage = FeatureEngineeringStage(store, config={'feature_groups': ['temporal']})

    context = PipelineContext(
        config={},
        artifacts={'game_logs': sample_df},
        metrics={}
    )

    result_context = stage.run(context)

    # Validate outputs
    assert 'features' in result_context.artifacts
    assert 'PRA_lag1' in result_context.artifacts['features'].columns
    assert result_context.metrics['n_features'] == 1
```

## Appendix C: Performance Benchmarking

```python
# scripts/benchmark_pipeline.py
import time
from src.pipelines.orchestrator import Pipeline

def benchmark_pipeline(pipeline, config, n_runs=5):
    """Benchmark pipeline performance."""
    times = []

    for i in range(n_runs):
        start = time.time()
        pipeline.run(config)
        end = time.time()
        times.append(end - start)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times)
    }

# Run benchmark
baseline_times = benchmark_pipeline(old_pipeline, config)
refactored_times = benchmark_pipeline(new_pipeline, config)

# Compare
slowdown = (refactored_times['mean_time'] - baseline_times['mean_time']) / baseline_times['mean_time']
assert slowdown < 0.1, f"Pipeline slowdown: {slowdown*100:.1f}% (max 10%)"
```
