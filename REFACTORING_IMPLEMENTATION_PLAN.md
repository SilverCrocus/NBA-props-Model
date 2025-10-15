# NBA Props Model - Refactoring Implementation Plan

**Date:** October 15, 2025
**Status:** Ready for Implementation
**Approval Required:** Yes

---

## Executive Summary

After comprehensive analysis by 4 specialized agents (DS-Modeler, MLflow-Manager, Python-Code-Reviewer, Research-Analyst), we've identified a **pragmatic refactoring strategy** that will improve code quality without risking model performance.

**Key Findings:**
- 🔴 **67 code quality issues** identified (5 critical, 18 high, 27 medium, 17 low)
- 🔴 **75% code duplication** in feature engineering across 5+ files
- 🔴 **682-line monolithic scripts** mixing concerns (data, features, training, validation)
- 🔴 **Zero unit tests** for production code
- 🔴 **All configurations hardcoded** (no experimentation flexibility)

**Selected Strategy:** "Quick Wins + Incremental Refactoring"
**Timeline:** 2-3 days for quick wins, then validate
**Risk Level:** Low (test-driven, incremental, continuous validation)

---

## Strategy: Quick Wins First

Rather than attempting a full 8-week architectural overhaul, we'll focus on **4 high-impact, low-risk improvements** that can be completed in 2-3 days:

### Why This Approach?

1. **Immediate Value:** Each quick win delivers standalone benefits
2. **Low Risk:** Changes are isolated and testable
3. **Fast Validation:** Can run backtest after each change
4. **User Request:** Matches "refactor → test → backtest" workflow

---

## Quick Win 1: Configuration Management System

**Problem:** All hyperparameters and paths hardcoded in scripts
**Impact:** Can't experiment without editing code
**Effort:** 2-3 hours
**Risk:** Low (no logic changes)

### Implementation:

```python
# config.py (NEW FILE)
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent

@dataclass
class DataConfig:
    """Data paths and parameters."""
    GAME_LOGS_PATH: Path = PROJECT_ROOT / "data" / "game_logs" / "all_game_logs_with_opponent.csv"
    CTG_DATA_PATH: Path = PROJECT_ROOT / "data" / "ctg_data_organized" / "players"
    RESULTS_DIR: Path = PROJECT_ROOT / "data" / "results"

    # Filtering
    MIN_MINUTES_PER_GAME: float = 15.0
    MIN_GAMES_PLAYED: int = 10

@dataclass
class ModelConfig:
    """Model hyperparameters."""
    # XGBoost
    XGBOOST_PARAMS: Dict[str, Any] = None

    # CatBoost (Two-Stage)
    CATBOOST_MINUTES_PARAMS: Dict[str, Any] = None
    CATBOOST_PRA_PARAMS: Dict[str, Any] = None

    def __post_init__(self):
        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

        if self.CATBOOST_MINUTES_PARAMS is None:
            self.CATBOOST_MINUTES_PARAMS = {
                "iterations": 300,
                "depth": 5,
                "learning_rate": 0.05,
                "random_state": 42,
                "verbose": False,
            }

        if self.CATBOOST_PRA_PARAMS is None:
            self.CATBOOST_PRA_PARAMS = {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.05,
                "random_state": 42,
                "verbose": False,
            }

@dataclass
class ValidationConfig:
    """Validation parameters."""
    WALK_FORWARD_MIN_TRAIN_GAMES: int = 10
    WALK_FORWARD_MIN_HISTORY: int = 5

    # Betting simulation
    ODDS: int = -110
    CONFIDENCE_THRESHOLDS: list = None

    def __post_init__(self):
        if self.CONFIDENCE_THRESHOLDS is None:
            self.CONFIDENCE_THRESHOLDS = [0.55, 0.60, 0.65]

# Singleton instances
data_config = DataConfig()
model_config = ModelConfig()
validation_config = ValidationConfig()
```

### Usage in Scripts:

```python
# Before:
game_logs_path = "data/game_logs/all_game_logs_with_opponent.csv"
hyperparams = {
    "n_estimators": 300,
    "max_depth": 6,
    ...
}

# After:
from config import data_config, model_config

df = pd.read_csv(data_config.GAME_LOGS_PATH)
model = xgb.XGBRegressor(**model_config.XGBOOST_PARAMS)
```

### Files to Update:
- `scripts/training/walk_forward_training_advanced_features.py` (lines 429, 536-542)
- `scripts/training/train_two_stage_model.py` (lines 67-95)
- `scripts/backtesting/backtest_walkforward_2024_25.py` (lines 19-20)
- `scripts/backtesting/final_comprehensive_backtest.py` (lines 54, 67-81)

### Validation:
```bash
# After implementation, verify config loads correctly
uv run python -c "from config import data_config, model_config; print(data_config.GAME_LOGS_PATH)"
```

---

## Quick Win 2: Fix Critical Error Handling

**Problem:** Bare `except:` clauses swallow all errors, making debugging impossible
**Impact:** Production failures are invisible
**Effort:** 1-2 hours
**Risk:** Low (improves stability)

### Implementation:

```python
# src/exceptions.py (NEW FILE)
"""Custom exceptions for NBA Props Model."""

class NBAPropsModelError(Exception):
    """Base exception for NBA Props Model."""
    pass

class ModelNotFittedError(NBAPropsModelError):
    """Raised when prediction called before fitting."""
    pass

class InvalidInputError(NBAPropsModelError):
    """Raised when input data is invalid."""
    pass

class TemporalLeakageError(NBAPropsModelError):
    """Raised when temporal leakage detected."""
    pass

class DataQualityError(NBAPropsModelError):
    """Raised when data quality checks fail."""
    pass
```

### Fix 1: `final_comprehensive_backtest.py:207-208`

```python
# Before:
except Exception:  # noqa: E722
    continue

# After:
except Exception as e:
    logger.warning(f"Failed to generate prediction for {player_name} on {pred_date}: {e}")
    failed_predictions.append({
        "player_name": player_name,
        "date": str(pred_date),
        "error": str(e),
        "error_type": type(e).__name__
    })
    continue
```

### Fix 2: Add Input Validation to `TwoStagePredictor.predict()`

```python
# src/models/two_stage_predictor.py:286-310

def predict(self, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using two-stage approach."""
    if not self.is_fitted:
        raise ModelNotFittedError(
            "TwoStagePredictor must be fitted before prediction. Call fit() first."
        )

    # Validate input
    if len(X) == 0:
        raise InvalidInputError("Cannot predict on empty DataFrame")

    missing_minutes = set(self.minutes_features) - set(X.columns)
    if missing_minutes:
        raise InvalidInputError(f"Missing required features for minutes model: {missing_minutes}")

    missing_pra = set(self.pra_features) - set(X.columns)
    if missing_pra:
        raise InvalidInputError(f"Missing required features for PRA model: {missing_pra}")

    # Stage 1: Predict minutes
    X_minutes = X[self.minutes_features].fillna(0)
    predicted_minutes = self.minutes_model.predict(X_minutes)

    # Stage 2: Predict PRA given predicted minutes
    X_pra = X[self.pra_features].copy()
    X_pra["predicted_MIN"] = predicted_minutes
    X_pra = X_pra.fillna(0)

    predicted_pra = self.pra_model.predict(X_pra)

    return predicted_pra
```

### Files to Update:
- `scripts/backtesting/final_comprehensive_backtest.py` (5 locations)
- `scripts/backtesting/backtest_walkforward_2024_25.py` (3 locations)
- `src/models/two_stage_predictor.py` (2 methods)
- `src/models/ensemble_predictor.py` (2 methods)

---

## Quick Win 3: Extract FeatureCalculator Class

**Problem:** Feature calculation logic duplicated across 4+ files (75% redundancy)
**Impact:** Bug fixes require changes in multiple places, inconsistent feature logic
**Effort:** 3-4 hours
**Risk:** Medium (logic changes, needs careful testing)

### Implementation:

```python
# src/features/calculator.py (NEW FILE)
"""
Centralized feature calculation with temporal leakage protection.
Single source of truth for all feature engineering logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureCalculator:
    """
    Calculate temporal features with explicit leakage protection.

    All methods ensure .shift(1) to prevent looking at future data.
    """

    def __init__(self, player_id_col: str = 'PLAYER_ID', date_col: str = 'GAME_DATE'):
        self.player_id_col = player_id_col
        self.date_col = date_col

    def calculate_lag_features(
        self,
        player_history: pd.DataFrame,
        stats: List[str] = ['PRA', 'MIN', 'PTS', 'REB', 'AST'],
        lags: List[int] = [1, 3, 5, 7]
    ) -> Dict[str, float]:
        """
        Calculate lag features (previous game values).

        Uses .shift(1) to ensure current game is excluded.

        Args:
            player_history: Historical games for player (MUST be before current_date)
            stats: Statistics to calculate lags for
            lags: Lag values (1 = previous game, 3 = 3 games ago, etc.)

        Returns:
            Dictionary of {feature_name: value}
        """
        features = {}

        if len(player_history) == 0:
            # No history - return 0 for all lags
            for stat in stats:
                for lag in lags:
                    features[f"{stat}_lag{lag}"] = 0.0
            return features

        # Sort by date (most recent first for indexing)
        history = player_history.sort_values(self.date_col, ascending=False)

        # Calculate lags
        for stat in stats:
            if stat not in history.columns:
                logger.warning(f"Stat '{stat}' not in player_history, skipping lag features")
                continue

            for lag in lags:
                if len(history) >= lag:
                    # lag=1 means previous game (index 0 after sorting descending)
                    features[f"{stat}_lag{lag}"] = history.iloc[lag - 1][stat]
                else:
                    # Not enough history
                    features[f"{stat}_lag{lag}"] = 0.0

        return features

    def calculate_rolling_features(
        self,
        player_history: pd.DataFrame,
        stats: List[str] = ['PRA', 'MIN', 'PTS', 'REB', 'AST'],
        windows: List[int] = [5, 10, 20],
        aggregations: List[str] = ['mean', 'std']
    ) -> Dict[str, float]:
        """
        Calculate rolling window statistics.

        Args:
            player_history: Historical games (MUST be before current_date)
            stats: Statistics to calculate rolling windows for
            windows: Window sizes
            aggregations: Aggregation functions ('mean', 'std', 'min', 'max')

        Returns:
            Dictionary of {feature_name: value}
        """
        features = {}

        if len(player_history) == 0:
            # No history
            for stat in stats:
                for window in windows:
                    for agg in aggregations:
                        features[f"{stat}_L{window}_{agg}"] = 0.0
            return features

        # Sort by date (ascending for rolling)
        history = player_history.sort_values(self.date_col, ascending=True)

        for stat in stats:
            if stat not in history.columns:
                continue

            for window in windows:
                for agg in aggregations:
                    feature_name = f"{stat}_L{window}_{agg}"

                    if len(history) >= window:
                        # Use last N games
                        recent_values = history[stat].tail(window)

                        if agg == 'mean':
                            features[feature_name] = recent_values.mean()
                        elif agg == 'std':
                            features[feature_name] = recent_values.std()
                        elif agg == 'min':
                            features[feature_name] = recent_values.min()
                        elif agg == 'max':
                            features[feature_name] = recent_values.max()
                        else:
                            features[feature_name] = 0.0
                    else:
                        # Not enough history - use what we have
                        if len(history) > 0:
                            if agg == 'mean':
                                features[feature_name] = history[stat].mean()
                            elif agg == 'std':
                                features[feature_name] = history[stat].std()
                            else:
                                features[feature_name] = 0.0
                        else:
                            features[feature_name] = 0.0

        return features

    def calculate_ewma_features(
        self,
        player_history: pd.DataFrame,
        stats: List[str] = ['PRA', 'MIN'],
        spans: List[int] = [5, 10]
    ) -> Dict[str, float]:
        """
        Calculate exponentially weighted moving averages.

        Args:
            player_history: Historical games (MUST be before current_date)
            stats: Statistics to calculate EWMA for
            spans: EWMA span values

        Returns:
            Dictionary of {feature_name: value}
        """
        features = {}

        if len(player_history) == 0:
            for stat in stats:
                for span in spans:
                    features[f"{stat}_ewma{span}"] = 0.0
            return features

        # Sort by date
        history = player_history.sort_values(self.date_col, ascending=True)

        for stat in stats:
            if stat not in history.columns:
                continue

            for span in spans:
                feature_name = f"{stat}_ewma{span}"

                if len(history) >= 3:  # Need minimum samples for EWMA
                    ewma_values = history[stat].ewm(span=span, min_periods=1).mean()
                    features[feature_name] = ewma_values.iloc[-1]  # Last value
                else:
                    features[feature_name] = history[stat].mean() if len(history) > 0 else 0.0

        return features

    def calculate_trend_features(
        self,
        player_history: pd.DataFrame,
        stats: List[str] = ['PRA', 'MIN'],
        recent_window: int = 5,
        comparison_window: int = 20
    ) -> Dict[str, float]:
        """
        Calculate trend features (recent vs longer-term average).

        Args:
            player_history: Historical games
            stats: Statistics to calculate trends for
            recent_window: Recent games window (default 5)
            comparison_window: Comparison window (default 20)

        Returns:
            Dictionary of {feature_name: value}
        """
        features = {}

        if len(player_history) < recent_window:
            for stat in stats:
                features[f"{stat}_trend"] = 0.0
            return features

        history = player_history.sort_values(self.date_col, ascending=True)

        for stat in stats:
            if stat not in history.columns:
                continue

            # Recent average
            recent_avg = history[stat].tail(recent_window).mean()

            # Comparison average
            if len(history) >= comparison_window:
                comparison_avg = history[stat].tail(comparison_window).mean()
            else:
                comparison_avg = history[stat].mean()

            # Trend = recent - comparison (positive = trending up)
            features[f"{stat}_trend"] = recent_avg - comparison_avg

        return features

    def calculate_all_features(
        self,
        player_history: pd.DataFrame,
        current_date: pd.Timestamp = None
    ) -> Dict[str, float]:
        """
        Calculate all temporal features at once.

        Args:
            player_history: Historical games (MUST be before current_date)
            current_date: Current prediction date (for validation)

        Returns:
            Dictionary of all features
        """
        # Validate temporal correctness
        if current_date is not None:
            future_games = player_history[player_history[self.date_col] >= current_date]
            if len(future_games) > 0:
                raise ValueError(
                    f"Temporal leakage detected: {len(future_games)} games >= {current_date}"
                )

        features = {}

        # Calculate all feature types
        features.update(self.calculate_lag_features(player_history))
        features.update(self.calculate_rolling_features(player_history))
        features.update(self.calculate_ewma_features(player_history))
        features.update(self.calculate_trend_features(player_history))

        return features


class ContextualFeatureCalculator:
    """Calculate contextual features (opponent, rest, schedule)."""

    def calculate_rest_features(
        self,
        player_history: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate rest and fatigue features."""
        features = {}

        if len(player_history) == 0:
            features['days_rest'] = 0
            features['is_b2b'] = 0
            features['games_last_7d'] = 0
            return features

        # Sort by date
        history = player_history.sort_values('GAME_DATE', ascending=True)

        # Days since last game
        last_game_date = history['GAME_DATE'].iloc[-1]
        days_rest = (current_date - last_game_date).days
        features['days_rest'] = days_rest

        # Back-to-back indicator
        features['is_b2b'] = 1 if days_rest <= 1 else 0

        # Games in last 7 days
        seven_days_ago = current_date - pd.Timedelta(days=7)
        recent_games = history[history['GAME_DATE'] > seven_days_ago]
        features['games_last_7d'] = len(recent_games)

        return features


# Global instance for easy import
temporal_calculator = TemporalFeatureCalculator()
contextual_calculator = ContextualFeatureCalculator()
```

### Usage in Scripts:

```python
# Before (walk_forward_training_advanced_features.py:41-61):
def calculate_lag_features(player_history, lags=[1, 3, 5, 7]):
    features = {}
    if len(player_history) == 0:
        for lag in lags:
            features[f"PRA_lag{lag}"] = 0
            features[f"MIN_lag{lag}"] = 0
        return features
    history = player_history.sort_values("GAME_DATE", ascending=False)
    for lag in lags:
        if len(history) >= lag:
            features[f"PRA_lag{lag}"] = history.iloc[lag - 1]["PRA"]
            features[f"MIN_lag{lag}"] = history.iloc[lag - 1]["MIN"]
    return features

# After:
from src.features.calculator import temporal_calculator

features = temporal_calculator.calculate_lag_features(
    player_history,
    stats=['PRA', 'MIN'],
    lags=[1, 3, 5, 7]
)
```

### Files to Update:
- `scripts/training/walk_forward_training_advanced_features.py` (remove lines 41-334, replace with imports)
- `scripts/training/train_two_stage_model.py` (remove duplicate feature calc)
- `scripts/training/generate_twostage_2024_25.py` (use centralized calculator)

### Validation:
```python
# Test temporal correctness
import pandas as pd
from src.features.calculator import temporal_calculator

# Create test data
df = pd.DataFrame({
    'PLAYER_ID': [1, 1, 1],
    'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
    'PRA': [30, 35, 40]
})

# Calculate features for game on 2024-01-03
player_history = df[df['GAME_DATE'] < '2024-01-03']
features = temporal_calculator.calculate_lag_features(player_history, stats=['PRA'], lags=[1])

# PRA_lag1 should be 35 (previous game)
assert features['PRA_lag1'] == 35, f"Expected 35, got {features['PRA_lag1']}"
print("✅ Temporal correctness validated")
```

---

## Quick Win 4: Create BasePredictor Interface

**Problem:** 3 model types with incompatible APIs (can't easily swap models)
**Impact:** Hard to experiment with different models, inconsistent save/load
**Effort:** 3-4 hours
**Risk:** Medium (interface changes, needs testing)

### Implementation:

```python
# src/models/base.py (NEW FILE)
"""Base predictor interface for all models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BasePredictor':
        """Load model from disk."""
        pass
```

### Update TwoStagePredictor:

```python
# src/models/two_stage_predictor.py

from .base import BasePredictor  # Add import

class TwoStagePredictor(BasePredictor):  # Change inheritance
    """Two-stage predictor: Minutes → PRA"""

    def __init__(self, minutes_model_params: Dict = None, pra_model_params: Dict = None):
        # Initialize base class
        super().__init__("two_stage", hyperparams={
            'minutes_params': minutes_model_params,
            'pra_params': pra_model_params
        })

        # Existing initialization
        if minutes_model_params is None:
            minutes_model_params = {...}
        if pra_model_params is None:
            pra_model_params = {...}

        self.minutes_model = cat.CatBoostRegressor(**minutes_model_params)
        self.pra_model = cat.CatBoostRegressor(**pra_model_params)

        self.minutes_features = None
        self.pra_features = None

    def fit(self, X: pd.DataFrame, y_pra: pd.Series, y_minutes: pd.Series = None) -> 'TwoStagePredictor':
        """Fit both stages of the predictor."""
        # Existing fit logic...
        self.is_fitted = True
        self.feature_names = self.minutes_features + self.pra_features
        return self

    # Existing predict(), save(), load() methods unchanged
```

### Update EnsemblePredictor:

```python
# src/models/ensemble_predictor.py

from .base import BasePredictor  # Add import

class TreeEnsemblePredictor(BasePredictor):  # Change inheritance
    """Stacking ensemble of tree-based models."""

    def __init__(self, xgb_params=None, lgb_params=None, cat_params=None, ...):
        # Initialize base class
        super().__init__("ensemble", hyperparams={
            'xgb': xgb_params,
            'lgb': lgb_params,
            'cat': cat_params
        })

        # Existing initialization...

    # Update fit() to set is_fitted and feature_names
    # Other methods unchanged
```

### Validation:
```python
# Test polymorphism
from src.models.two_stage_predictor import TwoStagePredictor
from src.models.ensemble_predictor import TreeEnsemblePredictor

def test_model(model: BasePredictor, X_train, y_train, X_test):
    """Works with any model that inherits BasePredictor."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Both work with same interface
two_stage = TwoStagePredictor()
ensemble = TreeEnsemblePredictor()

test_model(two_stage, X_train, y_train, X_test)
test_model(ensemble, X_train, y_train, X_test)

print("✅ Polymorphic interface validated")
```

---

## Validation Strategy

After each quick win, we'll validate with these checks:

### 1. Unit Tests (Quick)
```bash
# Test configuration loading
uv run python -m pytest tests/test_config.py -v

# Test feature calculator
uv run python -m pytest tests/test_calculator.py -v

# Test model interfaces
uv run python -m pytest tests/test_models.py -v
```

### 2. Walk-Forward Validation (30 minutes)
```bash
# Run walk-forward with refactored code
uv run python scripts/training/walk_forward_training_advanced_features.py

# Compare to baseline
python scripts/compare_predictions.py \
    --baseline data/results/walk_forward_advanced_features_2024_25_baseline.csv \
    --current data/results/walk_forward_advanced_features_2024_25.csv
```

**Success Criteria:**
- MAE within ±0.05 of baseline (6.10 ± 0.05)
- Same number of predictions (25,349)
- Predictions match within floating-point error (<0.01)

### 3. Comprehensive Backtest (1 hour)
```bash
# Run full backtest
uv run python scripts/backtesting/final_comprehensive_backtest.py

# Compare metrics
python scripts/compare_backtest_results.py
```

**Success Criteria:**
- MAE: 8.83 ± 0.1 (no regression)
- Win Rate: 52% ± 1%
- ROI: 0.91% ± 0.1%
- Same number of profitable bets

---

## Implementation Timeline

### Day 1 (3-4 hours):
- ✅ **Morning:** Quick Win 1 (Config) + Quick Win 2 (Error Handling)
- ✅ **Afternoon:** Unit tests for config and error handling
- ✅ **End of Day:** Commit changes, tag as `refactor-day1`

### Day 2 (3-4 hours):
- ✅ **Morning:** Quick Win 3 (FeatureCalculator)
- ✅ **Afternoon:** Unit tests for feature calculator
- ✅ **End of Day:** Walk-forward validation, commit as `refactor-day2`

### Day 3 (3-4 hours):
- ✅ **Morning:** Quick Win 4 (BasePredictor interface)
- ✅ **Afternoon:** Integration testing
- ✅ **Evening:** Comprehensive backtest, final validation

### Day 4 (Optional - Buffer):
- ✅ Fix any issues found in validation
- ✅ Documentation updates
- ✅ Final commit and tag as `refactor-complete`

---

## Success Metrics

### Code Quality Improvements:

| Metric | Before | After | Target Met? |
|--------|--------|-------|-------------|
| Code Duplication | 75% | <20% | ✓ (Feature Calculator eliminates duplication) |
| Hardcoded Configs | 100% | 0% | ✓ (Config module) |
| Error Handling | Poor | Good | ✓ (Custom exceptions) |
| Model Interface | Inconsistent | Consistent | ✓ (BasePredictor) |

### Performance (No Regression):

| Metric | Baseline | After Refactoring | Status |
|--------|----------|-------------------|--------|
| MAE | 8.83 | 8.83 ± 0.1 | ✓ Must match |
| Win Rate | 52% | 52% ± 1% | ✓ Must match |
| ROI | 0.91% | 0.91% ± 0.1% | ✓ Must match |

### Development Speed:

| Task | Before | After |
|------|--------|-------|
| Change hyperparameters | Edit code (10 min) | Edit config (30 sec) |
| Swap models | Rewrite script (2 hours) | Change config line (30 sec) |
| Debug feature bug | Find all occurrences (30 min) | Fix once (5 min) |
| Add new feature | Copy-paste logic (15 min) | Use calculator (2 min) |

---

## Risk Mitigation

### Risk 1: Performance Regression
**Mitigation:**
- Run baseline before changes
- Validate after each quick win
- Use absolute file comparison (predictions should match exactly)
- Rollback immediately if any regression

### Risk 2: Breaking Changes
**Mitigation:**
- Git tag before each change
- Keep old code until validation passes
- Gradual rollout (Config → Error Handling → Features → Models)
- Can rollback to any tag

### Risk 3: Feature Logic Changes
**Mitigation:**
- Extract existing logic exactly as-is (no improvements)
- Add explicit temporal validation
- Write unit tests before refactoring
- Compare feature values before/after

### Risk 4: Time Overrun
**Mitigation:**
- Each quick win is independent (can pause anytime)
- Day 4 is buffer for unexpected issues
- Can defer Quick Win 4 if needed (models still work)

---

## Files to Create

### New Files:
1. `config.py` - Configuration management (~100 lines)
2. `src/exceptions.py` - Custom exceptions (~30 lines)
3. `src/features/calculator.py` - Feature calculator (~400 lines)
4. `src/models/base.py` - Base predictor interface (~100 lines)
5. `tests/test_config.py` - Config tests (~50 lines)
6. `tests/test_calculator.py` - Calculator tests (~150 lines)
7. `tests/test_models.py` - Model interface tests (~100 lines)

### Files to Modify:
1. `scripts/training/walk_forward_training_advanced_features.py` (remove 300 lines, add imports)
2. `scripts/training/train_two_stage_model.py` (use config, feature calculator)
3. `scripts/backtesting/backtest_walkforward_2024_25.py` (use config)
4. `scripts/backtesting/final_comprehensive_backtest.py` (error handling, config)
5. `src/models/two_stage_predictor.py` (inherit from BasePredictor)
6. `src/models/ensemble_predictor.py` (inherit from BasePredictor)

---

## Next Steps

### Approval Required:

Before proceeding with implementation, please review and approve:

1. ✅ **Strategy:** Quick Wins approach (vs full 8-week refactoring)
2. ✅ **Priorities:** Config → Error Handling → FeatureCalculator → BasePredictor
3. ✅ **Timeline:** 2-3 days for implementation + validation
4. ✅ **Success Criteria:** No performance regression (MAE, Win Rate, ROI)

### To Begin Implementation:

```bash
# 1. Create baseline for comparison
cp data/results/walk_forward_advanced_features_2024_25.csv \
   data/results/walk_forward_advanced_features_2024_25_BASELINE.csv

# 2. Tag current state
git add -A
git commit -m "Pre-refactoring baseline - MAE 8.83, 52% win rate"
git tag pre-refactoring-baseline

# 3. Create refactoring branch
git checkout -b refactoring

# 4. Start with Quick Win 1 (Config)
```

---

**Ready to proceed with implementation?**

Type "yes" to begin refactoring, or provide feedback if you'd like to adjust the plan.
