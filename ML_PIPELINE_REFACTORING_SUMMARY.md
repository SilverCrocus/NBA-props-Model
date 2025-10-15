# ML Pipeline Refactoring: Executive Summary

**Date:** October 15, 2025
**Author:** NBA Props Model Team
**Status:** Recommendation for Implementation

---

## Problem Statement

The NBA Props model has achieved strong initial results (52% win rate, MAE 8.83) but the codebase has significant architectural issues:

- **75% code duplication** in feature engineering across 5+ files
- **Monolithic training scripts** (600+ lines, mixed concerns)
- **Zero configuration management** (all hyperparameters hardcoded)
- **Inconsistent model interfaces** (hard to swap XGBoost ↔ LightGBM ↔ CatBoost)
- **Limited experiment tracking** (MLflow underutilized)

**Impact:** Slow experimentation, poor reproducibility, high maintenance burden.

---

## Recommended Solution: 5-Part Architecture Overhaul

### 1. Feature Store Pattern
**Problem:** Feature logic duplicated across 5 files
**Solution:** Centralized feature registry with composable feature definitions

```python
# Before: Feature logic in 4+ places
def calculate_lag_features(...):  # In walk_forward script
def create_lag_features(...):    # In game_log_builder
def calculate_rolling_averages(...):  # In engineering.py

# After: Single source of truth
store = FeatureStore()
store.register(LagFeature('PRA', lag=1), group='temporal')
features = store.compute_features(df, feature_names=['PRA_lag1'])
```

**Benefits:**
- 75% → 20% code duplication
- Feature versioning
- Reusability across experiments
- Automated validation

---

### 2. Pipeline Orchestration
**Problem:** 682-line monolithic training scripts
**Solution:** Composable pipeline stages with validation

```python
# Before: Everything in one script
def walk_forward_train_and_validate():
    # 682 lines of mixed concerns

# After: Clean separation
pipeline = Pipeline([
    DataLoadingStage(config['data']),
    FeatureEngineeringStage(store, config['features']),
    ModelTrainingStage(config['model']),
    ValidationStage(config['validation'])
])
context = pipeline.run(config)
```

**Benefits:**
- Testable components
- Reusable stages
- Clear error handling
- Easy to modify/extend

---

### 3. Configuration Management
**Problem:** All configs hardcoded in Python
**Solution:** YAML-based configuration with validation

```yaml
# configs/experiments/baseline.yaml
model:
  type: xgboost
  hyperparams:
    n_estimators: 300
    learning_rate: 0.05

features:
  groups: [temporal, contextual, efficiency]
```

```bash
# Run different experiments without touching code
python train.py --config experiments/baseline.yaml
python train.py --config experiments/two_stage.yaml
python train.py --config experiments/ensemble.yaml --env production
```

**Benefits:**
- 100% reproducibility
- Version-controlled experiments
- Environment management (dev/staging/prod)
- Easy hyperparameter tuning

---

### 4. Unified Model Interface
**Problem:** 3 model types with incompatible APIs
**Solution:** BasePredictor pattern + ModelFactory

```python
# Before: Different interfaces
two_stage.fit(X, y_pra, y_minutes)  # 3 args
ensemble.fit(X, y)                   # 2 args
xgb_model.fit(X, y, verbose=False)   # kwargs

# After: Consistent interface
class BasePredictor:
    def fit(self, X, y, **kwargs): ...
    def predict(self, X): ...
    def save(self, path): ...
    def load(cls, path): ...

# Easy swapping via config
model = ModelFactory.create(
    model_type=config['model']['type'],  # 'xgboost', 'two_stage', 'ensemble'
    hyperparams=config['model']['hyperparams']
)
```

**Benefits:**
- Polymorphic model handling
- 5 minutes (vs 2 hours) to swap models
- Consistent save/load
- Extensible (register custom models)

---

### 5. Enhanced MLflow Integration
**Problem:** Incomplete experiment tracking
**Solution:** Full pipeline + feature + config logging

```python
# Log everything for reproducibility
tracker.log_feature_store_schema(store)  # Which features used
tracker.log_pipeline_config(config)       # Full config
tracker.log_walk_forward_results(results) # Time-series metrics
tracker.log_model(model)                  # Model artifact

# Auto-promote if improved
registry.auto_promote_if_improved(
    model_name="NBAPropsModel",
    new_version=5,
    metrics={'val_mae': 5.2},
    improvement_threshold=0.05  # Must improve 5%
)
```

**Benefits:**
- Complete experiment tracking
- Feature lineage
- Automated model promotion
- Easy comparison across runs

---

## Implementation Plan: 8-Week Roadmap

| Phase | Timeline | Deliverable | Value |
|-------|----------|-------------|-------|
| **Phase 1: Foundation** | Week 1-2 | Feature Store + Config System | Immediate experimentation velocity |
| **Phase 2: Pipeline** | Week 3-4 | Composable pipeline stages | Clean architecture, testability |
| **Phase 3: Models** | Week 5 | Unified model interface | Easy model swapping |
| **Phase 4: MLflow** | Week 6 | Enhanced tracking + registry | Complete reproducibility |
| **Phase 5: Migration** | Week 7-8 | Full migration + validation | Production-ready system |

**Incremental Delivery:** Each phase delivers standalone value. Can pause after any phase.

---

## Expected Impact

### Developer Experience

| Metric | Current | After Refactoring | Improvement |
|--------|---------|-------------------|-------------|
| **Time to run new experiment** | 2+ hours | 2 minutes | **60x faster** |
| **Time to swap models** | 2 hours | 5 minutes | **24x faster** |
| **Code duplication** | 75% | <20% | **3.75x reduction** |
| **Test coverage** | 20% | 80% | **4x improvement** |
| **Lines per module** | 600+ | <300 | **2x cleaner** |

### ML Performance (Must Not Regress)

| Metric | Current | Target |
|--------|---------|--------|
| MAE | 8.83 | 8.83 ± 0.1 |
| Win Rate | 52% | 52% ± 1% |
| ROI | 0.91% | 0.91% ± 0.1% |

**Risk Mitigation:** Extensive validation at each phase ensures no performance regression.

---

## Quick Wins (Can Start Today)

While full refactoring takes 8 weeks, these deliver immediate value:

### Quick Win 1: Configuration Files (2 days)
Move hyperparameters to YAML configs. Run experiments by changing config, not code.

### Quick Win 2: Feature Importance Logging (1 day)
Add `tracker.log_feature_importance()` to all training scripts.

### Quick Win 3: Model Save Standardization (2 days)
Standardize save/load across all models for easier deployment.

---

## Resource Requirements

- **Development Time:** 8 weeks (1 engineer full-time)
- **Risk:** Low (incremental rollout, extensive testing)
- **Dependencies:** None (all tools already in place)
- **Training Required:** Minimal (clear documentation + examples)

---

## Success Criteria

### Technical
✅ Code duplication <20%
✅ Test coverage >80%
✅ All configs in YAML
✅ 100% experiment reproducibility

### Performance
✅ MAE unchanged (8.83 ± 0.1)
✅ Win rate unchanged (52% ± 1%)
✅ No runtime slowdown >10%

### Operational
✅ <2 minutes to setup new experiment
✅ <5 minutes to swap model types
✅ Complete feature lineage tracking

---

## Recommendation

**Approve and prioritize for Q1 2026.**

This refactoring addresses fundamental architectural issues that currently limit:
- Experimentation velocity (new experiments take hours)
- Code maintainability (75% duplication)
- Reproducibility (hardcoded configs)
- Model flexibility (hard to swap models)

The 8-week investment will pay dividends through:
- **10x faster iteration** (critical for improving from 52% → 55%+ win rate)
- **Better reproducibility** (essential for production deployment)
- **Easier collaboration** (clear patterns and interfaces)
- **Reduced maintenance** (cleaner, more testable code)

**Next Steps:**
1. Approve architecture (this document)
2. Allocate 1 engineer for 8 weeks
3. Start Phase 1 (Feature Store + Config)
4. Weekly progress reviews

---

## Appendix: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         ML PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Data Loading │ ───> │  Feature     │ ───> │   Model      │ │
│  │    Stage     │      │ Engineering  │      │  Training    │ │
│  │              │      │    Stage     │      │    Stage     │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│         │                      │                      │         │
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │     Game     │      │   Feature    │      │    Model     │ │
│  │     Logs     │      │    Store     │      │   Factory    │ │
│  │              │      │  Registry    │      │              │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │ Validation   │ ───> │  Prediction  │ ───> │   MLflow     │ │
│  │    Stage     │      │    Stage     │      │   Tracking   │ │
│  │              │      │              │      │              │ │
│  └──────────────┘      └──────────────┘      └──────────────┘ │
│                                                       │         │
│                                                       ▼         │
│                                              ┌──────────────┐  │
│                                              │    Model     │  │
│                                              │   Registry   │  │
│                                              │              │  │
│                                              └──────────────┘  │
│                                                                  │
│                     Driven by YAML Config                       │
└─────────────────────────────────────────────────────────────────┘
```

---

**For detailed technical specifications, see:** `ML_PIPELINE_ANALYSIS.md`

**Questions?** Contact: NBA Props Model Team
