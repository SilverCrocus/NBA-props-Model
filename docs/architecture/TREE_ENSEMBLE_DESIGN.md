# Tree Ensemble Architecture Design

**Phase:** Phase 2 Week 2
**Goal:** Reduce MAE from 6.11 → 5.30-5.60 (8-13% improvement)
**Approach:** Stacking ensemble of XGBoost + LightGBM + CatBoost

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING DATA                            │
│              (Walk-Forward: 2023-24 Season)                  │
│                  80 features, 22,717 samples                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         LEVEL 0: BASE MODELS            │
        ├─────────────────────────────────────────┤
        │                                         │
        │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
        │  │ XGBoost  │  │LightGBM  │  │CatBoost│ │
        │  │ 300 trees│  │ 300 trees│  │300 iter│ │
        │  │ depth=6  │  │ depth=6  │  │depth=6 │ │
        │  │ lr=0.05  │  │ lr=0.05  │  │lr=0.05 │ │
        │  └──────────┘  └──────────┘  └────────┘ │
        │       │             │             │      │
        └───────┼─────────────┼─────────────┼──────┘
                │             │             │
                ▼             ▼             ▼
          pred_xgb      pred_lgb      pred_cat
                │             │             │
                └─────────────┴─────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │    LEVEL 1: META-LEARNER (Ridge)        │
        ├─────────────────────────────────────────┤
        │  Input: [pred_xgb, pred_lgb, pred_cat]  │
        │  Output: final_prediction                │
        │  α = 1.0 (L2 regularization)            │
        └─────────────────────────────────────────┘
                              │
                              ▼
                    FINAL PREDICTION
```

---

## Base Models Specification

### Model 1: XGBoost (Current Baseline)

**Role:** Proven performer, strong baseline

**Hyperparameters:**
```python
xgb_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}
```

**Expected MAE:** 6.10 (baseline)

---

### Model 2: LightGBM

**Role:** Fast, handles sparse features well, different splitting strategy

**Key Differences from XGBoost:**
- **Leaf-wise growth** (vs level-wise in XGBoost)
- **Histogram-based** splitting (faster)
- **Native categorical support**

**Hyperparameters:**
```python
lgb_params = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'num_leaves': 63,  # 2^6 - 1 for depth=6
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

**Expected MAE:** 6.05-6.15 (similar to XGBoost)

---

### Model 3: CatBoost

**Role:** Robust to overfitting, different regularization, ordered boosting

**Key Differences from XGBoost/LightGBM:**
- **Ordered boosting** (reduces overfitting)
- **Symmetric trees** (balanced splits)
- **Built-in categorical encoding**

**Hyperparameters:**
```python
cat_params = {
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 20,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'random_state': 42,
    'thread_count': -1,
    'verbose': False
}
```

**Expected MAE:** 6.00-6.10 (slightly better due to regularization)

---

## Meta-Learner Specification

### Ridge Regression (Recommended)

**Role:** Combine base model predictions with L2 regularization

**Why Ridge?**
- Simple, interpretable
- Prevents overfitting (L2 penalty)
- Learns optimal weights for each base model
- Fast to train

**Hyperparameters:**
```python
meta_params = {
    'alpha': 1.0,  # L2 regularization strength
    'fit_intercept': True,
    'solver': 'auto',
    'random_state': 42
}
```

**Alternative: Weighted Average**
```python
# Simple equal weights
final_pred = (pred_xgb + pred_lgb + pred_cat) / 3

# Or learned weights from validation set
w = optimize_weights(predictions, actuals)
final_pred = w[0]*pred_xgb + w[1]*pred_lgb + w[2]*pred_cat
```

---

## Training Strategy

### Walk-Forward Ensemble Training

**Challenge:** Maintain temporal integrity with two-level stacking

**Solution:** Two-phase walk-forward

#### Phase 1: Base Model Training (On Training Set)

```python
# For each training date:
for train_date in training_dates:
    # Get games today + historical data
    games_today = train_data[train_data['GAME_DATE'] == train_date]
    past_games = train_data[train_data['GAME_DATE'] < train_date]

    # Calculate features and train ALL base models
    X_train, y_train = build_features(past_games, games_today)

    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    cat_model.fit(X_train, y_train)
```

#### Phase 2: Meta-Learner Training (On Training Set)

```python
# Generate base model predictions on training set
train_meta_features = np.column_stack([
    xgb_model.predict(X_train),
    lgb_model.predict(X_train),
    cat_model.predict(X_train)
])

# Train meta-learner
meta_model.fit(train_meta_features, y_train)
```

#### Phase 3: Validation (Walk-Forward on 2024-25)

```python
# For each validation date:
for val_date in validation_dates:
    # Get games today + historical data
    games_today = val_data[val_data['GAME_DATE'] == val_date]
    past_games = all_data[all_data['GAME_DATE'] < val_date]

    # Calculate features
    X_val = build_features(past_games, games_today)

    # Base model predictions
    pred_xgb = xgb_model.predict(X_val)
    pred_lgb = lgb_model.predict(X_val)
    pred_cat = cat_model.predict(X_val)

    # Meta-learner final prediction
    meta_features = np.column_stack([pred_xgb, pred_lgb, pred_cat])
    final_pred = meta_model.predict(meta_features)
```

---

## Expected Performance Gains

### Individual Model Performance (Estimated)

| Model | MAE | Improvement | Notes |
|-------|-----|-------------|-------|
| XGBoost (baseline) | 6.10 | 0% | Current performance |
| LightGBM | 6.05 | +0.8% | Leaf-wise growth advantage |
| CatBoost | 6.00 | +1.6% | Ordered boosting reduces overfitting |

### Ensemble Performance (Estimated)

**Theory:** Ensemble reduces variance by averaging uncorrelated errors

**Diversity Sources:**
1. **Different algorithms**: XGBoost (level-wise), LightGBM (leaf-wise), CatBoost (ordered)
2. **Different regularization**: Each model has unique regularization approach
3. **Different tree structures**: Symmetric vs asymmetric trees

**Expected Correlation Matrix:**
```
           XGB    LGB    CAT
XGB       1.00   0.85   0.82
LGB       0.85   1.00   0.80
CAT       0.82   0.80   1.00
```

**Ensemble MAE Estimate:**
```
If individual MAEs: 6.10, 6.05, 6.00
And correlation: ~0.82
Then ensemble MAE: 5.50-5.70 (9-11% improvement)
```

### Conservative Estimate

**Target:** MAE 6.11 → **5.60-5.80** (5-8% improvement)

**Rationale:**
- Research shows 5-10% improvement from stacking in similar domains
- NBA props have high variance (harder to improve)
- Conservative estimate accounts for implementation challenges

---

## Implementation Plan

### Day 1: Architecture Design ✅ (This Document)

**Deliverables:**
- Architecture specification
- Hyperparameter grid
- Training strategy

---

### Day 2: Implement Base Models

**Tasks:**
1. Create `src/models/ensemble_predictor.py`
2. Implement `TreeEnsemblePredictor` class
3. Add LightGBM and CatBoost models
4. Test on small sample

**File Structure:**
```python
class TreeEnsemblePredictor:
    def __init__(self, xgb_params, lgb_params, cat_params, meta_params):
        self.xgb_model = xgb.XGBRegressor(**xgb_params)
        self.lgb_model = lgb.LGBMRegressor(**lgb_params)
        self.cat_model = cat.CatBoostRegressor(**cat_params)
        self.meta_model = Ridge(**meta_params)

    def fit(self, X, y):
        # Train base models
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.cat_model.fit(X, y)

        # Generate meta features
        meta_X = self._get_meta_features(X)

        # Train meta-learner
        self.meta_model.fit(meta_X, y)

    def predict(self, X):
        meta_X = self._get_meta_features(X)
        return self.meta_model.predict(meta_X)

    def _get_meta_features(self, X):
        return np.column_stack([
            self.xgb_model.predict(X),
            self.lgb_model.predict(X),
            self.cat_model.predict(X)
        ])
```

---

### Day 3-4: Training & Validation

**Tasks:**
1. Create `scripts/training/phase2_week2_tree_ensemble.py`
2. Implement walk-forward training
3. Train on full 2023-24 dataset
4. Validate on 2024-25 season
5. Log to MLflow

**Expected Runtime:** 10-15 minutes (3x slower than single model)

---

### Day 5: Analysis & Comparison

**Tasks:**
1. Compare ensemble vs individual models
2. Analyze base model diversity (correlation)
3. Feature importance from each model
4. Error analysis
5. Document results

---

## Success Criteria

### Minimum Success
- ✅ MAE < 6.00 (any improvement over 6.11)
- ✅ Ensemble outperforms best individual model

### Target Success
- ✅ MAE < 5.70 (7% improvement)
- ✅ Within ±5 pts > 52%

### Stretch Goal
- ✅ MAE < 5.50 (10% improvement)
- ✅ Win rate > 55% (profitable threshold)

---

## Risk Mitigation

### Risk 1: Overfitting
**Mitigation:** Ridge regularization in meta-learner, cross-validation on training set

### Risk 2: High Correlation Between Models
**Mitigation:** Use diverse algorithms, different hyperparameters, monitor correlation matrix

### Risk 3: Longer Training Time
**Mitigation:** Use n_jobs=-1, limit validation dates for testing, full run only at end

### Risk 4: Complex Debugging
**Mitigation:** Save individual model predictions, compare to baseline at each step

---

## Next Steps

1. ✅ Design complete (this document)
2. **Implement `TreeEnsemblePredictor` class**
3. **Create training script**
4. **Run on 2023-24 → 2024-25**
5. **Analyze results**

Ready to implement!
