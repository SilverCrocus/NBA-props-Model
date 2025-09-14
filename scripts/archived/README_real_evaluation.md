# Real Data Model Evaluation Scripts

## Overview
These scripts replace the synthetic notebook evaluation with REAL NBA game data analysis. No fallback logic, no made-up values - just actual game-by-game performance.

## Scripts

### 1. `feature_engineering_real_data.py`
Comprehensive feature engineering from real game logs.

```bash
uv run scripts/feature_engineering_real_data.py
```

**Features Created:**
- **Tier 1 - Core Performance**: Rolling averages, EWMA, usage rates, efficiency metrics
- **Tier 2 - Contextual Modulators**: Rest days, home/away, opponent quality, season phase
- **Tier 3 - Temporal Dynamics**: Trends, volatility, momentum, consistency scores

**Output:**
- `data/processed/training_data_real_2023-24.csv`
- 100+ engineered features from real games
- Target: Next game PRA (not synthetic estimates)

### 2. `evaluate_models_real_data.py`
Complete model evaluation pipeline using real data.

```bash
uv run scripts/evaluate_models_real_data.py
```

**What it does:**
1. Loads real game logs (no synthetic data)
2. Engineers features from actual games
3. Trains multiple models (Ridge, Lasso, XGBoost, LightGBM, etc.)
4. Evaluates on temporal test set
5. Creates visualizations
6. Compares with synthetic results

**Output:**
- `data/model_results/model_results_real_data.csv` - Performance metrics
- `data/model_results/feature_importance_real_data.csv` - Feature importance
- `data/model_results/model_evaluation_real_data.png` - Visualizations

## Key Differences from Notebook

| Aspect | Notebook (Synthetic) | Scripts (Real Data) |
|--------|---------------------|-------------------|
| **Data Source** | CTG season averages | NBA game logs |
| **Target** | PRA_estimate (formula) | Actual next game PRA |
| **Sample Size** | 503 players | 30,000+ games |
| **Features** | 21 engineered features | 100+ temporal features |
| **R² Score** | 0.996 (FAKE!) | 0.35-0.45 (REAL) |
| **MAE** | 0.356 (IMPOSSIBLE!) | 4-6 points (EXPECTED) |
| **Validation** | Random split | Temporal split |

## Expected Results

### Synthetic (Notebook) - UNREALISTIC
```
R² = 0.996
MAE = 0.356
MAPE = 10%
```
**Why it's fake:** The target was calculated from the features!

### Real Data (Scripts) - REALISTIC
```
R² = 0.35-0.45
MAE = 4-6 points
MAPE = 25-35%
```
**Why it's real:** Predicting actual future game performance

## Feature Categories

### Core Performance (Tier 1)
- `PRA_L3`, `PRA_L5`, `PRA_L10`, `PRA_L15` - Rolling averages
- `PRA_EMA5`, `PRA_EMA10` - Exponential weighted averages
- `usage_L5`, `TS_L5`, `eFG_L5` - Efficiency metrics
- `AST_TO_L5`, `REB_rate_L5` - Performance ratios

### Contextual Modulators (Tier 2)
- `days_rest`, `rest_category` - Rest patterns
- `is_home`, `PRA_home_L10`, `PRA_away_L10` - Home/away splits
- `season_progress`, `season_phase` - Time of season
- `games_played`, `games_played_pct` - Experience

### Temporal Dynamics (Tier 3)
- `PRA_trend_5`, `PRA_trend_10` - Performance trends
- `PRA_std_5`, `PRA_cv_5` - Volatility measures
- `PRA_momentum_5v15` - Momentum indicators
- `hot_streak`, `consistency_score` - Form indicators

## Model Performance Comparison

### Models Tested
1. **Baseline** - Last 5 games average
2. **Ridge Regression** - With different alpha values
3. **Lasso** - L1 regularization
4. **ElasticNet** - Combined L1/L2
5. **XGBoost** - Gradient boosting
6. **LightGBM** - Fast gradient boosting
7. **Random Forest** - Ensemble of trees

### Expected Rankings (Real Data)
```
1. XGBoost/LightGBM - MAE ~4.5-5.0
2. Random Forest - MAE ~5.0-5.5
3. Ridge Regression - MAE ~5.5-6.0
4. Baseline (L5 Avg) - MAE ~5.5-6.5
```

## Validation Strategy

### Temporal Split (Used in Scripts)
```
Training:   70% (early season games)
Validation: 15% (mid-season games)
Test:       15% (late season games)
```
**Why:** Never use future games to predict past (no data leakage)

### Random Split (Used in Notebook - WRONG!)
```
Training:   80% (random games)
Test:       20% (random games)
```
**Problem:** Uses future information to predict past

## Running the Complete Pipeline

```bash
# Step 1: Ensure game logs exist
ls data/game_logs/*.csv

# Step 2: Create engineered features
uv run scripts/feature_engineering_real_data.py

# Step 3: Run model evaluation
uv run scripts/evaluate_models_real_data.py

# Step 4: Check results
cat data/model_results/model_results_real_data.csv
```

## Interpreting Results

### Good Performance (Real Data)
- MAE: 4-6 points
- R²: 0.35-0.45
- MAPE: 25-35%
- Test ≈ Validation performance

### Overfitting Signs
- Test MAE >> Validation MAE
- R² > 0.6 (too good to be true)
- Complex models beating simple baseline by >50%

### What the Metrics Mean
- **MAE = 5**: Average error of 5 PRA points per game
- **R² = 0.40**: Model explains 40% of variance (good for sports!)
- **MAPE = 30%**: 30% average percentage error

## Why This Matters for Betting

### Synthetic Model (Notebook)
- Predicts season averages perfectly
- Useless for game-by-game betting
- Would lose money consistently

### Real Model (Scripts)
- Predicts actual game outcomes
- Accounts for recent form, rest, context
- Can identify value in betting lines

## Common Issues

### "FileNotFoundError: Game logs not found"
```bash
# Run the data fetching script first
uv run scripts/fetch_all_game_logs.py
```

### "Not enough samples"
- Need at least 20 games per player
- 2023-24 season should have ~30,000 games

### "Memory error"
- Reduce feature windows
- Use fewer rolling averages
- Process in chunks

## Next Steps

1. **Add more features**:
   - Opponent defensive rating
   - Injury reports
   - Lineup combinations

2. **Improve validation**:
   - Walk-forward analysis
   - Betting line backtesting
   - Out-of-sample season testing

3. **Production deployment**:
   - Real-time data updates
   - API for predictions
   - Monitoring and alerts