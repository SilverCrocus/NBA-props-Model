# NBA Props Model - Simple Usage Guide

## Prerequisites

Make sure you have fetched the game data:
```bash
uv run scripts/fetch_all_game_logs.py
```

This downloads all NBA game data (takes ~15-30 minutes, only need to do once).

## Model Evaluation

Run the complete evaluation with real NBA game data:

```bash
uv run scripts/evaluate_models_real_data.py
```

That's it! This single command will:
1. Load real game logs (28,000+ games)
2. Engineer 100+ features from actual games
3. Train multiple models (Ridge, XGBoost, LightGBM, etc.)
4. Evaluate performance on future games
5. Create visualizations
6. Save results to `data/model_results/`

## Expected Performance

When the script finishes, you should see:
- **MAE**: 4-6 points (good), 6-8 points (normal)
- **R²**: 0.35-0.50 (good), 0.50-0.60 (very good)
- **MAPE**: 60-80% (normal for NBA)

These are realistic numbers for predicting NBA player performance.

## Output Files

After running, check these files:
- `data/model_results/model_results_real_data.csv` - Model performance metrics
- `data/model_results/feature_importance_real_data.csv` - Important features
- `data/model_results/model_evaluation_real_data.png` - Visualizations

## What Makes This Real

Unlike synthetic models that achieve fake R² = 0.99:
- Uses actual game-by-game data
- Predicts real future performance
- Accounts for player variance
- Useful for actual betting decisions

## Troubleshooting

**"Game logs not found"**
```bash
uv run scripts/fetch_all_game_logs.py  # Fetch the data first
```

**"nba-api not installed"**
```bash
uv add nba-api
```

That's all you need to know! Keep it simple.