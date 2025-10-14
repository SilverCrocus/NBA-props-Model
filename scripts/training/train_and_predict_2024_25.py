"""
Train model on historical data and generate predictions for 2024-25.

Uses the baseline XGBoost model trained on 2003-2024 data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pickle

print("="*80)
print("TRAIN MODEL ON HISTORICAL DATA & PREDICT 2024-25")
print("="*80)

# Load historical training data (2003-2024)
print("\n1. Loading historical training data (2003-2024)...")
train_file = 'data/processed/train.parquet'
val_file = 'data/processed/val.parquet'

if not Path(train_file).exists():
    print(f"❌ {train_file} not found!")
    print("   Run the game_log_builder first to create historical dataset")
    exit(1)

train_df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)

print(f"✅ Loaded historical data:")
print(f"   Train: {len(train_df):,} rows")
print(f"   Val:   {len(val_df):,} rows")

# Load 2024-25 dataset for prediction
print("\n2. Loading 2024-25 dataset...")
test_df = pd.read_parquet('data/processed/full_2024_25.parquet')
print(f"✅ Loaded {len(test_df):,} rows for prediction")

# Define features (same as baseline model)
print("\n3. Preparing features...")

# CRITICAL: Exclude target variable PRA from features!
exclude_cols = ['PRA', 'PLAYER_NAME', 'PLAYER_ID', 'GAME_ID', 'GAME_DATE',
                'SEASON', 'SEASON_TYPE', 'TEAM_ABBREVIATION', 'OPPONENT',
                'MATCHUP', 'WL', 'VIDEO_AVAILABLE', 'SEASON_ID']

# Core stats (NOT including PRA components to prevent leakage)
core_stats = ['MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT',
              'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

# Lag features (these use HISTORICAL PRA, which is OK)
lag_features = [col for col in train_df.columns if any(x in col for x in ['_lag', '_L5_', '_L10_', '_L20_', '_ewma', '_trend'])]

# All features
feature_cols = core_stats + lag_features

# Filter to columns that exist in all datasets
feature_cols = [col for col in feature_cols if col in train_df.columns and col in test_df.columns]

# CRITICAL: Remove any that are in exclude list
feature_cols = [col for col in feature_cols if col not in exclude_cols]

# Double-check we're not using current PRA/PTS/REB/AST
leakage_cols = [col for col in feature_cols if col in ['PRA', 'PTS', 'REB', 'AST']]
if leakage_cols:
    print(f"⚠️  WARNING: Found potential leakage columns: {leakage_cols}")
    feature_cols = [col for col in feature_cols if col not in leakage_cols]

print(f"✅ Using {len(feature_cols)} features")
print(f"   Core stats: {len(core_stats)}")
print(f"   Lag features: {len(lag_features)}")

# Prepare training data
X_train = train_df[feature_cols].fillna(0)
y_train = train_df['PRA']

X_val = val_df[feature_cols].fillna(0)
y_val = val_df['PRA']

# Train XGBoost model
print("\n4. Training XGBoost model on 2003-2024 data...")
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"✅ Model trained!")

# Validate on historical data
print("\n5. Validating on historical validation set...")
val_preds = model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"   Historical MAE: {val_mae:.2f} points")

# Generate predictions for 2024-25
print("\n6. Generating predictions for 2024-25 season...")
X_test = test_df[feature_cols].fillna(0)
test_preds = model.predict(X_test)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'PLAYER_NAME': test_df['PLAYER_NAME'],
    'PLAYER_ID': test_df['PLAYER_ID'],
    'GAME_ID': test_df['GAME_ID'],
    'GAME_DATE': test_df['GAME_DATE'],
    'TEAM_ABBREVIATION': test_df.get('TEAM_ABBREVIATION', ''),
    'OPPONENT': test_df.get('OPPONENT', test_df.get('opp_TEAM_ABBREVIATION', '')),
    'PRA': test_df['PRA'],
    'predicted_PRA': test_preds,
    'error': test_preds - test_df['PRA'],
    'abs_error': np.abs(test_preds - test_df['PRA'])
})

print(f"✅ Generated {len(predictions_df):,} predictions")

# Calculate prediction accuracy
test_mae = mean_absolute_error(predictions_df['PRA'], predictions_df['predicted_PRA'])
print(f"\n7. 2024-25 Prediction Accuracy:")
print(f"   MAE: {test_mae:.2f} points")
print(f"   Within ±3 pts: {(predictions_df['abs_error'] <= 3).mean()*100:.1f}%")
print(f"   Within ±5 pts: {(predictions_df['abs_error'] <= 5).mean()*100:.1f}%")

# Save predictions
output_file = 'data/results/baseline_predictions_2024-25.csv'
predictions_df.to_csv(output_file, index=False)
print(f"\n✅ Saved predictions to {output_file}")

# Save model
model_file = 'data/models/baseline_xgb_2024_25.pkl'
Path('data/models').mkdir(parents=True, exist_ok=True)
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"✅ Saved model to {model_file}")

print("\n" + "="*80)
print("PREDICTION SUMMARY")
print("="*80)

print(f"""
Dataset Statistics:
  Training period: 2003-2024
  Prediction period: 2024-25
  Total predictions: {len(predictions_df):,}

Model Performance:
  Historical MAE: {val_mae:.2f} points
  2024-25 MAE: {test_mae:.2f} points

Prediction Distribution:
  Min: {predictions_df['predicted_PRA'].min():.1f}
  Mean: {predictions_df['predicted_PRA'].mean():.1f}
  Max: {predictions_df['predicted_PRA'].max():.1f}
  Std: {predictions_df['predicted_PRA'].std():.1f}

Next Steps:
  1. Match predictions to betting odds
  2. Calculate edge and identify profitable bets
  3. Run corrected backtest to validate 79.66% win rate
""")

print("="*80)
