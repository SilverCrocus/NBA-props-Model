#!/usr/bin/env python3
"""
Test script to demonstrate realistic model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("NBA PROPS MODEL - REALISTIC PERFORMANCE TEST")
    print("="*60)

    # Load clean data
    data_path = Path('/Users/diyagamah/Documents/nba_props_model/data/processed')
    df = pd.read_csv(data_path / 'player_features_2023_24_clean.csv')

    print(f"\nDataset shape: {df.shape}")

    # Use only numeric features for this test
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['PRA_estimate', 'PRA_estimate_noisy']]

    print(f"Using {len(feature_cols)} numeric features")

    # Prepare data with noisy target
    X = df[feature_cols].copy()
    y = df['PRA_estimate_noisy'].copy()  # Use noisy version

    print(f"\nTarget statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Range: {y.min():.2f} - {y.max():.2f}")

    # Three-way split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define SIMPLE models appropriate for 503 samples
    models = {
        'Ridge (alpha=10)': Ridge(alpha=10.0, random_state=42),
        'Ridge (alpha=5)': Ridge(alpha=5.0, random_state=42),
        'Lasso (alpha=1)': Lasso(alpha=1.0, random_state=42),
        'XGBoost (simple)': xgb.XGBRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.5,
            colsample_bytree=0.5,
            reg_alpha=5.0,
            reg_lambda=5.0,
            random_state=42,
            verbose=0
        )
    }

    print("\n" + "="*60)
    print("MODEL TRAINING AND VALIDATION")
    print("="*60)

    results = {}

    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)

        # Train
        if 'XGBoost' in name:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_val = model.predict(X_val_scaled)
            y_pred_test = model.predict(X_test_scaled)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        test_r2 = r2_score(y_test, y_pred_test)

        overfit_ratio = val_mae / train_mae

        results[name] = {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'overfit_ratio': overfit_ratio
        }

        print(f"  Train: MAE={train_mae:.3f}, R²={train_r2:.3f}")
        print(f"  Val:   MAE={val_mae:.3f}, R²={val_r2:.3f}")
        print(f"  Test:  MAE={test_mae:.3f}, R²={test_r2:.3f}")
        print(f"  Overfit ratio: {overfit_ratio:.2f}")

        if overfit_ratio > 1.3:
            print("  ⚠️  Model shows signs of overfitting")
        if test_r2 > 0.6:
            print("  ⚠️  Suspiciously high R² - check for issues")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Find best model based on validation
    best_model = min(results.keys(), key=lambda x: results[x]['val_mae'])
    print(f"\nBest model: {best_model}")
    print(f"  Val MAE: {results[best_model]['val_mae']:.3f}")
    print(f"  Test MAE: {results[best_model]['test_mae']:.3f}")
    print(f"  Test R²: {results[best_model]['test_r2']:.3f}")

    # Reality check
    print("\n" + "="*60)
    print("PERFORMANCE REALITY CHECK")
    print("="*60)

    baseline_mae = np.mean(np.abs(y_test - y_test.mean()))
    best_test_mae = results[best_model]['test_mae']
    improvement = (baseline_mae - best_test_mae) / baseline_mae * 100

    print(f"\nBaseline MAE (predict mean): {baseline_mae:.3f}")
    print(f"Best model MAE: {best_test_mae:.3f}")
    print(f"Improvement over baseline: {improvement:.1f}%")
    print(f"MAE as % of target mean: {best_test_mae / y_test.mean() * 100:.1f}%")

    # Expected performance
    print("\n" + "="*60)
    print("EXPECTED VS ACTUAL PERFORMANCE")
    print("="*60)

    print("\nExpected for NBA PRA prediction:")
    print("  R²: 0.35-0.55 (good models)")
    print("  MAE: 1.5-2.5 (for mean PRA ~9)")
    print("  MAPE: 20-30%")

    print(f"\nYour model achieved:")
    print(f"  R²: {results[best_model]['test_r2']:.3f}", end="")
    if results[best_model]['test_r2'] > 0.6:
        print(" [TOO HIGH - likely overfitting]")
    elif results[best_model]['test_r2'] > 0.35:
        print(" [Good - realistic range]")
    else:
        print(" [Acceptable]")

    print(f"  MAE: {best_test_mae:.3f}", end="")
    if best_test_mae < 1.0:
        print(" [TOO LOW - check for data leakage]")
    elif best_test_mae < 2.5:
        print(" [Good - realistic range]")
    else:
        print(" [Needs improvement]")

    mape = np.mean(np.abs((y_test - results[best_model]['test_mae']) / y_test)) * 100
    print(f"  MAPE: ~{mape:.0f}%", end="")
    if mape < 15:
        print(" [TOO LOW - suspicious]")
    elif mape < 30:
        print(" [Good - realistic range]")
    else:
        print(" [Acceptable]")

if __name__ == "__main__":
    main()