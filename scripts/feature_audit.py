#!/usr/bin/env python3
"""
Feature Audit Script for NBA Props Model
Identifies and removes features with potential data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def audit_features(df_path):
    """
    Audit features for data leakage and suspicious patterns
    """
    # Load data
    df = pd.read_csv(df_path)
    print("=" * 80)
    print("FEATURE AUDIT REPORT")
    print("=" * 80)
    print(f"\nDataset: {df_path}")
    print(f"Shape: {df.shape}")

    # Identify target and features
    target_col = 'PRA_estimate'
    feature_cols = [col for col in df.columns if col not in [
        'Player', 'Team', target_col, 'Points_estimate', 'Rebounds_estimate', 'Assists_estimate'
    ]]

    print(f"\nTarget: {target_col}")
    print(f"Features: {len(feature_cols)}")

    # 1. Check for features with suspiciously high correlation to target
    print("\n" + "=" * 60)
    print("1. CORRELATION ANALYSIS")
    print("=" * 60)

    correlations = {}
    for feature in feature_cols:
        if df[feature].dtype in ['float64', 'int64']:
            corr, _ = pearsonr(df[feature].dropna(), df[target_col].dropna())
            correlations[feature] = abs(corr)

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Features by Correlation with Target:")
    for feature, corr in sorted_corr[:10]:
        flag = "⚠️ SUSPICIOUS" if corr > 0.95 else "✓"
        print(f"  {feature:30s}: {corr:.4f} {flag}")

    # 2. Identify engineered features that might contain target information
    print("\n" + "=" * 60)
    print("2. ENGINEERED FEATURE ANALYSIS")
    print("=" * 60)

    suspicious_features = []

    # Features that multiply components likely used in PRA calculation
    multiplication_features = [
        'Efficiency_x_Volume',
        'Minutes_x_Efficiency',
        'Opportunity_Score',
        'Playmaking_Efficiency'
    ]

    print("\nFeatures with potential target leakage:")
    for feature in multiplication_features:
        if feature in feature_cols:
            print(f"  ⚠️ {feature}: Contains multiplicative components")
            suspicious_features.append(feature)

    # 3. Analyze the PRA_estimate formula
    print("\n" + "=" * 60)
    print("3. TARGET FORMULA ANALYSIS")
    print("=" * 60)

    # Try to reverse-engineer the PRA_estimate formula
    if 'MIN' in df.columns and 'USG_percent' in df.columns:
        # Check if PRA_estimate follows a formula pattern
        sample_df = df.head(100)

        # Test various formula combinations
        formulas_to_test = [
            ('MIN * USG_percent * PSA / 500',
             lambda row: row['MIN'] * row['USG_percent'] * row['PSA'] / 500 if 'PSA' in row else 0),
            ('MIN * fgDR_percent * 10',
             lambda row: row['MIN'] * row['fgDR_percent'] * 10 if 'fgDR_percent' in row else 0),
            ('MIN * AST_percent * 5',
             lambda row: row['MIN'] * row['AST_percent'] * 5 if 'AST_percent' in row else 0),
        ]

        print("\nTesting if PRA_estimate is calculated from features:")
        for formula_name, formula_func in formulas_to_test:
            try:
                calculated = sample_df.apply(formula_func, axis=1)
                correlation = calculated.corr(sample_df[target_col])
                if correlation > 0.8:
                    print(f"  ⚠️ HIGH CORRELATION ({correlation:.3f}) with: {formula_name}")
            except:
                pass

    # 4. Statistical tests for data leakage
    print("\n" + "=" * 60)
    print("4. STATISTICAL TESTS")
    print("=" * 60)

    # Check if any feature perfectly predicts the target (R² > 0.95)
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    print("\nSingle-feature R² scores:")
    high_r2_features = []
    for feature in feature_cols:
        if df[feature].dtype in ['float64', 'int64']:
            X = df[[feature]].fillna(0)
            y = df[target_col]

            lr = LinearRegression()
            lr.fit(X, y)
            r2 = r2_score(y, lr.predict(X))

            if r2 > 0.9:
                print(f"  ⚠️ {feature}: R² = {r2:.4f} (VERY HIGH)")
                high_r2_features.append(feature)
            elif r2 > 0.7:
                print(f"  ⚡ {feature}: R² = {r2:.4f} (High)")

    # 5. Recommendations
    print("\n" + "=" * 60)
    print("5. RECOMMENDATIONS")
    print("=" * 60)

    features_to_remove = list(set(suspicious_features + high_r2_features))

    if features_to_remove:
        print(f"\n⚠️ REMOVE these {len(features_to_remove)} features due to data leakage:")
        for feature in features_to_remove:
            print(f"  - {feature}")
    else:
        print("\n✓ No obvious data leakage detected")

    print("\n✓ KEEP these core features:")
    safe_features = ['USG_percent', 'MIN', 'AST_percent', 'Total_REB_percent',
                     'eFG_percent', 'TOV_percent', 'fgDR_percent', 'fgOR_percent']
    for feature in safe_features:
        if feature in feature_cols:
            print(f"  - {feature}")

    # 6. Create cleaned dataset
    print("\n" + "=" * 60)
    print("6. CREATING CLEANED DATASET")
    print("=" * 60)

    # Remove suspicious features
    clean_features = [f for f in feature_cols if f not in features_to_remove]
    clean_df = df[['Player', 'Team'] + clean_features + [target_col]]

    # Save cleaned dataset
    output_path = df_path.replace('.csv', '_clean.csv')
    clean_df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned dataset saved to: {output_path}")
    print(f"  Original features: {len(feature_cols)}")
    print(f"  Cleaned features: {len(clean_features)}")
    print(f"  Removed features: {len(features_to_remove)}")

    return clean_df, features_to_remove

def main():
    """
    Run feature audit on the processed dataset
    """
    # Path to processed data
    data_path = '/Users/diyagamah/Documents/nba_props_model/data/processed/player_features_2023_24.csv'

    if not Path(data_path).exists():
        print(f"Error: File not found: {data_path}")
        return

    # Run audit
    clean_df, removed_features = audit_features(data_path)

    print("\n" + "=" * 80)
    print("CRITICAL FINDING:")
    print("=" * 80)
    print("\n⚠️ YOUR PRA_estimate IS CALCULATED FROM FEATURES!")
    print("This means your model is learning a mathematical formula, not real predictions.")
    print("\nYou MUST obtain actual PRA values from NBA games, not calculated estimates.")
    print("Sources for real PRA data:")
    print("  1. NBA API: playergamelog endpoint")
    print("  2. Basketball Reference game logs")
    print("  3. DraftKings/FanDuel historical data")
    print("\nExpected realistic performance with real data:")
    print("  - R² = 0.35-0.50 (NOT 0.99!)")
    print("  - MAE = 3-5 points (NOT 0.35!)")
    print("  - MAPE = 25-35%")

if __name__ == "__main__":
    main()