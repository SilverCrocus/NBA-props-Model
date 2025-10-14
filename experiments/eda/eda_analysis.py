"""
Exploratory Data Analysis on Walk-Forward Validation Results
Identifies data quality issues and performance patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the data
print("="*80)
print("LOADING DATASETS")
print("="*80)

predictions_path = '/Users/diyagamah/Documents/nba_props_model/data/results/walkforward_predictions_2024-25.csv'
backtest_path = '/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2024_25.csv'
features_path = '/Users/diyagamah/Documents/nba_props_model/data/processed/full_2024_25.parquet'

predictions_df = pd.read_csv(predictions_path)
backtest_df = pd.read_csv(backtest_path)
features_df = pd.read_parquet(features_path)

print(f"Predictions shape: {predictions_df.shape}")
print(f"Backtest shape: {backtest_df.shape}")
print(f"Features shape: {features_df.shape}")

# ============================================================================
# 1. BASIC DATA STRUCTURE INSPECTION
# ============================================================================
print("\n" + "="*80)
print("1. PREDICTIONS DATASET STRUCTURE")
print("="*80)
print("\nColumns:")
print(predictions_df.columns.tolist())
print("\nFirst few rows:")
print(predictions_df.head())
print("\nData types:")
print(predictions_df.dtypes)
print("\nBasic statistics:")
print(predictions_df.describe())

print("\n" + "="*80)
print("2. BACKTEST DATASET STRUCTURE")
print("="*80)
print("\nColumns:")
print(backtest_df.columns.tolist())
print("\nFirst few rows:")
print(backtest_df.head())
print("\nData types:")
print(backtest_df.dtypes)
print("\nBasic statistics:")
print(backtest_df.describe())

# ============================================================================
# 3. PREDICTION QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. PREDICTION DISTRIBUTION ANALYSIS")
print("="*80)

if 'prediction' in predictions_df.columns:
    pred_col = 'prediction'
elif 'predicted_pra' in predictions_df.columns:
    pred_col = 'predicted_pra'
elif 'predicted_PRA' in predictions_df.columns:
    pred_col = 'predicted_PRA'
else:
    pred_cols = [col for col in predictions_df.columns if 'pred' in col.lower()]
    pred_col = pred_cols[0] if pred_cols else None

if 'actual' in predictions_df.columns:
    actual_col = 'actual'
elif 'actual_pra' in predictions_df.columns:
    actual_col = 'actual_pra'
elif 'PRA' in predictions_df.columns:
    actual_col = 'PRA'
else:
    actual_cols = [col for col in predictions_df.columns if 'actual' in col.lower() or col == 'PRA']
    actual_col = actual_cols[0] if actual_cols else None

print(f"\nPrediction column: {pred_col}")
print(f"Actual column: {actual_col}")

predictions = predictions_df[pred_col]
actuals = predictions_df[actual_col]

print(f"\nPrediction Statistics:")
print(f"  Count: {predictions.count()}")
print(f"  Mean: {predictions.mean():.2f}")
print(f"  Median: {predictions.median():.2f}")
print(f"  Std Dev: {predictions.std():.2f}")
print(f"  Min: {predictions.min():.2f}")
print(f"  Max: {predictions.max():.2f}")
print(f"  25th percentile: {predictions.quantile(0.25):.2f}")
print(f"  75th percentile: {predictions.quantile(0.75):.2f}")

print(f"\nActual Statistics:")
print(f"  Count: {actuals.count()}")
print(f"  Mean: {actuals.mean():.2f}")
print(f"  Median: {actuals.median():.2f}")
print(f"  Std Dev: {actuals.std():.2f}")
print(f"  Min: {actuals.min():.2f}")
print(f"  Max: {actuals.max():.2f}")

# Check for impossible values
print(f"\n⚠️  DATA QUALITY CHECKS:")
print(f"  Negative predictions: {(predictions < 0).sum()}")
print(f"  Predictions > 100: {(predictions > 100).sum()}")
print(f"  Missing predictions: {predictions.isna().sum()}")
print(f"  Missing actuals: {actuals.isna().sum()}")

# Outlier detection using IQR method
Q1 = predictions.quantile(0.25)
Q3 = predictions.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
outliers = ((predictions < lower_bound) | (predictions > upper_bound)).sum()
print(f"  Prediction outliers (3*IQR): {outliers} ({outliers/len(predictions)*100:.2f}%)")

# Check distribution normality
from scipy import stats
_, p_value = stats.shapiro(predictions.sample(min(5000, len(predictions))))
print(f"  Shapiro-Wilk normality test p-value: {p_value:.4f}")
print(f"    → Predictions are {'NOT ' if p_value < 0.05 else ''}normally distributed")

# ============================================================================
# 4. ERROR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. PREDICTION ERROR ANALYSIS")
print("="*80)

predictions_df['error'] = predictions_df[pred_col] - predictions_df[actual_col]
predictions_df['abs_error'] = np.abs(predictions_df['error'])
predictions_df['pct_error'] = (predictions_df['error'] / predictions_df[actual_col]) * 100

mae = predictions_df['abs_error'].mean()
rmse = np.sqrt((predictions_df['error']**2).mean())
mape = predictions_df['pct_error'].abs().mean()
bias = predictions_df['error'].mean()

print(f"\nOverall Error Metrics:")
print(f"  MAE (Mean Absolute Error): {mae:.3f}")
print(f"  RMSE (Root Mean Squared Error): {rmse:.3f}")
print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"  Bias (Mean Error): {bias:.3f}")
print(f"    → Model {'OVER' if bias > 0 else 'UNDER'}-predicts by {abs(bias):.3f} points on average")

# Check for systematic bias
overpredict = (predictions_df['error'] > 0).sum()
underpredict = (predictions_df['error'] < 0).sum()
print(f"\n  Over-predictions: {overpredict} ({overpredict/len(predictions_df)*100:.1f}%)")
print(f"  Under-predictions: {underpredict} ({underpredict/len(predictions_df)*100:.1f}%)")

# Error by prediction range
print(f"\nError by Prediction Range:")
predictions_df['pred_range'] = pd.cut(predictions_df[pred_col],
                                      bins=[0, 10, 20, 30, 40, 50, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50+'])
error_by_range = predictions_df.groupby('pred_range').agg({
    'abs_error': ['mean', 'std', 'count'],
    'error': 'mean'
})
print(error_by_range)

# ============================================================================
# 5. PLAYER-LEVEL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. PLAYER-LEVEL ERROR PATTERNS")
print("="*80)

player_cols = [col for col in predictions_df.columns if 'player' in col.lower() and 'id' not in col.lower()]
if player_cols:
    player_col = player_cols[0]

    player_errors = predictions_df.groupby(player_col).agg({
        'abs_error': ['mean', 'count'],
        'error': 'mean',
        pred_col: 'mean',
        actual_col: 'mean'
    }).round(3)

    player_errors.columns = ['MAE', 'Count', 'Bias', 'Avg_Pred', 'Avg_Actual']
    player_errors = player_errors[player_errors['Count'] >= 5]  # At least 5 predictions

    print(f"\nPlayers with HIGHEST MAE (min 5 predictions):")
    print(player_errors.nlargest(10, 'MAE'))

    print(f"\nPlayers with LOWEST MAE (min 5 predictions):")
    print(player_errors.nsmallest(10, 'MAE'))

    print(f"\nPlayers with HIGHEST POSITIVE BIAS (over-predicted):")
    print(player_errors.nlargest(10, 'Bias')[['Bias', 'MAE', 'Count']])

    print(f"\nPlayers with HIGHEST NEGATIVE BIAS (under-predicted):")
    print(player_errors.nsmallest(10, 'Bias')[['Bias', 'MAE', 'Count']])

# ============================================================================
# 6. TEAM-LEVEL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. TEAM-LEVEL ERROR PATTERNS")
print("="*80)

team_cols = [col for col in predictions_df.columns if 'team' in col.lower()]
if team_cols:
    team_col = team_cols[0]

    team_errors = predictions_df.groupby(team_col).agg({
        'abs_error': ['mean', 'count'],
        'error': 'mean',
        pred_col: 'mean'
    }).round(3)

    team_errors.columns = ['MAE', 'Count', 'Bias', 'Avg_Pred']

    print(f"\nTeams with HIGHEST MAE:")
    print(team_errors.nlargest(10, 'MAE'))

    print(f"\nTeams with LOWEST MAE:")
    print(team_errors.nsmallest(10, 'MAE'))

# ============================================================================
# 7. EDGE CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. EDGE CALIBRATION ANALYSIS")
print("="*80)

if 'edge' in backtest_df.columns:
    print(f"\nBacktest Records: {len(backtest_df)}")

    # Determine result column
    result_col = None
    if 'result' in backtest_df.columns:
        result_col = 'result'
    elif 'bet_result' in backtest_df.columns:
        result_col = 'bet_result'

    # Overall betting performance
    total_bets = len(backtest_df)
    if 'bet_won' in backtest_df.columns:
        wins = backtest_df['bet_won'].sum()
        losses = backtest_df['bet_lost'].sum()
        pushes = backtest_df['bet_pushed'].sum()
    elif result_col:
        wins = (backtest_df[result_col] > 0).sum()
        losses = (backtest_df[result_col] < 0).sum()
        pushes = (backtest_df[result_col] == 0).sum()
    else:
        wins = losses = pushes = 0

    win_rate = wins / total_bets * 100 if total_bets > 0 else 0

    print(f"\nOverall Betting Performance:")
    print(f"  Total bets: {total_bets}")
    print(f"  Wins: {wins} ({win_rate:.2f}%)")
    print(f"  Losses: {losses} ({(losses/total_bets)*100:.2f}%)")
    if pushes > 0:
        print(f"  Pushes: {pushes} ({(pushes/total_bets)*100:.2f}%)")

    if 'profit' in backtest_df.columns:
        total_profit = backtest_df['profit'].sum()
        roi = (total_profit / total_bets) * 100
        print(f"\n  Total profit: {total_profit:.2f} units")
        print(f"  ROI: {roi:.2f}%")

    # Edge calibration by buckets
    print(f"\nPerformance by Edge Bucket:")
    backtest_df['edge_bucket'] = pd.cut(backtest_df['edge'],
                                        bins=[-100, -10, -5, 0, 5, 10, 100],
                                        labels=['< -10%', '-10 to -5%', '-5 to 0%',
                                               '0 to 5%', '5 to 10%', '> 10%'])

    # Build aggregation dict dynamically
    agg_dict = {
        'edge': 'mean',
        'bet_won': ['count', 'sum', 'mean']
    }
    if result_col:
        agg_dict[result_col] = 'sum'

    edge_analysis = backtest_df.groupby('edge_bucket').agg(agg_dict)
    print(edge_analysis)

    # Predicted edge vs actual edge
    if 'bet_won' in backtest_df.columns:
        edge_comparison = backtest_df.groupby('edge_bucket').agg({
            'edge': 'mean',
            'bet_won': lambda x: x.mean() * 100  # win rate
        }).round(2)
        edge_comparison.columns = ['Predicted_Edge_%', 'Actual_Win_Rate_%']
    elif result_col:
        edge_comparison = backtest_df.groupby('edge_bucket').agg({
            'edge': 'mean',
            result_col: lambda x: (x > 0).mean() * 100  # win rate
        }).round(2)
        edge_comparison.columns = ['Predicted_Edge_%', 'Actual_Win_Rate_%']
    else:
        edge_comparison = None

    if edge_comparison is not None:
        edge_comparison['Expected_Win_Rate_%'] = edge_comparison['Predicted_Edge_%'] + 50
        edge_comparison['Calibration_Error'] = edge_comparison['Actual_Win_Rate_%'] - edge_comparison['Expected_Win_Rate_%']

        print(f"\nEdge Calibration Analysis:")
        print(edge_comparison)

# ============================================================================
# 8. FEATURE QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. FEATURE QUALITY ANALYSIS")
print("="*80)

print(f"\nFeature dataset shape: {features_df.shape}")
print(f"Number of features: {len(features_df.columns)}")

# Missing value analysis
missing_rates = (features_df.isna().sum() / len(features_df) * 100).sort_values(ascending=False)
high_missing = missing_rates[missing_rates > 5]

print(f"\nFeatures with >5% missing values: {len(high_missing)}")
if len(high_missing) > 0:
    print("\nTop features with missing values:")
    print(high_missing.head(20))

# Zero variance features
zero_var_features = []
for col in features_df.select_dtypes(include=[np.number]).columns:
    if features_df[col].std() == 0 or features_df[col].nunique() == 1:
        zero_var_features.append(col)

print(f"\nFeatures with zero variance: {len(zero_var_features)}")
if zero_var_features:
    print(zero_var_features)

# Infinite values
inf_features = []
for col in features_df.select_dtypes(include=[np.number]).columns:
    if np.isinf(features_df[col]).any():
        inf_count = np.isinf(features_df[col]).sum()
        inf_features.append((col, inf_count))

print(f"\nFeatures with infinite values: {len(inf_features)}")
if inf_features:
    print("Top features with infinite values:")
    for feat, count in sorted(inf_features, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {count} infinite values")

# Feature correlation with target
if 'pra' in features_df.columns or 'target' in features_df.columns:
    target_col = 'pra' if 'pra' in features_df.columns else 'target'
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f != target_col]

    correlations = []
    for feat in numeric_features[:50]:  # Top 50 features for speed
        try:
            corr = features_df[[feat, target_col]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append((feat, corr))
        except:
            pass

    correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop 10 features correlated with target:")
    for feat, corr in correlations[:10]:
        print(f"  {feat}: {corr:.3f}")

    print(f"\nBottom 10 features correlated with target (weakest):")
    for feat, corr in correlations[-10:]:
        print(f"  {feat}: {corr:.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
