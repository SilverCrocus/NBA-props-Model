"""
Create visualizations for EDA analysis
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
plt.rcParams['figure.figsize'] = (14, 10)

# Load the data
print("Loading datasets...")
predictions_df = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/walkforward_predictions_2024-25.csv')
backtest_df = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/backtest_walkforward_2024_25.csv')
features_df = pd.read_parquet('/Users/diyagamah/Documents/nba_props_model/data/processed/full_2024_25.parquet')

# Create output directory
output_dir = Path('/Users/diyagamah/Documents/nba_props_model/data/results/eda_plots')
output_dir.mkdir(exist_ok=True, parents=True)

# Column names
pred_col = 'predicted_PRA'
actual_col = 'PRA'
predictions_df['error'] = predictions_df[pred_col] - predictions_df[actual_col]
predictions_df['abs_error'] = np.abs(predictions_df['error'])

print("\nGenerating visualizations...")

# ============================================================================
# 1. PREDICTION VS ACTUAL SCATTER PLOT
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Scatter plot
ax = axes[0, 0]
ax.scatter(predictions_df[actual_col], predictions_df[pred_col], alpha=0.3, s=10)
ax.plot([0, 80], [0, 80], 'r--', label='Perfect Prediction', linewidth=2)
ax.set_xlabel('Actual PRA', fontsize=12)
ax.set_ylabel('Predicted PRA', fontsize=12)
ax.set_title('Predicted vs Actual PRA', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Add MAE and RMSE to plot
mae = predictions_df['abs_error'].mean()
rmse = np.sqrt((predictions_df['error']**2).mean())
ax.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Distribution of predictions vs actuals
ax = axes[0, 1]
ax.hist(predictions_df[actual_col], bins=50, alpha=0.5, label='Actual', density=True)
ax.hist(predictions_df[pred_col], bins=50, alpha=0.5, label='Predicted', density=True)
ax.set_xlabel('PRA', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution: Predicted vs Actual', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Error distribution
ax = axes[1, 0]
ax.hist(predictions_df['error'], bins=100, alpha=0.7, edgecolor='black')
ax.axvline(predictions_df['error'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {predictions_df["error"].mean():.2f}')
ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Absolute error by prediction magnitude
ax = axes[1, 1]
predictions_df['pred_range'] = pd.cut(predictions_df[pred_col],
                                      bins=[0, 10, 20, 30, 40, 50, 100],
                                      labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50+'])
error_by_range = predictions_df.groupby('pred_range')['abs_error'].agg(['mean', 'std'])
error_by_range['mean'].plot(kind='bar', ax=ax, yerr=error_by_range['std'],
                             capsize=5, alpha=0.7, color='steelblue')
ax.set_xlabel('Prediction Range', fontsize=12)
ax.set_ylabel('Mean Absolute Error', fontsize=12)
ax.set_title('MAE by Prediction Range', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_prediction_quality.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '1_prediction_quality.png'}")
plt.close()

# ============================================================================
# 2. ERROR PATTERNS BY GAMES IN HISTORY
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Error by games in history
ax = axes[0]
history_bins = pd.cut(predictions_df['games_in_history'],
                      bins=[0, 10, 20, 30, 40, 50, 100],
                      labels=['5-10', '10-20', '20-30', '30-40', '40-50', '50+'])
error_by_history = predictions_df.groupby(history_bins).agg({
    'abs_error': 'mean',
    pred_col: 'count'
})
ax.bar(range(len(error_by_history)), error_by_history['abs_error'],
       alpha=0.7, color='coral')
ax.set_xticks(range(len(error_by_history)))
ax.set_xticklabels(error_by_history.index, rotation=45)
ax.set_xlabel('Games in History', fontsize=12)
ax.set_ylabel('Mean Absolute Error', fontsize=12)
ax.set_title('MAE by Training Data Size', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Add sample counts
for i, count in enumerate(error_by_history[pred_col]):
    ax.text(i, error_by_history['abs_error'].iloc[i] + 0.2,
            f'n={count}', ha='center', fontsize=9)

# Scatter: games in history vs error
ax = axes[1]
sample = predictions_df.sample(min(5000, len(predictions_df)))
ax.scatter(sample['games_in_history'], sample['abs_error'], alpha=0.3, s=10)
z = np.polyfit(predictions_df['games_in_history'], predictions_df['abs_error'], 1)
p = np.poly1d(z)
ax.plot(predictions_df['games_in_history'].sort_values(),
        p(predictions_df['games_in_history'].sort_values()),
        "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.2f}')
ax.set_xlabel('Games in History', fontsize=12)
ax.set_ylabel('Absolute Error', fontsize=12)
ax.set_title('Error vs Training Data Size (Sampled)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_error_by_history.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '2_error_by_history.png'}")
plt.close()

# ============================================================================
# 3. PLAYER-LEVEL ERROR ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Top players by MAE
player_errors = predictions_df.groupby('PLAYER_NAME').agg({
    'abs_error': ['mean', 'count'],
    'error': 'mean',
    pred_col: 'mean',
    actual_col: 'mean'
}).round(3)
player_errors.columns = ['MAE', 'Count', 'Bias', 'Avg_Pred', 'Avg_Actual']
player_errors = player_errors[player_errors['Count'] >= 10]

# Worst performers
ax = axes[0, 0]
worst = player_errors.nlargest(15, 'MAE')
ax.barh(range(len(worst)), worst['MAE'], color='crimson', alpha=0.7)
ax.set_yticks(range(len(worst)))
ax.set_yticklabels(worst.index, fontsize=9)
ax.set_xlabel('Mean Absolute Error', fontsize=12)
ax.set_title('Players with Highest MAE (min 10 games)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Best performers
ax = axes[0, 1]
best = player_errors.nsmallest(15, 'MAE')
ax.barh(range(len(best)), best['MAE'], color='seagreen', alpha=0.7)
ax.set_yticks(range(len(best)))
ax.set_yticklabels(best.index, fontsize=9)
ax.set_xlabel('Mean Absolute Error', fontsize=12)
ax.set_title('Players with Lowest MAE (min 10 games)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Bias analysis - over-predicted
ax = axes[1, 0]
overpredict = player_errors.nlargest(15, 'Bias')
ax.barh(range(len(overpredict)), overpredict['Bias'], color='orange', alpha=0.7)
ax.set_yticks(range(len(overpredict)))
ax.set_yticklabels(overpredict.index, fontsize=9)
ax.set_xlabel('Bias (Predicted - Actual)', fontsize=12)
ax.set_title('Most Over-Predicted Players', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Bias analysis - under-predicted
ax = axes[1, 1]
underpredict = player_errors.nsmallest(15, 'Bias')
ax.barh(range(len(underpredict)), underpredict['Bias'], color='skyblue', alpha=0.7)
ax.set_yticks(range(len(underpredict)))
ax.set_yticklabels(underpredict.index, fontsize=9)
ax.set_xlabel('Bias (Predicted - Actual)', fontsize=12)
ax.set_title('Most Under-Predicted Players', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '3_player_level_errors.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '3_player_level_errors.png'}")
plt.close()

# ============================================================================
# 4. EDGE CALIBRATION ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Edge buckets
backtest_df['edge_bucket'] = pd.cut(backtest_df['edge'],
                                    bins=[-100, -10, -5, 0, 5, 10, 100],
                                    labels=['< -10%', '-10 to -5%', '-5 to 0%',
                                           '0 to 5%', '5 to 10%', '> 10%'])

# Win rate by edge bucket
ax = axes[0, 0]
edge_stats = backtest_df.groupby('edge_bucket').agg({
    'bet_won': lambda x: x.mean() * 100,
    'edge': 'mean',
    'bet_result': 'count'
})
edge_stats.columns = ['Win_Rate', 'Avg_Edge', 'Count']
edge_stats['Expected_Win_Rate'] = edge_stats['Avg_Edge'] + 50

x_pos = range(len(edge_stats))
width = 0.35
ax.bar([p - width/2 for p in x_pos], edge_stats['Win_Rate'],
       width, label='Actual Win Rate', alpha=0.7, color='steelblue')
ax.bar([p + width/2 for p in x_pos], edge_stats['Expected_Win_Rate'],
       width, label='Expected Win Rate', alpha=0.7, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(edge_stats.index, rotation=45)
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title('Win Rate vs Expected by Edge Bucket', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Calibration error
ax = axes[0, 1]
edge_stats['Calibration_Error'] = edge_stats['Win_Rate'] - edge_stats['Expected_Win_Rate']
colors = ['red' if x < 0 else 'green' for x in edge_stats['Calibration_Error']]
ax.bar(range(len(edge_stats)), edge_stats['Calibration_Error'], color=colors, alpha=0.7)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(range(len(edge_stats)))
ax.set_xticklabels(edge_stats.index, rotation=45)
ax.set_ylabel('Calibration Error (%)', fontsize=12)
ax.set_title('Calibration Error by Edge Bucket', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Edge scatter
ax = axes[1, 0]
ax.scatter(backtest_df['edge'], backtest_df['bet_won'].astype(int), alpha=0.3, s=10)
# Add smoothed line
from scipy.ndimage import gaussian_filter1d
sorted_indices = backtest_df['edge'].argsort()
sorted_edge = backtest_df['edge'].iloc[sorted_indices]
sorted_wins = backtest_df['bet_won'].iloc[sorted_indices].astype(float)
# Bin and average
bins = pd.cut(sorted_edge, bins=20)
binned_win_rate = backtest_df.groupby(pd.cut(backtest_df['edge'], bins=20))['bet_won'].mean()
bin_centers = [interval.mid for interval in binned_win_rate.index]
ax.plot(bin_centers, binned_win_rate.values, 'r-', linewidth=3, label='Observed Win Rate')
ax.plot(sorted_edge, (sorted_edge + 50) / 100, 'g--', linewidth=2, label='Expected Win Rate')
ax.set_xlabel('Edge (%)', fontsize=12)
ax.set_ylabel('Win Probability', fontsize=12)
ax.set_title('Edge Calibration Curve', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Bet counts by edge bucket
ax = axes[1, 1]
edge_counts = backtest_df['edge_bucket'].value_counts().sort_index()
ax.bar(range(len(edge_counts)), edge_counts.values, color='mediumpurple', alpha=0.7)
ax.set_xticks(range(len(edge_counts)))
ax.set_xticklabels(edge_counts.index, rotation=45)
ax.set_ylabel('Number of Bets', fontsize=12)
ax.set_title('Bet Distribution by Edge Bucket', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Add counts on bars
for i, v in enumerate(edge_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '4_edge_calibration.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '4_edge_calibration.png'}")
plt.close()

# ============================================================================
# 5. FEATURE QUALITY HEATMAP
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Missing values heatmap
ax = axes[0]
missing_rates = (features_df.isna().sum() / len(features_df) * 100).sort_values(ascending=False)
top_missing = missing_rates[missing_rates > 5].head(30)

if len(top_missing) > 0:
    ax.barh(range(len(top_missing)), top_missing.values, color='crimson', alpha=0.7)
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing.index, fontsize=9)
    ax.set_xlabel('Missing Rate (%)', fontsize=12)
    ax.set_title('Features with Highest Missing Rates', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

# Feature correlation with target (if exists)
ax = axes[1]
if 'pra' in features_df.columns:
    target_col = 'pra'
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f != target_col]

    # Sample features for correlation
    sample_features = numeric_features[:50]
    correlations = []
    for feat in sample_features:
        try:
            corr = features_df[[feat, target_col]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append((feat, corr))
        except:
            pass

    if correlations:
        correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:30]
        feat_names, corr_values = zip(*correlations)

        colors = ['green' if x > 0 else 'red' for x in corr_values]
        ax.barh(range(len(corr_values)), corr_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(corr_values)))
        ax.set_yticklabels(feat_names, fontsize=9)
        ax.set_xlabel('Correlation with Target', fontsize=12)
        ax.set_title('Top 30 Features by Correlation with PRA', fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '5_feature_quality.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / '5_feature_quality.png'}")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print(f"Saved to: {output_dir}")
print("="*80)
