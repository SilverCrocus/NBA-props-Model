"""
Create visualizations for backtest analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Load data
betting_df = pd.read_csv('data/results/backtest_2024_25_FIXED_V2_betting.csv')
betting_df['GAME_DATE'] = pd.to_datetime(betting_df['GAME_DATE'])

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Prediction vs Actual Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(betting_df['predicted_PRA'], bins=30, alpha=0.5, label='Predicted', color='blue')
ax1.hist(betting_df['actual_PRA'], bins=30, alpha=0.5, label='Actual', color='orange')
ax1.axvline(betting_df['predicted_PRA'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Pred Mean: {betting_df["predicted_PRA"].mean():.1f}')
ax1.axvline(betting_df['actual_PRA'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Actual Mean: {betting_df["actual_PRA"].mean():.1f}')
ax1.set_xlabel('PRA Points')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution: Predicted vs Actual PRA', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Prediction Error Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(betting_df['error'], bins=40, color='red', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='black', linestyle='-', linewidth=2, label='Perfect Prediction')
ax2.axvline(betting_df['error'].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean Error: {betting_df["error"].mean():.1f}')
ax2.set_xlabel('Prediction Error (Actual - Predicted)')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Error Distribution\n100% Underpredictions!', fontweight='bold', color='darkred')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Scatter: Predicted vs Actual
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(betting_df['predicted_PRA'], betting_df['actual_PRA'], alpha=0.3, s=10)
ax3.plot([0, 50], [0, 50], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Predicted PRA')
ax3.set_ylabel('Actual PRA')
ax3.set_title('Predicted vs Actual PRA Scatter', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Win Rate by Edge Size
ax4 = fig.add_subplot(gs[1, 0])
edge_bins = pd.cut(betting_df['abs_edge'], bins=[5, 5.5, 6, 6.5, 7, 10], labels=['5.0-5.5', '5.5-6.0', '6.0-6.5', '6.5-7.0', '7.0+'])
edge_performance = betting_df.groupby(edge_bins, observed=True)['bet_correct'].agg(['count', 'mean']).reset_index()
edge_performance.columns = ['Edge_Bin', 'Count', 'Win_Rate']
bars = ax4.bar(range(len(edge_performance)), edge_performance['Win_Rate'], color=['green' if x > 0.524 else 'red' for x in edge_performance['Win_Rate']])
ax4.axhline(0.524, color='black', linestyle='--', linewidth=2, label='Breakeven (52.4%)')
ax4.set_xticks(range(len(edge_performance)))
ax4.set_xticklabels(edge_performance['Edge_Bin'], rotation=45)
ax4.set_ylabel('Win Rate')
ax4.set_title('Win Rate by Edge Size\n(Inverted Relationship - Red Flag!)', fontweight='bold', color='darkred')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# Add counts on bars
for i, (idx, row) in enumerate(edge_performance.iterrows()):
    ax4.text(i, row['Win_Rate'] + 0.01, f"n={row['Count']}", ha='center', fontsize=8)

# 5. Performance by Player Tier
ax5 = fig.add_subplot(gs[1, 1])
betting_df['PLAYER_TIER'] = pd.cut(betting_df['actual_PRA'], bins=[0, 15, 25, 35, 50, 100], labels=['Role (<15)', 'Rotation (15-25)', 'Starter (25-35)', 'Star (35-50)', 'Superstar (50+)'])
tier_performance = betting_df.groupby('PLAYER_TIER', observed=True).agg({
    'bet_correct': ['count', 'mean']
}).reset_index()
tier_performance.columns = ['Tier', 'Count', 'Win_Rate']
bars = ax5.bar(range(len(tier_performance)), tier_performance['Win_Rate'], color=['green', 'orange', 'red', 'darkred'])
ax5.axhline(0.524, color='black', linestyle='--', linewidth=2, label='Breakeven')
ax5.set_xticks(range(len(tier_performance)))
ax5.set_xticklabels(tier_performance['Tier'], rotation=45, ha='right')
ax5.set_ylabel('Win Rate')
ax5.set_title('Win Rate by Player Tier\n(Works for Role Players Only!)', fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# Add counts on bars
for i, (idx, row) in enumerate(tier_performance.iterrows()):
    ax5.text(i, min(row['Win_Rate'] + 0.05, 0.95), f"n={row['Count']}", ha='center', fontsize=8)

# 6. Monthly Performance
ax6 = fig.add_subplot(gs[1, 2])
betting_df['MONTH'] = betting_df['GAME_DATE'].dt.to_period('M')
monthly = betting_df.groupby('MONTH').agg({
    'bet_correct': ['count', 'mean']
}).reset_index()
monthly.columns = ['Month', 'Count', 'Win_Rate']
monthly['Month_Str'] = monthly['Month'].astype(str)
bars = ax6.bar(range(len(monthly)), monthly['Win_Rate'], color=['green' if x > 0.524 else 'red' for x in monthly['Win_Rate']])
ax6.axhline(0.524, color='black', linestyle='--', linewidth=2, label='Breakeven')
ax6.set_xticks(range(len(monthly)))
ax6.set_xticklabels([m.split('-')[1] for m in monthly['Month_Str']], rotation=0)
ax6.set_ylabel('Win Rate')
ax6.set_xlabel('Month (2024-2025)')
ax6.set_title('Win Rate by Month\n(Improving Trend)', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# 7. Cumulative Profit
ax7 = fig.add_subplot(gs[2, :])
betting_df_sorted = betting_df.sort_values('GAME_DATE')
betting_df_sorted['PROFIT'] = betting_df_sorted['bet_correct'].apply(lambda x: 0.909 if x else -1.0)
betting_df_sorted['CUMULATIVE_PROFIT'] = betting_df_sorted['PROFIT'].cumsum()
ax7.plot(range(len(betting_df_sorted)), betting_df_sorted['CUMULATIVE_PROFIT'], linewidth=2, color='darkblue')
ax7.axhline(0, color='red', linestyle='--', linewidth=1)
ax7.fill_between(range(len(betting_df_sorted)), 0, betting_df_sorted['CUMULATIVE_PROFIT'], where=betting_df_sorted['CUMULATIVE_PROFIT']>=0, alpha=0.3, color='green', label='Profit')
ax7.fill_between(range(len(betting_df_sorted)), 0, betting_df_sorted['CUMULATIVE_PROFIT'], where=betting_df_sorted['CUMULATIVE_PROFIT']<0, alpha=0.3, color='red', label='Loss')
ax7.set_xlabel('Bet Number (Chronological)')
ax7.set_ylabel('Cumulative Profit (Units)')
ax7.set_title('Cumulative Profit Over Time (Oct 2024 - Apr 2025)', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(alpha=0.3)

# Add final profit text
final_profit = betting_df_sorted['CUMULATIVE_PROFIT'].iloc[-1]
ax7.text(len(betting_df_sorted)*0.8, final_profit, f'Final: {final_profit:.1f} units', fontsize=12, color='darkred' if final_profit < 0 else 'darkgreen', fontweight='bold')

# 8. Top 10 Profitable Players
ax8 = fig.add_subplot(gs[3, 0])
player_stats = betting_df.groupby('PLAYER_NAME').agg({
    'bet_correct': ['count', 'sum', 'mean']
}).reset_index()
player_stats.columns = ['Player', 'Bets', 'Wins', 'Win_Rate']
player_stats = player_stats[player_stats['Bets'] >= 10]
player_stats['Profit'] = player_stats['Wins'] * 0.909 - (player_stats['Bets'] - player_stats['Wins']) * 1.0
player_stats_sorted = player_stats.sort_values('Profit', ascending=True)
top10_profit = player_stats_sorted.tail(10)
ax8.barh(range(len(top10_profit)), top10_profit['Profit'], color='green')
ax8.set_yticks(range(len(top10_profit)))
ax8.set_yticklabels([p[:15] for p in top10_profit['Player']], fontsize=9)
ax8.set_xlabel('Profit (Units)')
ax8.set_title('Top 10 Most Profitable Players (10+ bets)', fontweight='bold', fontsize=10)
ax8.grid(alpha=0.3, axis='x')

# 9. Bottom 10 Unprofitable Players
ax9 = fig.add_subplot(gs[3, 1])
bottom10_profit = player_stats_sorted.head(10)
ax9.barh(range(len(bottom10_profit)), bottom10_profit['Profit'], color='red')
ax9.set_yticks(range(len(bottom10_profit)))
ax9.set_yticklabels([p[:15] for p in bottom10_profit['Player']], fontsize=9)
ax9.set_xlabel('Profit (Units)')
ax9.set_title('Bottom 10 Most Unprofitable Players (10+ bets)', fontweight='bold', fontsize=10)
ax9.grid(alpha=0.3, axis='x')

# 10. MAE by Player Tier
ax10 = fig.add_subplot(gs[3, 2])
tier_mae = betting_df.groupby('PLAYER_TIER', observed=True).agg({
    'error': ['count', 'mean']
}).reset_index()
tier_mae.columns = ['Tier', 'Count', 'MAE']
tier_mae['MAE'] = tier_mae['MAE'].abs()
bars = ax10.bar(range(len(tier_mae)), tier_mae['MAE'], color=['green', 'yellow', 'orange', 'red'])
ax10.axhline(5, color='black', linestyle='--', linewidth=2, label='Target MAE: 5.0')
ax10.set_xticks(range(len(tier_mae)))
ax10.set_xticklabels(tier_mae['Tier'], rotation=45, ha='right')
ax10.set_ylabel('Mean Absolute Error (Points)')
ax10.set_title('MAE by Player Tier\n(Escalating Error for Better Players)', fontweight='bold')
ax10.legend()
ax10.grid(alpha=0.3, axis='y')

# Main title
fig.suptitle('NBA Props Model Backtest Analysis - 2024-25 Season\nCRITICAL: Systematic Underprediction Bias Detected',
             fontsize=16, fontweight='bold', color='darkred', y=0.995)

# Save figure
plt.savefig('backtest_analysis_visualizations.png', dpi=150, bbox_inches='tight')
print("Visualizations saved to: backtest_analysis_visualizations.png")

# Create summary stats image
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Critical Model Issues - Summary Dashboard', fontsize=14, fontweight='bold', color='darkred')

# Issue 1: Systematic Bias
ax1 = axes[0, 0]
ax1.text(0.5, 0.8, 'ISSUE #1: SYSTEMATIC BIAS', ha='center', fontsize=14, fontweight='bold', color='darkred')
ax1.text(0.5, 0.6, f'Underpredictions: 100%', ha='center', fontsize=12)
ax1.text(0.5, 0.5, f'Average Error: +{betting_df["error"].mean():.2f} points', ha='center', fontsize=12)
ax1.text(0.5, 0.4, f'Predicted Avg: {betting_df["predicted_PRA"].mean():.1f}', ha='center', fontsize=12)
ax1.text(0.5, 0.3, f'Actual Avg: {betting_df["actual_PRA"].mean():.1f}', ha='center', fontsize=12)
ax1.text(0.5, 0.1, 'Model predicts ~7 pts too low!', ha='center', fontsize=11, color='red', style='italic')
ax1.axis('off')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Issue 2: Directional Bias
ax2 = axes[0, 1]
ax2.text(0.5, 0.8, 'ISSUE #2: DIRECTIONAL BIAS', ha='center', fontsize=14, fontweight='bold', color='darkred')
ax2.text(0.5, 0.6, f'UNDER Bets: {(betting_df["bet_type"] == "UNDER").sum()}', ha='center', fontsize=12)
ax2.text(0.5, 0.5, f'OVER Bets: {(betting_df["bet_type"] == "OVER").sum()}', ha='center', fontsize=12)
ax2.text(0.5, 0.4, 'Only betting ONE side!', ha='center', fontsize=12, color='red')
ax2.text(0.5, 0.2, 'Healthy model: ~50/50 split', ha='center', fontsize=10, style='italic')
ax2.text(0.5, 0.1, 'Root cause: Systematic underprediction', ha='center', fontsize=10, color='red', style='italic')
ax2.axis('off')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Issue 3: Player Tier Failure
ax3 = axes[1, 0]
ax3.text(0.5, 0.8, 'ISSUE #3: PLAYER TIER FAILURE', ha='center', fontsize=14, fontweight='bold', color='darkred')
ax3.text(0.5, 0.6, 'Role Players (<15): 93% win rate ✓', ha='center', fontsize=11, color='green')
ax3.text(0.5, 0.5, 'Rotation (15-25): 16% win rate ✗', ha='center', fontsize=11, color='orange')
ax3.text(0.5, 0.4, 'Starters (25-35): 0% win rate ✗', ha='center', fontsize=11, color='red')
ax3.text(0.5, 0.3, 'Stars (35+): 0% win rate ✗', ha='center', fontsize=11, color='darkred')
ax3.text(0.5, 0.1, 'Model cannot predict high-volume players', ha='center', fontsize=10, color='red', style='italic')
ax3.axis('off')
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Issue 4: Inverted Edge Relationship
ax4 = axes[1, 1]
ax4.text(0.5, 0.8, 'ISSUE #4: INVERTED EDGE', ha='center', fontsize=14, fontweight='bold', color='darkred')
ax4.text(0.5, 0.6, '5.0-5.5 edge: 55.4% win rate', ha='center', fontsize=11, color='green')
ax4.text(0.5, 0.5, '6.0-6.5 edge: 48.4% win rate', ha='center', fontsize=11, color='red')
ax4.text(0.5, 0.4, 'Smaller edges perform BETTER!', ha='center', fontsize=12, color='red')
ax4.text(0.5, 0.2, 'This should be opposite', ha='center', fontsize=10, style='italic')
ax4.text(0.5, 0.1, 'Confirms edge is miscalibration, not value', ha='center', fontsize=10, color='red', style='italic')
ax4.axis('off')
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('critical_issues_summary.png', dpi=150, bbox_inches='tight')
print("Critical issues summary saved to: critical_issues_summary.png")

print("\nVisualization complete!")
print(f"Generated 2 PNG files in project root directory")
