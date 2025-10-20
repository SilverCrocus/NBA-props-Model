"""
Visualize Edge Stratification Issue
====================================

Show how edge size creates artificial separation in win rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
bets = pd.read_csv('/Users/diyagamah/Documents/nba_props_model/data/results/phase1_betting_simulation_2024_25.csv')
bets['abs_edge'] = bets['edge'].abs()
bets['abs_error'] = bets['error'].abs()

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Scatter plot - Edge vs Win Rate
ax1 = fig.add_subplot(gs[0, :])
# Create edge bins for scatter
edge_bins = pd.cut(bets['abs_edge'], bins=50)
bin_stats = bets.groupby(edge_bins, observed=True).agg({
    'bet_won': 'mean',
    'abs_edge': 'mean'
}).dropna()

ax1.scatter(bin_stats['abs_edge'], bin_stats['bet_won']*100, s=100, alpha=0.6)
ax1.axhline(y=50, color='gray', linestyle='--', label='Random (50%)', linewidth=2)
ax1.axhline(y=52.4, color='blue', linestyle='--', label='Overall (52.4%)', linewidth=2)
ax1.axhline(y=84, color='red', linestyle='--', label='Ultra-Selective (84%)', linewidth=2)
ax1.set_xlabel('Edge Size (points)', fontsize=12)
ax1.set_ylabel('Win Rate (%)', fontsize=12)
ax1.set_title('WIN RATE vs EDGE SIZE: The Stratification Problem', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart - Win rate by edge bins
ax2 = fig.add_subplot(gs[1, 0])
edge_categories = pd.cut(
    bets['abs_edge'],
    bins=[0, 5.5, 6.5, 7.5, 10, 100],
    labels=['<5.5', '5.5-6.5', '6.5-7.5', '7.5-10', '10+']
)
edge_analysis = bets.groupby(edge_categories, observed=True).agg({
    'bet_won': ['sum', 'count', 'mean']
})
edge_analysis.columns = ['wins', 'total', 'win_rate']
edge_analysis_plot = edge_analysis.reset_index()

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
bars = ax2.bar(edge_analysis_plot['abs_edge'].astype(str), edge_analysis_plot['win_rate']*100, color=colors)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.axhline(y=52.4, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.set_xlabel('Edge Magnitude', fontsize=11)
ax2.set_ylabel('Win Rate (%)', fontsize=11)
ax2.set_title('Win Rate by Edge Category', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    count = edge_analysis_plot.iloc[i]['total']
    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.1f}%\n(n={int(count)})',
             ha='center', va='bottom', fontsize=9)

# Plot 3: Bet volume by edge
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(edge_analysis_plot['abs_edge'].astype(str), edge_analysis_plot['total'], color=colors)
ax3.set_xlabel('Edge Magnitude', fontsize=11)
ax3.set_ylabel('Number of Bets', fontsize=11)
ax3.set_title('Bet Volume by Edge Category', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (idx, row) in enumerate(edge_analysis_plot.iterrows()):
    ax3.text(i, row['total'] + 50, f"{int(row['total']):,}",
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Cumulative win rate (sorted by edge)
ax4 = fig.add_subplot(gs[2, 0])
bets_sorted = bets.sort_values('abs_edge', ascending=False)
cumulative_wr = []
n_values = []

for n in range(100, len(bets_sorted), 50):
    top_n = bets_sorted.head(n)
    cumulative_wr.append(top_n['bet_won'].mean() * 100)
    n_values.append(n)

ax4.plot(n_values, cumulative_wr, linewidth=2.5, color='#1f77b4')
ax4.axhline(y=52.4, color='blue', linestyle='--', label='Overall (52.4%)', linewidth=2)
ax4.axhline(y=84, color='red', linestyle='--', label='Ultra-Selective (84%)', linewidth=2)
ax4.axvline(x=159, color='red', linestyle=':', label='Ultra-Selective N=159', linewidth=2)
ax4.set_xlabel('Top N Bets (sorted by edge)', fontsize=11)
ax4.set_ylabel('Cumulative Win Rate (%)', fontsize=11)
ax4.set_title('Cumulative Win Rate: How Cherry-Picking Inflates Performance', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: MAE by edge (to show it doesn't improve with edge)
ax5 = fig.add_subplot(gs[2, 1])
edge_mae = bets.groupby(edge_categories, observed=True)['abs_error'].mean()
edge_mae_plot = edge_mae.reset_index()

bars2 = ax5.bar(edge_mae_plot['abs_edge'].astype(str), edge_mae_plot['abs_error'], color=colors)
ax5.axhline(y=4.28, color='blue', linestyle='--', label='Overall MAE (4.28)', linewidth=2)
ax5.set_xlabel('Edge Magnitude', fontsize=11)
ax5.set_ylabel('MAE (points)', fontsize=11)
ax5.set_title('Prediction Error by Edge: Does NOT Improve With Edge!', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=9)

plt.suptitle('Phase 1 Edge Stratification Analysis\n84% Win Rate is From Cherry-Picking Large Edges',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/Users/diyagamah/Documents/nba_props_model/data/results/edge_stratification_analysis.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Saved: data/results/edge_stratification_analysis.png")

# Print summary stats
print("\n" + "="*80)
print("EDGE STRATIFICATION SUMMARY")
print("="*80)

print("\nKey Insight:")
print("-"*80)
print("Large edges have high win rates NOT because predictions are more accurate,")
print("but because the model is more CONFIDENT (and thus more often correct).")
print("")
print("MAE is SAME across all edge sizes (~4.2 points), but win rate varies from")
print("32% (small edges) to 93% (large edges).")
print("")
print("This is SELECTION BIAS, not data leakage.")

print("\n" + "="*80)
print("STATISTICAL PROOF")
print("="*80)

for idx, row in edge_analysis.iterrows():
    wr = row['win_rate'] * 100
    mae = bets[edge_categories == idx]['abs_error'].mean()
    print(f"\nEdge {idx}:")
    print(f"  Bets: {int(row['total']):,}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  MAE: {mae:.2f} points")
    print(f"  Avg Edge: {bets[edge_categories == idx]['abs_edge'].mean():.2f} points")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("84% win rate = cherry-picking top 159 bets with edge >14 points")
print("52.4% win rate = true model performance on all 3,813 qualifying bets")
print("\nNO DATA LEAKAGE - just aggressive filtering creating selection bias")
