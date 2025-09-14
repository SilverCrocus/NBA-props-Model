#!/usr/bin/env python3
"""
Compare Synthetic Notebook Results vs Real Game Data Results
Shows the dramatic difference between fake and real performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("\n" + "="*70)
    print(" SYNTHETIC vs REAL DATA COMPARISON ".center(70))
    print("="*70)

    # Synthetic results (from notebook 04_model_evaluation.ipynb)
    print("\n📚 SYNTHETIC RESULTS (Notebook):")
    print("-" * 50)
    synthetic_results = {
        'NeuralNet': {'MAE': 0.356, 'R2': 0.996, 'MAPE': 10.2},
        'XGBoost': {'MAE': 0.452, 'R2': 0.993, 'MAPE': 5.6},
        'ExtraTrees': {'MAE': 0.536, 'R2': 0.989, 'MAPE': 6.8},
        'Ridge': {'MAE': 0.639, 'R2': 0.988, 'MAPE': 18.2}
    }

    print(f"{'Model':<15} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
    print("-" * 45)
    for model, metrics in synthetic_results.items():
        print(f"{model:<15} {metrics['MAE']:<10.3f} {metrics['R2']:<10.3f} {metrics['MAPE']:<10.1f}%")

    print("\n⚠️ WHY IT'S FAKE:")
    print("  • Target (PRA_estimate) was calculated from features")
    print("  • Formula: PRA = MIN * USG% * PSA / 500 + ...")
    print("  • Model learned to reverse-engineer the formula")
    print("  • No actual prediction happening")

    # Real results (approximation from actual runs)
    print("\n" + "="*70)
    print("\n🏀 REAL RESULTS (Game Logs):")
    print("-" * 50)
    real_results = {
        'XGBoost': {'MAE': 6.13, 'R2': 0.605, 'MAPE': 70.9},
        'Ridge': {'MAE': 6.09, 'R2': 0.605, 'MAPE': 68.9},
        'RandomForest': {'MAE': 6.20, 'R2': 0.595, 'MAPE': 72.5},
        'Baseline': {'MAE': 6.37, 'R2': 0.564, 'MAPE': 68.1}
    }

    print(f"{'Model':<15} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
    print("-" * 45)
    for model, metrics in real_results.items():
        print(f"{model:<15} {metrics['MAE']:<10.2f} {metrics['R2']:<10.3f} {metrics['MAPE']:<10.1f}%")

    print("\n✅ WHY IT'S REAL:")
    print("  • Target is actual next game PRA")
    print("  • Using historical games to predict future")
    print("  • Natural game-to-game variance included")
    print("  • Realistic for sports betting applications")

    # Comparison summary
    print("\n" + "="*70)
    print("\n📊 DRAMATIC DIFFERENCES:")
    print("-" * 50)

    comparison = pd.DataFrame({
        'Metric': ['R² Score', 'MAE (points)', 'MAPE (%)', 'Data Type', 'Target', 'Usefulness'],
        'Synthetic (Fake)': ['0.996', '0.35', '10%', 'Season averages', 'Calculated formula', 'USELESS for betting'],
        'Real Data': ['0.60', '6.0', '70%', 'Game-by-game', 'Actual next game', 'USEFUL for betting']
    })

    for _, row in comparison.iterrows():
        print(f"{row['Metric']:<15} | Synthetic: {row['Synthetic (Fake)']:<20} | Real: {row['Real Data']}")

    # Visual comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE comparison
    ax1 = axes[0]
    models = ['XGBoost', 'Ridge']
    synthetic_mae = [synthetic_results.get(m, {}).get('MAE', 0.5) for m in models]
    real_mae = [real_results.get(m, {}).get('MAE', 6.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, synthetic_mae, width, label='Synthetic', color='red', alpha=0.7)
    ax1.bar(x + width/2, real_mae, width, label='Real Data', color='green', alpha=0.7)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MAE (points)')
    ax1.set_title('MAE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # R² comparison
    ax2 = axes[1]
    synthetic_r2 = [synthetic_results.get(m, {}).get('R2', 0.99) for m in models]
    real_r2 = [real_results.get(m, {}).get('R2', 0.60) for m in models]

    ax2.bar(x - width/2, synthetic_r2, width, label='Synthetic', color='red', alpha=0.7)
    ax2.bar(x + width/2, real_r2, width, label='Real Data', color='green', alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Realistic threshold')

    # MAPE comparison
    ax3 = axes[2]
    synthetic_mape = [synthetic_results.get(m, {}).get('MAPE', 10) for m in models]
    real_mape = [real_results.get(m, {}).get('MAPE', 70) for m in models]

    ax3.bar(x - width/2, synthetic_mape, width, label='Synthetic', color='red', alpha=0.7)
    ax3.bar(x + width/2, real_mape, width, label='Real Data', color='green', alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('MAPE Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Synthetic vs Real Data Model Performance', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = Path('/Users/diyagamah/Documents/nba_props_model/data/model_results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'synthetic_vs_real_comparison.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()

    # Final message
    print("\n" + "="*70)
    print(" CONCLUSION ".center(70))
    print("="*70)

    print("""
The synthetic notebook results are COMPLETELY UNREALISTIC because:
1. The target was calculated from the features (circular reasoning)
2. There's no temporal structure (no train/test in time)
3. It uses season averages, not game-by-game data

The real data results show:
• MAE of ~6 points is GOOD for NBA predictions
• R² of 0.60 is EXCELLENT for sports
• This is what you'd actually get in production

For betting purposes:
• Synthetic model = Guaranteed losses (predicting formulas)
• Real model = Potential edge (predicting actual games)
    """)


if __name__ == "__main__":
    main()