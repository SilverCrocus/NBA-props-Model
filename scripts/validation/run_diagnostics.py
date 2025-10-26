"""
Run betting diagnostics with automatic column mapping
Handles your specific CLV ledger format

Author: Elite Data Science Team
Date: 2025 - 10 - 25
"""

import sys
from pathlib import Path

import pandas as pd

from scripts.validation.betting_diagnostics import BettingDiagnostics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load CLV ledger and map columns to diagnostic format

    Your columns:
    - entry_odds_dec → entry_odds
    - won → win
    - result → actual_pra
    - ev_entry → entry_ev

    Required columns for diagnostics:
    - entry_odds, predicted_pra, actual_pra, line, side
    - stake, win, player_sigma, entry_ev
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} bets")
    print(f"Original columns: {df.columns.tolist()}")

    # Column mapping
    column_mapping = {
        "entry_odds_dec": "entry_odds",
        "won": "win",
        "result": "actual_pra",
        "ev_entry": "entry_ev",
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Verify required columns exist
    required = [
        "entry_odds",
        "predicted_pra",
        "actual_pra",
        "line",
        "side",
        "stake",
        "win",
        "player_sigma",
        "entry_ev",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns after mapping: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Filter to only completed bets (have actual_pra)
    df_complete = df[df["actual_pra"].notna()].copy()
    print(f"\nFiltered to {len(df_complete)} completed bets (have results)")

    # Data validation
    print("\nData validation:")
    print(
        f"  - Entry odds range: {
            df_complete['entry_odds'].min():.2f} to {
            df_complete['entry_odds'].max():.2f}"
    )
    print(
        f"  - Predicted PRA range: {
            df_complete['predicted_pra'].min():.1f} to {
            df_complete['predicted_pra'].max():.1f}"
    )
    print(
        f"  - Actual PRA range: {
            df_complete['actual_pra'].min():.1f} to {
            df_complete['actual_pra'].max():.1f}"
    )
    print(
        f"  - Line range: {
            df_complete['line'].min():.1f} to {
            df_complete['line'].max():.1f}"
    )
    print(
        f"  - Stake range: {
            df_complete['stake'].min():.2f} to {
            df_complete['stake'].max():.2f}"
    )
    print(f"  - Win rate: {df_complete['win'].mean():.2%}")
    print(
        f"  - OVER bets: {len(df_complete[df_complete['side'] == 'OVER'])} ({len(df_complete[df_complete['side'] == 'OVER']) / len(df_complete) * 100:.1f}%)"  # noqa: E501
    )
    print(
        f"  - UNDER bets: {len(df_complete[df_complete['side'] == 'UNDER'])} ({len(df_complete[df_complete['side'] == 'UNDER']) / len(df_complete) * 100:.1f}%)"  # noqa: E501
    )

    # Calculate basic metrics
    profit = (df_complete["win"] * (df_complete["entry_odds"] - 1)) - (
        1 - df_complete["win"]
    )  # noqa: E501
    total_staked = df_complete["stake"].sum()
    roi = (profit.sum() / total_staked * 100) if total_staked > 0 else 0

    print("\nPerformance Summary:")
    print(f"  - Total profit/loss: {profit.sum():.2f} units")
    print(f"  - Total staked: {total_staked:.2f} units")
    print(f"  - ROI: {roi:.2f}%")
    print(
        f"  - Breakeven WR: {df_complete['entry_odds'].apply(lambda x: 1 / x).mean():.2%}"
    )  # noqa: E501

    return df_complete


def main():
    """Run comprehensive diagnostics"""
    # Path to CLV ledger
    data_path = Path(__file__).parent.parent.parent / "data" / "clv_ledger.csv"

    if not data_path.exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("\nPlease ensure clv_ledger.csv exists in data/ directory")
        sys.exit(1)

    # Load and prepare data
    df = load_and_prepare_data(str(data_path))

    if len(df) < 10:
        print(
            f"\nWARNING: Only {
                len(df)} bets available. Need at least 20 for meaningful diagnostics."
        )  # noqa: E501
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)

    # Initialize diagnostics
    print("\n" + "=" * 80)
    print("INITIALIZING DIAGNOSTIC FRAMEWORK")
    print("=" * 80)

    diagnostics = BettingDiagnostics(df)

    # Run all diagnostics
    print("\nRunning comprehensive diagnostics...")
    results = diagnostics.run_all_diagnostics()

    # Generate output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "validation_results"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate report
    report_path = output_dir / "betting_diagnostics_report.txt"
    print("\nGenerating text report...")
    report_text = diagnostics.generate_report(save_path=str(report_path))

    # Print to console
    print("\n" + "=" * 80)
    print(report_text)
    print("=" * 80)

    # Create visualizations
    print("\nGenerating diagnostic visualizations...")
    try:
        diagnostics.create_visualizations(save_dir=str(output_dir))
        print(
            f"✓ Visualizations saved to: {
                output_dir /
                'diagnostic_plots.png'}"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not create visualizations: {e}")

    # Save results as JSON
    json_path = output_dir / "betting_diagnostics_results.json"
    print("\nSaving JSON results...")

    # Convert results to JSON-serializable format
    import json

    import numpy as np

    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer, np.floating)):
                    json_results[key][k] = float(v)
                elif isinstance(v, pd.DataFrame):
                    json_results[key][k] = v.to_dict()
                elif isinstance(v, pd.Series):
                    json_results[key][k] = v.to_dict()
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"✓ JSON results saved to: {json_path}")

    # Summary recommendations
    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS & RECOMMENDATIONS")
    print("=" * 80)

    recommendations = diagnostics._generate_recommendations()
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✓ No critical issues detected. Model appears well-calibrated.")
        print("  Continue monitoring with monthly diagnostics.")

    # Health score
    print("\n" + "=" * 80)
    print("MODEL HEALTH SCORE")
    print("=" * 80)

    score = 0
    max_score = 7

    # 1. Edge correlation
    if results["proxy_clv"]["edge_correlation"] > 0.15:
        score += 1
        print("✓ [1 / 1] Proxy CLV: Strong edge correlation")
    elif results["proxy_clv"]["edge_correlation"] > 0.05:
        score += 0.5
        print("⚠ [0.5 / 1] Proxy CLV: Weak edge correlation")
    else:
        print("✗ [0 / 1] Proxy CLV: No edge detected")

    # 2. Kelly calibration
    if abs(results["stake_diagnostics"]["stake_win_corr"]) < 0.05:
        score += 1
        print("✓ [1 / 1] Stake: Well-calibrated Kelly")
    elif abs(results["stake_diagnostics"]["stake_win_corr"]) < 0.10:
        score += 0.5
        print("⚠ [0.5 / 1] Stake: Minor Kelly issues")
    else:
        print("✗ [0 / 1] Stake: Kelly overconfidence detected")

    # 3. Side bias
    if abs(results["side_bias"]["wr_gap"]) < 0.05:
        score += 1
        print("✓ [1 / 1] Side Bias: Well-balanced")
    elif abs(results["side_bias"]["wr_gap"]) < 0.10:
        score += 0.5
        print("⚠ [0.5 / 1] Side Bias: Moderate imbalance")
    else:
        print("✗ [0 / 1] Side Bias: Large OVER/UNDER gap")

    # 4. Edge monotonicity
    if results["edge_monotonicity"]["is_monotonic"]:
        score += 1
        print("✓ [1 / 1] Monotonicity: Edges properly calibrated")
    else:
        print("✗ [0 / 1] Monotonicity: Non-monotonic edges")

    # 5. Calibration
    if results["calibration"]["ece"] < 0.05:
        score += 1
        print("✓ [1 / 1] Calibration: Excellent ECE")
    elif results["calibration"]["ece"] < 0.10:
        score += 0.5
        print("⚠ [0.5 / 1] Calibration: Acceptable ECE")
    else:
        print("✗ [0 / 1] Calibration: Poor ECE")

    # 6. ROI positive
    if results["roi_mystery"]["actual_roi"] > 3:
        score += 1
        print("✓ [1 / 1] ROI: Profitable (> 3%)")
    elif results["roi_mystery"]["actual_roi"] > 0:
        score += 0.5
        print("⚠ [0.5 / 1] ROI: Slightly profitable")
    else:
        print("✗ [0 / 1] ROI: Losing money")

    # 7. Advanced diagnostics
    if (
        results["advanced"]["sigma_calibrated"] and results["advanced"]["has_true_edge"]
    ):  # noqa: E501
        score += 1
        print("✓ [1 / 1] Advanced: Sigma calibrated & true edge")
    elif (
        results["advanced"]["sigma_calibrated"] or results["advanced"]["has_true_edge"]
    ):  # noqa: E501
        score += 0.5
        print("⚠ [0.5 / 1] Advanced: Partial pass")
    else:
        print("✗ [0 / 1] Advanced: Failed diagnostics")

    # Overall assessment
    print(
        f"\nOVERALL HEALTH SCORE: {score:.1f}/{max_score} ({score / max_score * 100:.1f}%)"
    )  # noqa: E501

    if score >= 6:
        print(
            "🟢 EXCELLENT: Model is well-calibrated and profitable. Continue betting."
        )  # noqa: E501
    elif score >= 4:
        print(
            "🟡 ACCEPTABLE: Model has issues but is salvageable. Implement recommended fixes."
        )  # noqa: E501
    else:
        print(
            "🔴 POOR: Model has critical issues. STOP BETTING and fix issues before continuing."
        )  # noqa: E501

    # File summary
    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"📄 Text Report: {report_path}")
    print(f"📊 Visualizations: {output_dir / 'diagnostic_plots.png'}")
    print(f"📋 JSON Results: {json_path}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
