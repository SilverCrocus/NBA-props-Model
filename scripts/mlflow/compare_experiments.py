"""
Compare experiments and generate reports
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mlflow_integration.utils import (
    compare_runs,
    get_best_model,
    generate_experiment_report,
    compare_phases,
    export_runs_to_csv
)
import pandas as pd


def compare_specific_runs(run_ids: list):
    """Compare specific runs by ID"""
    print("=" * 80)
    print("COMPARING SPECIFIC RUNS")
    print("=" * 80)
    print()

    comparison_df = compare_runs(
        run_ids=run_ids,
        metrics=['val_mae', 'betting_roi', 'betting_win_rate'],
        params=['n_estimators', 'max_depth', 'learning_rate']
    )

    print(comparison_df.to_string())
    print()


def show_best_models(experiment_name: str, metric: str = 'val_mae', top_n: int = 5):
    """Show best models from an experiment"""
    print("=" * 80)
    print(f"TOP {top_n} MODELS - {experiment_name}")
    print(f"Optimizing: {metric}")
    print("=" * 80)
    print()

    ascending = metric in ['val_mae', 'val_rmse']  # Lower is better for these

    best_models = get_best_model(
        experiment_name=experiment_name,
        metric=metric,
        ascending=ascending,
        top_n=top_n
    )

    if not best_models:
        print(f"No models found in experiment: {experiment_name}")
        return

    for i, model in enumerate(best_models, 1):
        print(f"{i}. {model.get('run_name', 'N/A')}")
        print(f"   Run ID: {model['run_id']}")
        print(f"   {metric}: {model.get('metric_value', 'N/A'):.4f}")

        # Show other key metrics
        if 'val_mae' in model and metric != 'val_mae':
            print(f"   MAE: {model['val_mae']:.4f}")
        if 'betting_roi' in model and metric != 'betting_roi':
            print(f"   ROI: {model['betting_roi']:.4f}")
        if 'betting_win_rate' in model and metric != 'betting_win_rate':
            print(f"   Win Rate: {model['betting_win_rate']:.4f}")

        print()


def generate_report(experiment_name: str, output_file: str = None):
    """Generate experiment report"""
    print(f"Generating report for experiment: {experiment_name}")
    print()

    report = generate_experiment_report(
        experiment_name=experiment_name,
        output_path=output_file
    )

    print(report)

    if output_file:
        print(f"\nReport saved to: {output_file}")


def compare_all_phases():
    """Compare best models from each phase"""
    print("=" * 80)
    print("PHASE COMPARISON - BEST MODELS")
    print("=" * 80)
    print()

    comparison_df = compare_phases()

    if comparison_df.empty:
        print("No completed phases found yet")
        return

    print(comparison_df.to_string(index=False))
    print()


def export_experiment(experiment_name: str, output_file: str):
    """Export experiment runs to CSV"""
    print(f"Exporting experiment: {experiment_name}")

    export_runs_to_csv(
        experiment_name=experiment_name,
        output_path=output_file
    )

    print(f"Exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare MLflow experiments")

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Compare runs command
    compare_parser = subparsers.add_parser('compare', help='Compare specific runs')
    compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')

    # Best models command
    best_parser = subparsers.add_parser('best', help='Show best models')
    best_parser.add_argument('experiment', help='Experiment name')
    best_parser.add_argument('--metric', default='val_mae', help='Metric to optimize')
    best_parser.add_argument('--top-n', type=int, default=5, help='Number of top models')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate experiment report')
    report_parser.add_argument('experiment', help='Experiment name')
    report_parser.add_argument('--output', help='Output file path')

    # Phases command
    subparsers.add_parser('phases', help='Compare all phases')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export experiment to CSV')
    export_parser.add_argument('experiment', help='Experiment name')
    export_parser.add_argument('output', help='Output CSV file path')

    args = parser.parse_args()

    if args.command == 'compare':
        compare_specific_runs(args.run_ids)

    elif args.command == 'best':
        show_best_models(args.experiment, args.metric, args.top_n)

    elif args.command == 'report':
        generate_report(args.experiment, args.output)

    elif args.command == 'phases':
        compare_all_phases()

    elif args.command == 'export':
        export_experiment(args.experiment, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
