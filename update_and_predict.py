#!/usr/bin/env python3
"""
Update NBA Data & Make Predictions
===================================

Automates the full workflow:
1. Updates game logs with latest NBA data
2. Retrains production model (optional)
3. Generates betting recommendations

Usage:
    uv run update_and_predict.py                    # Today's games, $1000 bankroll  # noqa: E501
    uv run update_and_predict.py 2025 - 10 - 24         # Specific date
    uv run update_and_predict.py 2025 - 10 - 24 5000    # Specific date + bankroll  # noqa: E501
    uv run update_and_predict.py --skip-retrain     # Skip model retraining
    uv run update_and_predict.py --skip-update      # Skip data update
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def print_step(step_num: int, step_name: str):
    """Print formatted step header"""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 80)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status

    Args:
        cmd: Command as list of strings
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    print(f"\n▶ {description}...")
    print(f"   Command: {' '.join(cmd)}")

    try:
        __result = subprocess.run(cmd, check=True)  # noqa: F841
        print(f"✅ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update data, retrain model, and generate betting recommendations"
    )  # noqa: E501
    parser.add_argument(
        "date",
        nargs="?",
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "bankroll",
        nargs="?",
        type=float,
        default=1000.0,
        help="Bankroll in dollars (default: $1,000)",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip updating game logs (use existing data)",
    )
    parser.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Skip retraining model (use existing production model)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="moderate",
        choices=["conservative", "moderate", "aggressive", "maximum"],
        help="Betting strategy (default: moderate)",
    )

    args = parser.parse_args()

    # Set target date
    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")  # noqa: E501

    # Print header
    print("=" * 80)
    print("NBA BETTING WORKFLOW - AUTOMATED")
    print("=" * 80)
    print(f"\n📅 Target Date: {target_date}")
    print(f"💰 Bankroll: ${args.bankroll:,.2f}")
    print(f"📊 Strategy: {args.strategy.upper()}")
    print("\nSteps to execute:")

    step_count = 0
    if not args.skip_update:
        step_count += 1
        print(f"  {step_count}. Update game logs with latest NBA data")
    if not args.skip_retrain:
        step_count += 1
        print(f"  {step_count}. Retrain production ensemble model")
    step_count += 1
    print(f"  {step_count}. Generate betting recommendations")

    # Confirmation
    response = input("\n▶ Proceed with workflow? (y/n): ")
    if response.lower() != "y":
        print("❌ Aborted by user")
        return 1

    step_num = 0

    # ======================================================================
    # STEP 1: UPDATE GAME LOGS
    # ======================================================================
    if not args.skip_update:
        step_num += 1
        print_step(step_num, "UPDATE GAME LOGS")

        cmd = ["uv", "run", "python", "scripts/update_latest_games.py"]

        if not run_command(cmd, "Update latest game logs"):
            print("\n⚠️  WARNING: Game log update failed")
            print("   Continuing with existing data...")
            response = input("   Continue? (y/n): ")
            if response.lower() != "y":
                return 1
    else:
        print("\n⏭️  SKIPPED: Update game logs (using existing data)")

    # ======================================================================
    # STEP 2: RETRAIN PRODUCTION MODEL
    # ======================================================================
    if not args.skip_retrain:
        step_num += 1
        print_step(step_num, "RETRAIN PRODUCTION MODEL")

        cmd = ["uv", "run", "python", "scripts/production/train_ensemble_v1_production.py"]

        print("\n⏱️  Note: Model training takes 5 - 15 minutes depending on hardware")  # noqa: E501

        if not run_command(cmd, "Retrain production ensemble"):
            print("\n❌ ERROR: Model training failed")
            print("   Cannot proceed without a trained model")
            return 1
    else:
        print("\n⏭️  SKIPPED: Retrain model (using existing production model)")

        # Verify model exists
        model_dir = Path("models")
        model_files = [
            model_dir / "production_fold_1.pkl",
            model_dir / "production_fold_2.pkl",
            model_dir / "production_fold_3.pkl",
            model_dir / "production_meta.pkl",
        ]

        missing_files = [f for f in model_files if not f.exists()]

        if missing_files:
            print("\n❌ ERROR: Production model files missing:")
            for f in missing_files:
                print(f"   - {f}")
            print(
                "\n   Please run with model training enabled (remove --skip-retrain)"
            )  # noqa: E501
            return 1

        print("   ✅ Verified: Production model exists")

    # ======================================================================
    # STEP 3: GENERATE RECOMMENDATIONS
    # ======================================================================
    step_num += 1
    print_step(step_num, "GENERATE BETTING RECOMMENDATIONS")

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/production/daily_betting_recommendations.py",
        "--date",
        target_date,
        "--bankroll",
        str(args.bankroll),
        "--strategy",
        args.strategy,
        "--save-html",
    ]

    if not run_command(cmd, "Generate betting recommendations"):
        print("\n❌ ERROR: Failed to generate recommendations")
        return 1

    # ======================================================================
    # COMPLETION
    # ======================================================================
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE!")
    print("=" * 80)

    print("\n📂 Results saved to:")
    print(f"   CSV: data/betting/recommendations_{target_date}.csv")
    print(f"   HTML: data/betting/recommendations_{target_date}.html")

    print("\n💡 Quick Commands:")
    print(f"   View CSV:  cat data/betting/recommendations_{target_date}.csv")
    print(f"   Open HTML: open data/betting/recommendations_{target_date}.html")

    print("\n🎰 Good luck!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
