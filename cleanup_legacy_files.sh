#!/bin/bash

# NBA Props Model - Legacy File Cleanup Script
# Date: October 20, 2025
# Purpose: Remove 100 legacy/experimental files (~52.7MB) identified in audit
# Safety: All files have timestamps and are superseded by current versions

echo "=========================================================================="
echo "NBA PROPS MODEL - LEGACY FILE CLEANUP"
echo "=========================================================================="
echo ""
echo "This script will delete 100 legacy files (~52.7MB) identified as safe"
echo "to remove in the October 20, 2025 codebase audit."
echo ""
echo "Files to be deleted:"
echo "  - 13 legacy model files (keeping FIXED_V2 + calibrated)"
echo "  - 8 legacy production scripts (keeping current pipeline)"
echo "  - 4 experimental backtest scripts"
echo "  - 57 intermediate result files (keeping ULTRA_SELECTIVE + baseline)"
echo "  - 12 superseded documentation files (keeping final reports)"
echo "  - 6 archived research files"
echo ""
echo "SAFETY CHECK:"
echo "  ✓ All files timestamped from Oct 2025 development cycle"
echo "  ✓ Not referenced in current production code"
echo "  ✓ Have superior replacements in KEEP list"
echo ""
read -p "Are you SURE you want to proceed? (type 'YES' to confirm): " confirm

if [ "$confirm" != "YES" ]; then
    echo ""
    echo "Cleanup cancelled. No files were deleted."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Track deleted files and disk space saved
deleted_count=0
disk_saved=0

# Function to safely delete file if it exists
safe_delete() {
    local file="$1"
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        rm "$file"
        if [ $? -eq 0 ]; then
            echo "✓ Deleted: $file"
            ((deleted_count++))
            ((disk_saved+=size))
        else
            echo "✗ Failed to delete: $file"
        fi
    else
        echo "⊘ Not found (already deleted?): $file"
    fi
}

echo "=========================================================================="
echo "PRIORITY 1: MODELS (13 files, ~1.8MB)"
echo "=========================================================================="
echo ""

safe_delete "models/best_model_XGBoost.pkl"
safe_delete "models/production_model_latest.pkl"
safe_delete "models/production_model_oct2025_20251020_143055.pkl"
safe_delete "models/production_model_FIXED_latest.pkl"
safe_delete "models/production_model_FIXED_20251020_151841.pkl"
safe_delete "models/quantile_models.pkl"
safe_delete "models/isotonic_calibrator.pkl"
safe_delete "models/two_stage_minutes_pra_features.pkl"
safe_delete "models/two_stage_minutes_pra_minutes.cbm"
safe_delete "models/two_stage_minutes_pra_pra.cbm"
safe_delete "models/two_stage_model_features.pkl"
safe_delete "models/two_stage_model_minutes.cbm"
safe_delete "models/two_stage_model_pra.cbm"

echo ""
echo "=========================================================================="
echo "PRIORITY 1: PRODUCTION SCRIPTS (8 files)"
echo "=========================================================================="
echo ""

safe_delete "scripts/production/train_production_model_2025.py"
safe_delete "scripts/production/fetch_oct22_schedule.py"
safe_delete "scripts/production/predict_oct22_2025.py"
safe_delete "scripts/production/betting_recommendations_oct22.py"
safe_delete "scripts/production/train_model_FIXED.py"
safe_delete "scripts/production/create_betting_recommendations_oct22_V2.py"
safe_delete "scripts/production/apply_calibration_oct22.py"
safe_delete "scripts/production/fetch_oct22_games_and_odds.py"

echo ""
echo "=========================================================================="
echo "PRIORITY 1: DOCUMENTATION (12 files)"
echo "=========================================================================="
echo ""

safe_delete "EXECUTIVE_SUMMARY.md"
safe_delete "OVERFITTING_EXECUTIVE_SUMMARY.md"
safe_delete "OVERFITTING_DIAGNOSTIC_REPORT.md"
safe_delete "PHASE_1_2_IMPLEMENTATION_SUMMARY.md"
safe_delete "OPTIMAL_BETTING_STRATEGY.md"
safe_delete "OPTIMAL_BETTING_STRATEGY_RESEARCH.md"
safe_delete "ACTIONABLE_IMPROVEMENT_PLAN.md"
safe_delete "TRAINING_SCRIPT_FIXES.md"
safe_delete "WALK_FORWARD_VALIDATION_ASSESSMENT.md"
safe_delete "BACKTEST_ANALYSIS_FINDINGS.md"
safe_delete "NBA_PROPS_BACKTEST_ANALYSIS_REPORT.md"
safe_delete "BANKROLL_SIMULATION_RESULTS.md"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: BACKTEST SCRIPTS (4 files)"
echo "=========================================================================="
echo ""

safe_delete "scripts/production/monte_carlo_simulation.py"
safe_delete "scripts/production/backtest_optimal_strategy.py"
safe_delete "scripts/production/monte_carlo_optimal_strategy.py"
safe_delete "scripts/production/simulate_1000_bankroll.py"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: RESULT FILES - Pre-Calibration (20 files)"
echo "=========================================================================="
echo ""

safe_delete "data/results/backtest_2024_25_FIXED_V2.csv"
safe_delete "data/results/backtest_2024_25_FIXED_V2_betting.csv"
safe_delete "data/results/backtest_2024_25_corrected.csv"
safe_delete "data/results/backtest_2024_25_corrected_summary.json"
safe_delete "data/results/backtest_calibrated_2024_25.csv"
safe_delete "data/results/backtest_improved_edge_2024_25.csv"
safe_delete "data/results/backtest_improved_edge_2024_25_summary.json"
safe_delete "data/results/backtest_optimal_strategy.csv"
safe_delete "data/results/backtest_optimal_strategy_summary.json"
safe_delete "data/results/baseline_predictions_2024_25.csv"
safe_delete "data/results/feature_importance_2024_25.csv"
safe_delete "data/results/backtest_2024_25.csv"
safe_delete "data/results/backtest_2024_25_summary.json"
safe_delete "data/results/backtest_2024_25_analysis.json"
safe_delete "data/results/prediction_intervals_2024_25.csv"
safe_delete "data/results/residual_analysis_2024_25.csv"
safe_delete "data/results/metrics_by_player_type_2024_25.csv"
safe_delete "data/results/backtest_2024_25_betting_simulation.csv"
safe_delete "data/results/backtest_results.json"
safe_delete "data/results/model_performance_metrics.json"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: RESULT FILES - Historical Archive (8 files)"
echo "=========================================================================="
echo ""

safe_delete "data/results/backtest_2023_24_corrected.csv"
safe_delete "data/results/backtest_2023_24_corrected_summary.json"
safe_delete "data/results/baseline_predictions_2023-24.csv"
safe_delete "data/results/walkforward_predictions_2023-24.csv"
safe_delete "data/results/walkforward_predictions_2023-24.json"
safe_delete "data/results/two_stage_validation_2023_24.csv"
safe_delete "data/results/walkforward_validation_2023-24.json"
safe_delete "data/results/comprehensive_validation_2023-24.json"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: RESULT FILES - Experimental (20 files)"
echo "=========================================================================="
echo ""

safe_delete "data/results/walk_forward_predictions_leak_free_2024_25.csv"
safe_delete "data/results/walk_forward_predictions_advanced_2024_25.csv"
safe_delete "data/results/walkforward_predictions_2024-25.csv"
safe_delete "data/results/walkforward_predictions_2024-25_v2.csv"
safe_delete "data/results/walkforward_predictions_2024-25_v3.csv"
safe_delete "data/results/walkforward_predictions_v4.csv"
safe_delete "data/results/two_stage_predictions_2024_25.csv"
safe_delete "data/results/two_stage_predictions_minutes_pra_2024_25.csv"
safe_delete "data/results/tree_ensemble_predictions_2024_25.csv"
safe_delete "data/results/position_defense_predictions_2024_25.csv"
safe_delete "data/results/FINAL_BACKTEST_2024_25_CORRECTED.csv"
safe_delete "data/results/week1_improvements_predictions.csv"
safe_delete "data/results/WINNER_two_stage_backtest_2024_25.csv"
safe_delete "data/results/predictions_with_minutes_2024_25.csv"
safe_delete "data/results/advanced_validation_results.json"
safe_delete "data/results/walkforward_validation_2024-25.json"
safe_delete "data/results/comprehensive_validation_2024-25.json"
safe_delete "data/results/monte_carlo_results.json"
safe_delete "data/results/optimal_strategy_analysis.json"
safe_delete "data/results/edge_performance_analysis.json"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: RESULT FILES - Obsolete Predictions (5 files)"
echo "=========================================================================="
echo ""

safe_delete "data/results/betting_recommendations_oct22_2025.csv"
safe_delete "data/results/betting_recommendations_oct22_2025_V2.csv"
safe_delete "data/results/predictions_2025_10_22.csv"
safe_delete "data/results/predictions_2025_10_22_CALIBRATED.csv"
safe_delete "data/results/tomorrow_predictions.csv"

echo ""
echo "=========================================================================="
echo "PRIORITY 2: RESULT FILES - Data Quality (4 files)"
echo "=========================================================================="
echo ""

safe_delete "data/results/data_quality_report.json"
safe_delete "data/results/comprehensive_data_quality_assessment.json"
safe_delete "data/results/training_data_quality_report.json"
safe_delete "data/results/comprehensive_backtest_results.json"

echo ""
echo "=========================================================================="
echo "PRIORITY 3: RESEARCH FILES (6 files)"
echo "=========================================================================="
echo ""

safe_delete "research/EXECUTIVE_SUMMARY_1_WEEK.md"
safe_delete "research/QUICK_REFERENCE_CARD.md"
safe_delete "research/ONE_WEEK_TIMELINE_ANALYSIS.md"
safe_delete "research/RESEARCH_SUMMARY.md"
safe_delete "research/feature_engineering_checklist.md"
safe_delete "research/implementation_quick_reference.md"

echo ""
echo "=========================================================================="
echo "CLEANUP SUMMARY"
echo "=========================================================================="
echo ""
echo "Files deleted: $deleted_count"
echo "Disk space saved: $((disk_saved / 1024 / 1024))MB"
echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Remaining production files:"
echo "  Models: 5 (FIXED_V2, calibrated, calibrator, scaler, encoder)"
echo "  Scripts: 8 (FIXED_V2 pipeline + ultra-selective)"
echo "  Results: 8 (ultra-selective + calibrated + baseline)"
echo "  Docs: 9 (final reports + quick reference)"
echo ""
echo "Next step: Commit cleanup with message:"
echo "  'Cleanup: Remove 100 legacy files from Oct 2025 dev cycle'"
echo ""
echo "=========================================================================="
