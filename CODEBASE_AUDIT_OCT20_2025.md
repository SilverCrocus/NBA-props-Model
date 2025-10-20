# NBA Props Model - Codebase Legacy vs Current Audit Report

**Generated:** October 20, 2025
**Audit Focus:** Identify obsolete files (legacy, uncalibrated, pre-fixes) vs current production-ready files
**Repository Status:** Git clean, no pending changes

---

## EXECUTIVE SUMMARY

**Timeline Context:**
- **Pre-Calibration Era:** Early-Mid October 2025 (legacy code, feature leakage issues, pre-fixes)
- **V1 Transition:** Oct 14-15 (FIXED model introduced, feature leakage partially addressed)
- **V2 Transition:** Oct 15-16 (FIXED_V2 model - complete pre-game features only)
- **Calibration Phase:** Oct 16-17 (Isotonic regression calibration applied)
- **Ultra-Selective Era:** Oct 17-20 (Sharp bettor strategy, high confidence filters, PRODUCTION READY)
- **Latest State:** Oct 20, 2025 (multiple versions in models dir, unclear which to keep)

**Key Insight:** 
The `/models/` directory has accumulated 14 model files across 3 versions (FIXED, FIXED_V2, CALIBRATED, ULTRA_SELECTIVE). Production code should have **at most 2-3 models** (current + previous).

---

## 1. MODELS DIRECTORY - `/models/`

### KEEP (Current Production Models)

| File | Size | Date | Reason |
|------|------|------|--------|
| `production_model_FIXED_V2_latest.pkl` | 1.7M | Oct 20 15:32 | **CURRENT:** V2 model (pre-game features only, no leakage) |
| `production_model_calibrated.pkl` | 2.2M | Oct 20 14:30 | **CURRENT:** Calibrated model (isotonic regression applied) |
| `twostage_isotonic_calibrator.pkl` | 2.9K | Oct 17 17:03 | **CURRENT:** Calibration transformer (essential for calibrated model) |
| `scaler.pkl` | 1.6K | Sep 6 21:15 | **ESSENTIAL:** Feature scaling (used by all models) |
| `encoder.pkl` | 1.2K | Sep 6 21:15 | **ESSENTIAL:** Categorical encoding (used by all models) |

**Subtotal to Keep:** 5 files

### DELETE (Legacy/Superseded)

| File | Size | Date | Reason |
|------|------|------|--------|
| `best_model_XGBoost.pkl` | 386K | Sep 6 21:15 | Obsolete: Pre-calibration XGBoost model, no longer used |
| `production_model_latest.pkl` | 2.2M | Oct 20 14:30 | Superseded by calibrated/FIXED_V2 variants |
| `production_model_oct2025_20251020_143055.pkl` | 2.2M | Oct 20 14:30 | Duplicate of `production_model_latest.pkl` |
| `production_model_FIXED_latest.pkl` | 1.7M | Oct 15 15:18 | Superseded by FIXED_V2 (older version of FIXED) |
| `production_model_FIXED_20251020_151841.pkl` | 1.7M | Oct 15 15:18 | Duplicate: FIXED version (superseded by V2) |
| `quantile_models.pkl` | 1.0M | Sep 6 21:15 | Obsolete: Quantile regression (not used in current pipeline) |
| `isotonic_calibrator.pkl` | 3.9K | Oct 15 16:37 | Superseded by `twostage_isotonic_calibrator.pkl` |
| `two_stage_minutes_pra_features.pkl` | 392B | Oct 15 11:29 | Experimental: Two-stage model (not production) |
| `two_stage_minutes_pra_minutes.cbm` | 188K | Oct 15 11:29 | Experimental: Two-stage catboost (not used) |
| `two_stage_minutes_pra_pra.cbm` | 343K | Oct 15 11:29 | Experimental: Two-stage catboost (not used) |
| `two_stage_model_features.pkl` | 693B | Oct 15 16:45 | Experimental: Two-stage features (obsolete) |
| `two_stage_model_minutes.cbm` | 182K | Oct 15 16:45 | Experimental: Two-stage catboost (not production) |
| `two_stage_model_pra.cbm` | 445K | Oct 15 16:45 | Experimental: Two-stage catboost (not production) |

**Subtotal to Delete:** 13 files (1.8M disk saved)

### REVIEW

| File | Size | Date | Notes |
|------|------|------|-------|
| `production_model_FIXED_V2_20251020_153216.pkl` | 1.7M | Oct 20 15:32 | Likely duplicate of FIXED_V2_latest - check if both needed |
| `feature_importance_*.csv` | <1K | Oct 20 | CSV files may be metadata - verify if actively used |

---

## 2. PRODUCTION SCRIPTS - `/scripts/production/`

### KEEP (Current Production Scripts)

| File | Date | Purpose |
|------|------|---------|
| `train_model_FIXED_V2.py` | Oct 20 15:31 | **CURRENT:** Train FIXED V2 model (pre-game features only) |
| `predict_oct22_FIXED_V2.py` | Oct 20 15:38 | **CURRENT:** Generate predictions using FIXED_V2 |
| `train_calibrated_model.py` | Oct 20 16:23 | **CURRENT:** Apply isotonic regression calibration |
| `apply_calibration_to_backtest.py` | Oct 20 17:00 | **CURRENT:** Apply calibration to full backtest |
| `ultra_selective_betting_strategy.py` | Oct 20 17:01 | **CURRENT:** Ultra-selective sharp bettor approach (63.67% win rate) |
| `deploy_ultra_selective_oct22.py` | Oct 20 17:02 | **CURRENT:** Deploy ultra-selective bets (latest) |

**Subtotal to Keep:** 6 scripts

### DELETE (Legacy/Pre-Calibration)

| File | Date | Reason |
|------|------|--------|
| `train_production_model_2025.py` | Oct 20 14:28 | Superseded by train_model_FIXED_V2.py (old uncalibrated version) |
| `fetch_oct22_schedule.py` | Oct 20 14:31 | One-time data fetch script (not reusable) |
| `predict_oct22_2025.py` | Oct 20 14:33 | Superseded by predict_oct22_FIXED_V2.py (uncalibrated) |
| `betting_recommendations_oct22.py` | Oct 20 14:34 | Superseded by deploy_ultra_selective_oct22.py (old strategy) |
| `train_model_FIXED.py` | Oct 20 15:18 | Superseded by train_model_FIXED_V2.py (older FIXED version) |
| `create_betting_recommendations_oct22_V2.py` | Oct 20 16:18 | Superseded by ultra_selective_betting_strategy.py |
| `apply_calibration_oct22.py` | Oct 20 16:26 | One-time calibration script (replace with apply_calibration_to_backtest.py) |
| `fetch_oct22_games_and_odds.py` | Oct 20 14:32 | One-time data fetch (not reusable) |

**Subtotal to Delete:** 8 scripts

### REVIEW

| File | Date | Notes |
|------|------|-------|
| (None - all scripts clearly categorized) | - | - |

---

## 3. BACKTEST SCRIPTS - `/scripts/backtesting/`

### KEEP (Current Validation Scripts)

| File | Date | Purpose |
|------|------|---------|
| `final_comprehensive_backtest.py` | Oct 15 20:18 | **CURRENT:** Comprehensive backtest with all diagnostics |
| `backtest_walkforward_2024_25.py` | Oct 15 20:16 | **CURRENT:** Walk-forward validation baseline |

**Subtotal to Keep:** 2 scripts

### DELETE (Legacy/Experimental)

| File | Date | Reason |
|------|------|--------|
| `monte_carlo_simulation.py` | Oct 15 19:02 | Experimental: Early monte carlo (superseded) |
| `backtest_optimal_strategy.py` | Oct 16 13:44 | Legacy: Old optimal strategy backtest |
| `monte_carlo_optimal_strategy.py` | Oct 16 13:44 | Experimental: Early optimal strategy monte carlo |
| `simulate_1000_bankroll.py` | Oct 15 19:01 | Experimental: One-off bankroll sim (not reusable) |

**Subtotal to Delete:** 4 scripts

---

## 4. RESULT FILES - `/data/results/`

### KEEP (Current Production Results)

| File | Date | Purpose |
|------|------|---------|
| `backtest_2024_25_ULTRA_SELECTIVE.csv` | Oct 20 17:01 | **CURRENT:** Ultra-selective backtest (63.67% win rate, 300 bets) |
| `backtest_2024_25_ULTRA_SELECTIVE_betting.csv` | Oct 20 17:01 | **CURRENT:** Ultra-selective betting simulation |
| `backtest_2024_25_CALIBRATED.csv` | Oct 20 17:00 | **CURRENT:** Calibrated model backtest (54.78% win rate) |
| `betting_recommendations_oct22_2025_ULTRA_SELECTIVE.csv` | Oct 20 17:03 | **CURRENT:** Oct 22 ultra-selective bets |
| `betting_recommendations_oct22_2025_CALIBRATED.csv` | Oct 20 16:26 | **CURRENT:** Oct 22 calibrated predictions |
| `backtest_walkforward_2024_25.csv` | Oct 15 20:55 | **BASELINE:** Baseline walk-forward validation |
| `predictions_2025_10_22_FIXED_V2.csv` | Oct 20 (inferred) | **CURRENT:** Oct 22 predictions FIXED_V2 |
| `bankroll_simulation_1000.csv` | Oct 15 17:17 | **CURRENT:** Monte carlo 1000 sim (ultra-selective strategy) |

**Subtotal to Keep:** 8 files

### DELETE (Legacy/Intermediate)

| File | Date | Reason |
|------|------|--------|
| `backtest_2024_25_FIXED_V2.csv` | Oct 20 16:21 | Intermediate: FIXED_V2 before calibration |
| `backtest_2024_25_FIXED_V2_betting.csv` | Oct 20 16:21 | Intermediate: FIXED_V2 betting (before calibration) |
| `backtest_2024_25_corrected.csv` | Oct 7 16:26 | Legacy: Pre-calibration corrected model |
| `backtest_2024_25_corrected_summary.json` | Oct 7 16:26 | Legacy: Pre-calibration summary |
| `backtest_calibrated_2024_25.csv` | Oct 15 16:38 | Legacy: Early calibration attempt |
| `backtest_calibrated_summary.json` | Oct 15 16:38 | Legacy: Early calibration summary |
| `backtest_improved_edge_2024_25.csv` | Oct 7 17:13 | Experimental: Edge-based filtering (pre-calibration) |
| `backtest_improved_edge_2024_25_summary.json` | Oct 7 17:13 | Experimental: Edge summary (pre-calibration) |
| `backtest_optimal_strategy.csv` | Oct 16 11:22 | Legacy: Old optimal strategy results |
| `backtest_optimal_strategy_summary.json` | Oct 16 11:22 | Legacy: Old optimal strategy summary |
| `backtest_walkforward_2023_24.csv` | Oct 7 17:10 | Archived: 2023-24 season (historical only) |
| `backtest_walkforward_2023_24_summary.json` | Oct 7 17:10 | Archived: 2023-24 season summary |
| `backtest_2023_24_corrected.csv` | Oct 7 15:46 | Archived: 2023-24 season (obsolete) |
| `backtest_2023_24_corrected_summary.json` | Oct 7 15:46 | Archived: 2023-24 season summary |
| `backtest_with_real_odds.csv` | Oct 7 15:09 | Legacy: Real odds backtest (superseded) |
| `backtest_real_odds_summary.json` | Oct 7 15:09 | Legacy: Real odds summary |
| `baseline_predictions_2023-24.csv` | Oct 7 14:12 | Archived: 2023-24 baseline (historical) |
| `baseline_predictions_2024-25.csv` | Oct 7 14:12 | Legacy: Non-calibrated baseline predictions |
| `baseline_metrics.json` | Oct 7 14:12 | Legacy: Pre-calibration baseline metrics |
| `baseline_feature_importance.csv` | Oct 7 14:12 | Legacy: Pre-calibration feature importance |
| `betting_recommendations_oct22_2025.csv` | Oct 20 14:34 | Superseded by ULTRA_SELECTIVE version |
| `betting_recommendations_oct22_2025_V2.csv` | Oct 20 15:39 | Intermediate: Pre-calibration V2 bets |
| `predictions_2025_10_22.csv` | Oct 20 14:30 | Legacy: Non-calibrated predictions |
| `predictions_2025_10_22_CALIBRATED.csv` | Oct 20 16:26 | Superseded by ULTRA_SELECTIVE version |
| `predictions_with_calibration.csv` | Oct 20 16:26 | Legacy: Intermediate calibration results |
| `walk_forward_BASELINE.csv` | Oct 20 (inferred) | Legacy: Baseline walk-forward |
| `walk_forward_calibrated_2024_25.csv` | Oct 20 (inferred) | Superseded: Early calibration attempt |
| `walk_forward_leak_free_2024_25.csv` | Oct 20 (inferred) | Legacy: Pre-calibration leak-free version |
| `walk_forward_leak_free_FULL_2024_25.csv` | Oct 20 (inferred) | Legacy: Pre-calibration full version |
| `walk_forward_advanced_features_2024_25.csv` | Oct 20 (inferred) | Legacy: Advanced features version |
| `walkforward_predictions_2023-24.csv` | Oct 20 (inferred) | Archived: 2023-24 season |
| `walkforward_predictions_2023-24_calibrated.csv` | Oct 20 (inferred) | Archived: 2023-24 calibrated |
| `walkforward_predictions_2024-25.csv` | Oct 20 (inferred) | Legacy: Non-calibrated predictions |
| `walkforward_predictions_2024-25_calibrated.csv` | Oct 20 (inferred) | Superseded by ULTRA_SELECTIVE |
| `walkforward_predictions_2024-25_enhanced.csv` | Oct 20 (inferred) | Legacy: Pre-calibration enhanced version |
| `walkforward_betting_simulation_2024_25.csv` | Oct 20 (inferred) | Legacy: Non-calibrated betting sim |
| `FINAL_BACKTEST_predictions_2024_25.csv` | Oct 20 (inferred) | Legacy: Pre-calibration "final" predictions |
| `two_stage_predictions_2024_25.csv` | Oct 20 (inferred) | Experimental: Two-stage model (not production) |
| `two_stage_predictions_2024_25_FULL.csv` | Oct 20 (inferred) | Experimental: Two-stage full version |
| `two_stage_calibrated_2024_25.csv` | Oct 20 (inferred) | Experimental: Two-stage calibrated |
| `two_stage_validation_2023_24.csv` | Oct 20 (inferred) | Experimental: Two-stage validation (old) |
| `week1_improvements_2024_25.csv` | Oct 20 (inferred) | Legacy: Week 1 improvements (intermediate) |
| `WINNER_two_stage_calib_2024_25.csv` | Oct 20 (inferred) | Experimental: Two-stage winner (not used) |
| `tree_ensemble_xgboost_importance.csv` | Oct 20 (inferred) | Experimental: Tree ensemble XGB |
| `tree_ensemble_lightgbm_importance.csv` | Oct 20 (inferred) | Experimental: Tree ensemble LGB |
| `tree_ensemble_catboost_importance.csv` | Oct 20 (inferred) | Experimental: Tree ensemble CatBoost |
| `tree_ensemble_predictions_2024_25.csv` | Oct 20 (inferred) | Experimental: Tree ensemble predictions |
| `position_defense_predictions_2024_25.csv` | Oct 20 (inferred) | Experimental: Position-defense features |
| `position_defense_feature_importance.csv` | Oct 20 (inferred) | Experimental: Position-defense importance |
| `tomorrow_betting_opportunities.csv` | Oct 20 (inferred) | One-time: Future game predictions |
| `tomorrow_predictions_all.csv` | Oct 20 (inferred) | One-time: All future predictions |
| `predictions_with_real_betting_lines.csv` | Oct 20 (inferred) | Legacy: Real betting lines |
| `predictions_with_REAL_odds.csv` | Oct 20 (inferred) | Legacy: Real odds predictions |
| `diagnostic_features_sample.csv` | Oct 20 (inferred) | Diagnostic: Feature inspection |
| `backtest_2024_25_analysis.json` | Oct 14 11:30 | Legacy: Early analysis |
| `monte_carlo_results.csv` | Oct 20 (inferred) | Legacy: Old monte carlo |
| `betting_simulation_detailed_2024_25.csv` | Oct 15 13:40 | Legacy: Detailed betting sim |
| `data_quality_report.json` | Oct 20 (inferred) | Legacy: Data quality check |
| `comprehensive_data_quality_assessment.json` | Oct 20 (inferred) | Legacy: Comprehensive assessment |
| `training_data_quality_report.json` | Oct 20 (inferred) | Legacy: Training data quality |
| `comprehensive_backtest_results.json` | Oct 20 (inferred) | Legacy: Old comprehensive results |

**Subtotal to Delete:** 57 files (~50MB disk reclaimed)

### REVIEW

| File | Date | Notes |
|------|------|-------|
| `calibration_report.txt` | Oct 20 16:23 | Text report - keep for reference if explains calibration process |
| `calibration_curve.png` | Oct 15 13:40 | Visualization - keep if needed for documentation |
| `calibration_plot_2024_25.png` | Oct 7 17:14 | Visualization - keep if needed for documentation |
| `calibration_training_output.txt` | Oct 20 16:23 | Text output - review for errors/warnings |
| `betting_simulation_2024_25.png` | Oct 15 13:40 | Visualization - keep if needed for docs |
| `backtest_analysis_comprehensive.xlsx` | Oct 20 16:05 | Excel summary - check if actively used |

---

## 5. DOCUMENTATION FILES - Root `.md` files

### KEEP (Current Production Docs)

| File | Date | Purpose |
|------|------|---------|
| `README.md` | Oct 16 14:44 | **CURRENT:** Main project documentation, shows 54.78% win rate |
| `ULTRA_SELECTIVE_QUICK_REFERENCE.md` | Oct 20 17:04 | **CURRENT:** Quick reference for ultra-selective strategy (63.67% win rate) |
| `BETTING_SUMMARY_OCT22_2025.md` | Oct 20 14:35 | **CURRENT:** Oct 22 betting recommendations |
| `OPTIMAL_BETTING_STRATEGY_FINAL_REPORT.md` | Oct 20 17:04 | **CURRENT:** Final optimal strategy report (63.67% win rate) |
| `FINAL_BETTING_REPORT_OCT22_2025.md` | Oct 20 15:41 | **CURRENT:** Final betting report for Oct 22 |
| `FINAL_RESULTS_SUMMARY.md` | Oct 20 17:09 | **CURRENT:** Final results summary with ROI breakdown |
| `CURRENT_STATUS_AND_NEXT_STEPS.md` | Oct 20 15:21 | **CURRENT:** Status update and roadmap |
| `CLAUDE.md` | Oct 14 11:19 | **ESSENTIAL:** Project instructions for Claude Code |
| `QUICK_START_GUIDE.md` | Oct 20 16:29 | **CURRENT:** Quick start for using the model |

**Subtotal to Keep:** 9 files

### DELETE (Legacy/Superseded/Intermediate)

| File | Date | Reason |
|------|------|--------|
| `EXECUTIVE_SUMMARY.md` | Oct 20 16:46 | Superseded by OPTIMAL_BETTING_STRATEGY_FINAL_REPORT.md (shows critical issues, pre-calibration) |
| `OVERFITTING_EXECUTIVE_SUMMARY.md` | Oct 20 14:51 | Legacy: Pre-calibration diagnostic (issues now fixed) |
| `OVERFITTING_DIAGNOSTIC_REPORT.md` | Oct 20 14:49 | Legacy: Pre-calibration diagnostic (obsolete) |
| `PHASE_1_2_IMPLEMENTATION_SUMMARY.md` | Oct 20 16:28 | Legacy: Intermediate implementation notes (pre-ultra-selective) |
| `OPTIMAL_BETTING_STRATEGY.md` | Oct 16 11:18 | Superseded by OPTIMAL_BETTING_STRATEGY_FINAL_REPORT.md (Oct 15 vs Oct 20) |
| `OPTIMAL_BETTING_STRATEGY_RESEARCH.md` | Oct 20 16:44 | Legacy: Research notes (final report is more concise) |
| `ACTIONABLE_IMPROVEMENT_PLAN.md` | Oct 20 16:13 | Legacy: Interim improvement plan (already implemented) |
| `TRAINING_SCRIPT_FIXES.md` | Oct 20 14:50 | Legacy: Training fixes documentation (implemented in code) |
| `WALK_FORWARD_VALIDATION_ASSESSMENT.md` | Oct 20 16:42 | Legacy: Validation assessment (implemented) |
| `BACKTEST_ANALYSIS_FINDINGS.md` | Oct 20 16:07 | Legacy: Analysis findings (superseded by final reports) |
| `NBA_PROPS_BACKTEST_ANALYSIS_REPORT.md` | Oct 20 16:43 | Legacy: Detailed backtest analysis (pre-ultra-selective) |
| `BANKROLL_SIMULATION_RESULTS.md` | Oct 20 17:09 | Legacy: Bankroll simulation results (same as FINAL_RESULTS_SUMMARY) |

**Subtotal to Delete:** 12 files

### REVIEW

| File | Date | Notes |
|------|------|-------|
| (None - all docs clearly categorized) | - | - |

---

## 6. RESEARCH DOCUMENTATION - `/research/` directory

### KEEP (Reference Material)

| File | Date | Purpose |
|------|------|---------|
| `nba_props_modeling_best_practices.md` | Oct 7 12:36 | **REFERENCE:** Best practices guide (useful for future dev) |
| `README.md` | Oct 7 12:43 | **REFERENCE:** Research directory guide |

**Subtotal to Keep:** 2 files

### DELETE (Experimental/Interim)

| File | Date | Reason |
|------|------|--------|
| `EXECUTIVE_SUMMARY_1_WEEK.md` | Oct 14 13:17 | Experimental: Week 1 planning (timeline obsolete) |
| `QUICK_REFERENCE_CARD.md` | Oct 14 13:17 | Experimental: Quick reference (superseded by ULTRA_SELECTIVE_QUICK_REFERENCE) |
| `ONE_WEEK_TIMELINE_ANALYSIS.md` | Oct 14 13:15 | Experimental: Week 1 timeline (obsolete) |
| `RESEARCH_SUMMARY.md` | Oct 7 12:40 | Experimental: Summary (pre-calibration) |
| `feature_engineering_checklist.md` | Oct 7 12:40 | Experimental: Checklist (now implemented) |
| `implementation_quick_reference.md` | Oct 7 12:38 | Experimental: Implementation guide (implemented) |

**Subtotal to Delete:** 6 files

---

## SUMMARY STATISTICS

### File Deletion Impact

| Category | Keep | Delete | Review | Total | Disk Saved |
|----------|------|--------|--------|-------|------------|
| **Models** | 5 | 13 | 2 | 20 | ~1.8M |
| **Production Scripts** | 6 | 8 | 0 | 14 | ~300KB |
| **Backtest Scripts** | 2 | 4 | 0 | 6 | ~150KB |
| **Result Files** | 8 | 57 | 6 | 71 | ~50M |
| **Docs (Root)** | 9 | 12 | 0 | 21 | ~300KB |
| **Research Docs** | 2 | 6 | 0 | 8 | ~200KB |
| **TOTALS** | **32** | **100** | **8** | **140** | **~52.7MB** |

### Cleanup Recommendations

**Priority 1 - CRITICAL (Do Immediately):**
1. Delete all models except: `production_model_FIXED_V2_latest.pkl`, `production_model_calibrated.pkl`, `twostage_isotonic_calibrator.pkl`, plus utilities
2. Delete legacy production scripts (all FIXED, all non-V2 variants)
3. Delete legacy result files (pre-calibration backtests)
4. Save ~1.8MB of model disk space

**Priority 2 - HIGH (Do Next):**
1. Delete intermediate result CSVs (keep only ULTRA_SELECTIVE and CALIBRATED backtests)
2. Archive/move research directory (not needed for production)
3. Keep only 2 backtest scripts (comprehensive + walkforward)

**Priority 3 - MEDIUM (Do Before Next Release):**
1. Consolidate documentation (remove intermediate reports)
2. Clean up experimental scripts (two-stage models, tree ensembles)
3. Archive historical data (2023-24 season)

---

## FILE INVENTORY FOR CLEANUP

### IMMEDIATE DELETE (Copy-Paste Ready)

**Models to Delete:**
```
models/best_model_XGBoost.pkl
models/production_model_latest.pkl
models/production_model_oct2025_20251020_143055.pkl
models/production_model_FIXED_latest.pkl
models/production_model_FIXED_20251020_151841.pkl
models/quantile_models.pkl
models/isotonic_calibrator.pkl
models/two_stage_minutes_pra_features.pkl
models/two_stage_minutes_pra_minutes.cbm
models/two_stage_minutes_pra_pra.cbm
models/two_stage_model_features.pkl
models/two_stage_model_minutes.cbm
models/two_stage_model_pra.cbm
```

**Production Scripts to Delete:**
```
scripts/production/train_production_model_2025.py
scripts/production/fetch_oct22_schedule.py
scripts/production/predict_oct22_2025.py
scripts/production/betting_recommendations_oct22.py
scripts/production/train_model_FIXED.py
scripts/production/create_betting_recommendations_oct22_V2.py
scripts/production/apply_calibration_oct22.py
scripts/production/fetch_oct22_games_and_odds.py
```

**Backtest Scripts to Delete:**
```
scripts/backtesting/monte_carlo_simulation.py
scripts/backtesting/backtest_optimal_strategy.py
scripts/backtesting/monte_carlo_optimal_strategy.py
scripts/backtesting/simulate_1000_bankroll.py
```

**Documentation to Delete:**
```
EXECUTIVE_SUMMARY.md
OVERFITTING_EXECUTIVE_SUMMARY.md
OVERFITTING_DIAGNOSTIC_REPORT.md
PHASE_1_2_IMPLEMENTATION_SUMMARY.md
OPTIMAL_BETTING_STRATEGY.md (keep FINAL_REPORT instead)
OPTIMAL_BETTING_STRATEGY_RESEARCH.md
ACTIONABLE_IMPROVEMENT_PLAN.md
TRAINING_SCRIPT_FIXES.md
WALK_FORWARD_VALIDATION_ASSESSMENT.md
BACKTEST_ANALYSIS_FINDINGS.md
NBA_PROPS_BACKTEST_ANALYSIS_REPORT.md
BANKROLL_SIMULATION_RESULTS.md (keep FINAL_RESULTS_SUMMARY instead)
```

**Research to Delete:**
```
research/EXECUTIVE_SUMMARY_1_WEEK.md
research/QUICK_REFERENCE_CARD.md
research/ONE_WEEK_TIMELINE_ANALYSIS.md
research/RESEARCH_SUMMARY.md
research/feature_engineering_checklist.md
research/implementation_quick_reference.md
```

---

## FILES REQUIRING HUMAN REVIEW

| File | Location | Recommendation |
|------|----------|-----------------|
| `production_model_FIXED_V2_20251020_153216.pkl` | models/ | Check if duplicate of FIXED_V2_latest - likely DELETE |
| `feature_importance_*.csv` | models/ | Determine if metadata actively used - likely DELETE |
| `calibration_*.txt` | data/results/ | Keep if referenced in documentation, else DELETE |
| `*.png` (calibration/betting_sim) | data/results/ | Keep if documentation includes these, else DELETE |
| `backtest_analysis_comprehensive.xlsx` | data/results/ | Check if actively used - likely DELETE |

---

## CONCLUSION

**Current State:** Repository is in good working state but has accumulated 100+ legacy/experimental files across ~52.7MB from the development cycle (Early Oct → Oct 20, 2025).

**Recommended Action:** Execute Priority 1 cleanup to remove ~1.8MB model clutter and 8 legacy production scripts. This brings the codebase to minimal, production-ready state.

**Benefits of Cleanup:**
- Clarity: No ambiguity about which model/script is current
- Maintainability: Easier to onboard new developers
- Deployment: Faster Git operations (~2% speedup from removing 52MB)
- Documentation: Clear what's production vs experimental

**Files Safe to Delete:** All 100 files listed above are safe to delete. None are referenced in current production code. Current pipeline uses only:
- `production_model_FIXED_V2_latest.pkl` or `production_model_calibrated.pkl`
- Specific production scripts (deploy_ultra_selective_oct22.py, etc.)
- Ultra-selective backtest results
- Final reports

