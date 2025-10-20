# Quick Start Guide - NBA Props Model (Phase 1 + 2)

**Model Version:** V2 Calibrated (Phase 1 + 2)
**Last Updated:** October 20, 2025
**Expected Win Rate:** 54-56%

---

## For October 22, 2025 Bets

### Top 5 Recommendations (Conservative)

```
1. Buddy Hield (GSW)       | OVER 15.5 | Pred: 22.2 | Edge: +6.7
2. Rui Hachimura (LAL)     | OVER 21.5 | Pred: 28.2 | Edge: +6.7
3. Austin Reaves (LAL)     | OVER 34.5 | Pred: 41.1 | Edge: +6.6
4. Draymond Green (GSW)    | OVER 23.5 | Pred: 29.9 | Edge: +6.4
5. Jonathan Kuminga (GSW)  | OVER 23.5 | Pred: 29.6 | Edge: +6.1
```

**Recommended Action:**
- Bet: $25-50 per bet
- Total wager: $125-250
- Expected return: +$7.50 to +$12.50 (3-5% ROI)

---

## How to Generate Predictions for Future Games

### Step 1: Collect Latest Odds
```bash
# (You'll need to manually download odds from your sportsbook)
# Save to: data/upcoming/odds_YYYY_MM_DD.csv
```

### Step 2: Generate Raw Predictions
```bash
# Use your existing prediction script
uv run python scripts/production/predict_upcoming_games.py
# Output: data/results/predictions_YYYY_MM_DD.csv
```

### Step 3: Apply Calibration
```bash
# Apply Phase 2 calibration
uv run python scripts/production/apply_calibration_oct22.py
# (Update date paths in script first)
# Output: data/results/betting_recommendations_YYYY_MM_DD_CALIBRATED.csv
```

---

## Key Model Improvements

### Phase 1: Betting Strategy (✅ Implemented)
- **What:** Skip 10+ pt edges (model overconfident)
- **Why:** 10+ edges had 49.74% win rate (losing)
- **Impact:** Win rate 50.58% → 52.03%

### Phase 2: Calibration (✅ Implemented)
- **What:** Isotonic regression to fix systematic bias
- **Why:** Model predicted ~7 pts too low on average
- **Impact:** MAE reduced 26.7% (8.90 → 6.52 pts)

### Combined Impact
- **Expected win rate:** 54-56%
- **Expected ROI:** +3-5%

---

## Betting Guidelines

### DO ✅
- Start with small bets ($25-50)
- Focus on highest confidence bets (top 5-10)
- Track every bet (player, line, pred, actual, win/loss)
- Re-evaluate after 20 bets
- Use calibrated predictions (not raw)

### DON'T ❌
- Bet more than 5% of bankroll per bet
- Chase losses with larger bets
- Ignore tracking (you need data to improve)
- Use raw predictions (always apply calibration)
- Bet on star players (excluded by strategy)

---

## Performance Tracking

### After Each Bet, Record:
1. Player name
2. Game date
3. Predicted PRA (calibrated)
4. Betting line
5. Edge size
6. Bet type (OVER/UNDER)
7. Actual PRA
8. Win/Loss

### Calculate Weekly:
- **Win rate:** Should be 54-56%
- **ROI:** Should be +3-5%
- **MAE:** Should be 6-7 pts

### Red Flags 🚩
- Win rate < 52% after 20 bets → Re-evaluate strategy
- MAE > 8 pts for 2 weeks → Re-calibrate model
- Mean residual > 2 pts → Systematic bias returning

---

## File Locations

### Models
- **Calibrated model:** `models/production_model_calibrated.pkl`
- **Base model:** `models/production_model_latest.pkl`

### Predictions
- **Oct 22 calibrated:** `data/results/predictions_2025_10_22_CALIBRATED.csv`
- **Oct 22 bets:** `data/results/betting_recommendations_oct22_2025_CALIBRATED.csv`

### Scripts
- **Train calibrator:** `scripts/production/train_calibrated_model.py`
- **Apply calibration:** `scripts/production/apply_calibration_oct22.py`
- **Backtest:** `scripts/production/backtest_2024_25_FIXED_V2.py`

### Documentation
- **Full summary:** `PHASE_1_2_IMPLEMENTATION_SUMMARY.md`
- **Root cause analysis:** `BACKTEST_ANALYSIS_FINDINGS.md`
- **Implementation plan:** `ACTIONABLE_IMPROVEMENT_PLAN.md`

---

## Troubleshooting

### "Predictions too high/low"
- Check calibration was applied: `predicted_PRA_calibrated` column should exist
- Verify using calibrated model: `models/production_model_calibrated.pkl`

### "No bets passing filters"
- Check edge range: Should be 5-7 pts
- Verify star exclusion: 61 players excluded
- Confirm odds loaded: Should have betting lines

### "Win rate below 52%"
- Small sample variance: Need 20+ bets for significance
- Track over 2-3 weeks before re-evaluating
- If persistent, consider re-calibration

---

## Support & Updates

### When to Re-Calibrate
1. Win rate < 52% for 3+ weeks
2. MAE > 8 pts for 2+ weeks
3. Systematic bias detected (mean residual > 2 pts)

### How to Re-Calibrate
```bash
uv run python scripts/production/train_calibrated_model.py
```

This will:
1. Use latest backtest data
2. Fit new isotonic regressor
3. Save updated calibrated model

---

## Quick Reference: Expected vs Actual

### Backtest Validation (2024-25 Season)
```
Phase 0 (Original):     50.58% win rate, -3.44% ROI
Phase 1 (Filter only):  52.03% win rate, -0.67% ROI ✅ Validated
Phase 1+2 (Calibrated): 54-56% win rate, +3-5% ROI  ⚠️ Projected
```

### October 22, 2025 Test
```
Predictions: 42 players
Matched to odds: 433 player-line combinations
Filtered bets: 9 recommendations
Bet size: $25-50 per bet
Expected wins: 5-6 out of 9 (55%)
Expected profit: +$7.50 to +$12.50
```

---

**Remember:** This is a long-term profitability strategy. Short-term variance is expected. Stay disciplined, track results, and adjust based on data.

**Good luck! 🍀**
