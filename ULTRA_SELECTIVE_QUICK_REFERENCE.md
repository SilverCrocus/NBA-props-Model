# Ultra-Selective Strategy - Quick Reference

**Win Rate:** 63.67% (validated)
**ROI:** +21.54%
**Volume:** 300 bets/season (~1-3 per day)

---

## October 22, 2025 Bets

```
✅ 1. Rui Hachimura (LAL) - OVER 21.5 (Quality: 0.840)
✅ 2. Buddy Hield (GSW) - OVER 15.5 (Quality: 0.795)
✅ 3. Draymond Green (GSW) - OVER 23.5 (Quality: 0.755)

Recommended: Bet $50-75 per bet ($150-225 total)
Expected wins: 2 out of 3
```

---

## Quick Stats

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| **Win Rate** | 63.67% | +13.1 pp |
| **ROI** | +21.54% | +25.0 pp |
| **Bets/Season** | 300 | -6,495 |
| **Significance** | p < 0.0001 | ✅ Validated |

---

## Strategy Rules

### 4-Tier Quality Scoring (Threshold: 0.75)

1. **Edge Quality (30%):** Prefer 6.5-7 pt edges
2. **Prediction Confidence (25%):** Sweet spot 18-28 PRA
3. **Game Context (25%):** Minutes >28, avoid B2Bs
4. **Player Consistency (20%):** Low variance preferred

### Filters
- ✅ Edge: 5-7 points (calibrated)
- ✅ Quality: ≥0.75 (top 20%)
- ❌ Exclude: 61 star players

---

## Bet Sizing

**Recommended: Fractional Kelly (0.25)**

- $1,000 bankroll → $50 per bet
- $2,000 bankroll → $100 per bet
- $5,000 bankroll → $250 per bet

---

## Daily Workflow

```bash
# 1. Generate predictions
uv run python scripts/production/predict_upcoming_games.py

# 2. Apply calibration
uv run python scripts/production/apply_calibration_to_predictions.py

# 3. Get ultra-selective bets
uv run python scripts/production/deploy_ultra_selective_oct22.py
```

---

## Tracking Checklist

After each bet:
- [ ] Record: Player, Line, Prediction, Actual, Win/Loss
- [ ] Update win rate (target: 63.67%)
- [ ] Update ROI (target: 21.54%)
- [ ] Check quality score consistency

Weekly review:
- [ ] Win rate ≥ 60%?
- [ ] ROI ≥ 15%?
- [ ] Mean residual < 2 pts?
- [ ] Quality scores consistent?

---

## Stop-Loss Triggers

**Re-evaluate if:**
- ❌ Win rate < 55% after 20 bets
- ❌ Mean residual > 2 pts (bias returning)
- ❌ Quality scores dropping
- ❌ Fewer than 200 bets/season

---

## Expected Performance by Volume

| Bets | Win Rate | Expected Profit |
|------|----------|-----------------|
| 10 | 60-67% | $100-200 (@$50/bet) |
| 50 | 61-66% | $500-1,000 |
| 100 | 62-65% | $1,000-2,000 |
| 300 | 63-65% | $3,000-6,000 |

---

## Key Files

**Models:**
- `models/production_model_calibrated.pkl`

**Scripts:**
- `scripts/production/deploy_ultra_selective_oct22.py`

**Results:**
- `data/results/betting_recommendations_oct22_2025_ULTRA_SELECTIVE.csv`

**Documentation:**
- `OPTIMAL_BETTING_STRATEGY_FINAL_REPORT.md` (full details)
- `ULTRA_SELECTIVE_QUICK_REFERENCE.md` (this file)

---

**Remember:** This is a marathon, not a sprint. Expect variance in small samples. Trust the process and track results diligently.

**Good luck! 🍀**
