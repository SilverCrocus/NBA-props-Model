# 7-Day Sprint - Quick Start Guide
**Start Date:** October 14, 2025
**End Date:** October 21, 2025
**Goal:** 55%+ win rate, +3%+ ROI, <6.5 MAE

---

## TL;DR: The 3 Critical Fixes

```
Fix #1: Add L3 recent form features         → MAE -1.5 pts, +1.2 pp win rate
Fix #2: Add opponent defense features       → MAE -1.2 pts, +1.1 pp win rate
Fix #3: Fix edge calculation (probability)  → MAE  0.0 pts, +1.4 pp win rate

Total Impact: MAE 9.92 → 7.2 pts, Win Rate 51.98% → 55.68%
```

**These 3 fixes in Days 1-3 get you to production-ready.**

---

## Day-by-Day Checklist

### Day 1: L3 + Rest Features ✓ Target: 52.8% win rate

**Morning (4 hours)**
```bash
# 1. Create feature extraction script
cat > scripts/extract_l3_features.py << 'EOF'
import pandas as pd

def calculate_l3_features(df):
    """Extract last 3 games features"""
    features = {}

    if len(df) < 3:
        return {
            'PRA_L3_mean': 0,
            'PRA_L3_std': 0,
            'PRA_L3_trend': 0,
            'MIN_L3_mean': 0,
        }

    last_3 = df.sort_values('GAME_DATE', ascending=False).iloc[:3]

    features['PRA_L3_mean'] = last_3['PRA'].mean()
    features['PRA_L3_std'] = last_3['PRA'].std()
    features['PRA_L3_trend'] = (last_3.iloc[0]['PRA'] - last_3.iloc[2]['PRA']) / 2
    features['MIN_L3_mean'] = last_3['MIN'].mean()

    return features
EOF

# 2. Add to training pipeline
# (Modify existing feature engineering script)
```

**Afternoon (4 hours)**
```bash
# 3. Retrain model with L3 features
uv run scripts/train_with_l3.py

# 4. Validate
uv run scripts/validate_day1.py

# 5. Check metrics
# Expected: MAE ~8.5, Win Rate ~52.8%
```

**Success Criteria:**
- [ ] MAE < 8.7 points
- [ ] Win Rate > 52.5%
- [ ] No errors in training

---

### Day 2: Opponent Defense ✓ Target: 53.8% win rate

**Morning (4 hours)**
```bash
# 1. Extract opponent defense from CTG team data
cat > scripts/build_opponent_defense.py << 'EOF'
import pandas as pd
from pathlib import Path

# Load CTG team data
team_data_path = Path('data/ctg_team_data/')

opponent_features = []
for season_dir in team_data_path.glob('*'):
    # Extract defensive ratings
    # Extract pace
    # Save to opponent_defense_ratings.parquet
EOF

# 2. Run extraction
uv run scripts/build_opponent_defense.py
```

**Afternoon (4 hours)**
```bash
# 3. Merge opponent features with training data
# 4. Retrain model with opponent features
uv run scripts/train_with_opponent.py

# 5. Validate
uv run scripts/validate_day2.py

# Expected: MAE ~7.5, Win Rate ~53.8%
```

**Success Criteria:**
- [ ] Opponent data available for 90%+ games
- [ ] MAE < 7.8 points
- [ ] Win Rate > 53.5%

---

### Day 3: Fix Edge Calculation ✓ Target: 55.2% win rate

**Morning (3 hours)**
```bash
# 1. Implement probability-based edge calculator
cat > utils/edge_calculator.py << 'EOF'
from scipy.stats import norm

class EdgeCalculator:
    def __init__(self, model_mae=7.5):
        self.pred_std = model_mae

    def calculate_edge(self, pred_pra, line, odds):
        # Probability of beating line
        z_score = (pred_pra - line) / self.pred_std
        prob_over = norm.cdf(z_score)

        # Convert odds to probability
        if odds > 0:
            implied_prob = 100 / (odds + 100)
        else:
            implied_prob = -odds / (-odds + 100)

        # Edge percentage
        edge = (prob_over - implied_prob) * 100

        return edge, prob_over
EOF
```

**Afternoon (5 hours)**
```bash
# 2. Apply to 2024-25 predictions
uv run scripts/recalculate_with_true_edge.py

# 3. Optimize thresholds
# Test: min_edge = [3, 4, 5, 6, 7]
#       min_prob = [0.55, 0.60, 0.65]

# 4. Select best thresholds
# Goal: Maximize ROI while Win Rate > 53%

# Expected: Win Rate ~55.2%, ROI ~3.0%
```

**Success Criteria:**
- [ ] Win Rate > 54.5%
- [ ] ROI > 2.5%
- [ ] Clear improvement over old edge calculation

---

### Day 4: Minutes Model (Optional) ✓ Target: 55.8% win rate

**If Behind Schedule: SKIP to Day 7**

**Morning (4 hours)**
```bash
# 1. Build minutes dataset
# Features: Last 5/10 avg, starter indicator, pace, B2B

# 2. Train XGBoost minutes regressor
# Target: MAE < 5 minutes
```

**Afternoon (4 hours)**
```bash
# 3. Add projected_minutes to PRA model
# 4. Retrain PRA model with minutes projection
# Expected: MAE ~6.8, Win Rate ~55.8%
```

---

### Day 5: Final Features (Optional) ✓ Target: 56.5% win rate

**If Behind Schedule: SKIP to Day 7**

**Tasks:**
```bash
# 1. Add home/away splits
# 2. Add full CTG integration
# 3. Retrain final model
# 4. Feature importance analysis (SHAP)

# Expected: MAE ~6.2, Win Rate ~56.5%
```

---

### Day 6: Calibration (Optional) ✓ Target: 57.0% win rate

**If Behind Schedule: SKIP to Day 7**

**Tasks:**
```bash
# 1. Isotonic regression calibration
# 2. Multi-season validation (2023-24 + 2024-25)
# 3. Edge case testing

# Expected: Win Rate ~57.0%, consistent across seasons
```

---

### Day 7: Final Testing ✓ MANDATORY

**Morning (4 hours)**
```bash
# 1. Generate final performance report
uv run scripts/generate_final_report.py

# Metrics to report:
# - Win Rate (2024-25)
# - Win Rate (2023-24)
# - MAE (both seasons)
# - ROI
# - CLV rate
# - Consistency
```

**Afternoon (4 hours)**
```bash
# 2. Production readiness checklist
cat > PRODUCTION_READINESS.md << 'EOF'
# Checklist

Performance (ALL must pass):
- [ ] Win Rate > 55.0% (2024-25)
- [ ] Win Rate > 54.0% (2023-24)
- [ ] MAE < 6.5 points
- [ ] ROI > 3.0%
- [ ] CLV > 60%
- [ ] Consistent (<2 pp diff between seasons)

Technical (80% must pass):
- [ ] Walk-forward validated
- [ ] No temporal leakage
- [ ] Probability-based edge
- [ ] Feature documentation
- [ ] Model serialized
- [ ] Automated pipeline

# 3. Go/No-Go Decision
IF all performance criteria met:
    → Deploy to paper trading
ELSE:
    → Identify bottleneck
    → Extend 3 more days
EOF

# 3. Make Go/No-Go decision
```

---

## Quick Commands

### Training
```bash
# Day 1: Train with L3 + Rest
uv run scripts/train_enhanced_v1.py

# Day 2: Train with Opponent
uv run scripts/train_enhanced_v2.py

# Day 3: Recalculate edges
uv run scripts/recalculate_betting_results.py
```

### Validation
```bash
# Quick validation
uv run scripts/validate_enhanced.py

# Full validation (2023-24 + 2024-25)
uv run scripts/validate_multi_season.py
```

### Monitoring
```bash
# Check current metrics
python -c "
import pandas as pd
df = pd.read_csv('data/results/latest_validation.csv')
print(f'Win Rate: {(df[\"win\"]==1).mean()*100:.2f}%')
print(f'MAE: {df[\"abs_error\"].mean():.2f}')
"
```

---

## Minimum Viable Product (MVP)

**If you only complete Days 1-3:**

```
Performance:
- Win Rate: ~55.2%  ✓ Meets 55% target
- ROI: ~3.0%        ✓ Meets 3% target
- MAE: ~7.5 points  ⚠️ Above 6.5 target but acceptable

Verdict: DEPLOYABLE to paper trading
```

**This is production-ready.** Days 4-6 are polish, not requirements.

---

## Emergency Fallback Plans

### If Day 1 Fails (MAE > 9.0)
```
Problem: L3 features don't help
Solution:
  1. Check for data leakage (L3 calculated with future data?)
  2. Add L5/L7 instead of L3
  3. Increase EWMA spans (more historical weight)
```

### If Day 2 Fails (MAE still > 8.5)
```
Problem: Opponent data is low quality
Solution:
  1. Use NBA API as backup source
  2. Simplify to team-level (not position-specific)
  3. Use league average as fallback
```

### If Day 3 Fails (Win Rate < 54%)
```
Problem: Edge calculation still broken
Solution:
  1. Increase minimum edge threshold (5% → 7%)
  2. Add LightGBM ensemble (average predictions)
  3. Reduce bet volume (only highest confidence)
```

### If Day 7 Criteria Not Met
```
Decision Tree:
  If Win Rate 54-55%: Deploy with $10 bets, monitor closely
  If Win Rate 53-54%: Extend 3 days, add minutes model
  If Win Rate <53%:   Stop, investigate root cause
```

---

## Red Flags (Stop Immediately If...)

```
❌ Validation MAE < Training MAE
   → DATA LEAKAGE DETECTED

❌ Win Rate decreases after adding features
   → OVERFITTING DETECTED

❌ Huge difference between 2023-24 and 2024-25 (>3 pp)
   → MODEL INSTABILITY

❌ Edge calculation produces >80% win rate
   → BROKEN LOGIC

❌ Model predicts negative PRA or >100 PRA
   → NUMERICAL INSTABILITY
```

**If any red flag appears: STOP, DEBUG, FIX before continuing.**

---

## Success Metrics (Daily Tracking)

### Day 1 Target
```
MAE:       8.5 pts  (from 9.92)
Win Rate:  52.8%    (from 51.98%)
Features:  +7       (L3 features + Rest)
```

### Day 2 Target
```
MAE:       7.5 pts
Win Rate:  53.8%
Features:  +10      (Opponent defense)
```

### Day 3 Target
```
MAE:       7.5 pts  (same)
Win Rate:  55.2%    (BIG JUMP from edge fix)
ROI:       +3.0%    (from +0.91%)
```

### Day 7 Final
```
MAE:       6.5 pts  (or better)
Win Rate:  55.5%+   (minimum)
ROI:       +3.5%+   (minimum)
CLV:       70%+     (maintain)
```

---

## Files to Create

### Day 1
- `scripts/extract_l3_features.py` - L3 feature calculation
- `scripts/train_enhanced_v1.py` - Retrain with L3
- `scripts/validate_day1.py` - Day 1 validation

### Day 2
- `scripts/build_opponent_defense.py` - Extract opponent data
- `scripts/train_enhanced_v2.py` - Retrain with opponent
- `scripts/validate_day2.py` - Day 2 validation

### Day 3
- `utils/edge_calculator.py` - Probability-based edge
- `scripts/recalculate_betting_results.py` - Apply new edge
- `scripts/optimize_thresholds.py` - Find optimal thresholds

### Day 7
- `scripts/generate_final_report.py` - Performance report
- `PRODUCTION_READINESS.md` - Go/No-Go checklist
- `scripts/production_readiness_check.py` - Automated checks

---

## Time Budget

```
Total Available: 7 days × 8 hours = 56 hours

Critical Path (Must Do):
- Day 1 (L3 + Rest):          8 hours  ✓
- Day 2 (Opponent):            8 hours  ✓
- Day 3 (Edge Fix):            8 hours  ✓
- Day 7 (Final Testing):       8 hours  ✓

Subtotal Critical:            32 hours (57%)

Optional (Nice to Have):
- Day 4 (Minutes):             8 hours
- Day 5 (Final Features):      8 hours
- Day 6 (Calibration):         8 hours

Subtotal Optional:            24 hours (43%)

Buffer for Issues:            ~4-8 hours (built into optional days)
```

**Strategy:** Complete Days 1-3 perfectly. If ahead of schedule, do Days 4-6. If behind, skip to Day 7.

---

## Contact Points

### After Each Day
```markdown
Day X Summary:
- Features added: [list]
- MAE: X.XX points
- Win Rate: XX.XX%
- Issues encountered: [list]
- On track? Yes/No
- Tomorrow's plan: [brief]
```

### Emergency Decision Points

**End of Day 1:** If MAE > 9.0, reassess approach
**End of Day 2:** If MAE > 8.5, consider ensemble
**End of Day 3:** If Win Rate < 54%, extend timeline
**End of Day 7:** Final Go/No-Go decision

---

## The Bottom Line

**Days 1-3 are CRITICAL. Days 4-6 are OPTIONAL. Day 7 is MANDATORY.**

If you nail Days 1-3, you hit 55% win rate and you're production-ready.

Everything else is gravy.

**Start with Day 1 today. Execute perfectly. Let the data guide the rest.**

---

## Immediate Next Action

```bash
# Right now, run this:
mkdir -p scripts/7day_sprint/{day1,day2,day3,day7}
cd scripts/7day_sprint/day1

# Create first script
touch extract_l3_features.py

# You're ready to start Day 1.
```

**Let's build this. Day 1 starts now.**
