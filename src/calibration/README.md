# Corrected Calibration System

Production-ready implementation of the corrected mathematical formulations for the NBA Props betting calibration system.

## Overview

This system implements:

1. **Kelly Sizing with Side-Specific Rolling ECE** - Dynamic stake sizing with calibration quality penalty
2. **Beta Calibrator with Platt Fallback** - Hybrid calibration that adapts to sample size
3. **CLV Gate Function** - Side-specific validation using Closing Line Value
4. **Enhanced Bet Logger** - Comprehensive diagnostic tracking for model validation

## Components

### 1. Kelly Sizing (`kelly_sizing.py`)

Corrected Kelly sizing that prevents negative stakes and applies side-specific ECE penalties.

**Key Features:**
- Rolling ECE computation (last 50 bets) by side
- ECE multiplier clamped to [0, 1]
- Stricter penalty for UNDER bets (6% vs 5% ECE threshold)
- Graceful handling of insufficient data

**Usage:**
```python
from src.calibration.kelly_sizing import calculate_kelly_side_specific

# Recent bet history (last 50-100 bets)
recent_bets = pd.DataFrame({
    'side': ['OVER', 'OVER', 'UNDER', ...],
    'prob_cal': [0.60, 0.65, 0.58, ...],
    'won': [1, 0, 1, ...]
})

# Calculate stake
stake, diagnostics = calculate_kelly_side_specific(
    ev=0.08,           # 8% expected value
    odds=1.91,         # Decimal odds (-110 American)
    bankroll=10000,    # Current bankroll
    side='OVER',       # OVER or UNDER
    recent_bets=recent_bets,
    kelly_fraction=0.1,  # 10% of Kelly (conservative)
    max_bet_pct=0.05     # Cap at 5% of bankroll
)

print(f"Stake: ${stake:.2f}")
print(f"ECE Multiplier: {diagnostics['kelly_mult']:.2f}")
print(f"Rolling ECE: {diagnostics['ece_rolling']:.3f}")
```

**Diagnostics Returned:**
- `kelly_base`: Full Kelly fraction
- `kelly_frac`: Fractional Kelly (after fraction applied)
- `ece_rolling`: Rolling ECE for this side
- `kelly_mult`: ECE penalty multiplier [0, 1]
- `stake_final`: Final stake after all adjustments
- `blocked_by_ece`: True if ECE penalty blocked bet

**Mathematical Flow:**
```
1. Full Kelly: f* = EV / (decimal_odds - 1)
2. Fractional Kelly: f = f* × kelly_fraction
3. ECE Penalty:
   - If ECE <= threshold: multiplier = 1.0
   - If ECE > threshold: multiplier = max(0, 1 - (ECE - threshold) / threshold)
4. Final Stake: min(f × multiplier × bankroll, max_bet_pct × bankroll)
```

---

### 2. Beta Calibrator (`beta_calibrator.py`)

Hybrid calibrator that uses Beta calibration for large samples and Platt scaling for small samples.

**Key Features:**
- **NO clipping by default** (for accurate metrics)
- Clipping only when `clip_for_betting=True`
- Separate calibrators for OVER and UNDER
- Automatic fallback to Platt for n < 30
- Preserves row order for validation

**Usage:**
```python
from src.calibration.beta_calibrator import BetaCalibrator

# Training data
probs_raw = np.array([0.60, 0.65, 0.58, ...])
outcomes = np.array([1, 1, 0, ...])
sides = np.array(['OVER', 'OVER', 'UNDER', ...])

# Fit calibrator
cal = BetaCalibrator(min_samples_beta=30, calibrator_type='auto')
metrics = cal.fit(probs_raw, outcomes, sides, validate=True)

# For validation metrics (NO clipping)
probs_cal = cal.predict(probs_raw, sides, clip_for_betting=False)
ece = compute_ece(probs_cal, outcomes)
brier = brier_score_loss(outcomes, probs_cal)

# For production betting (WITH clipping [0.5, 0.999])
probs_cal_betting = cal.predict(probs_raw, sides, clip_for_betting=True)

# Save for production
cal.save('models/beta_calibrator.pkl')

# Load in production
cal_loaded = BetaCalibrator.load('models/beta_calibrator.pkl')
```

**Calibrator Selection:**
- `n >= 30`: Beta calibration (more flexible, 3+ parameters)
- `n < 30`: Platt scaling (more stable, 2 parameters)

**Validation Metrics:**
```python
cal.print_summary()

# Output:
# ========================================
# BETA CALIBRATOR SUMMARY
# ========================================
#
# 📊 Sample Sizes:
#    OVER:  102 samples (beta)
#    UNDER: 25 samples (platt)
#
# 📈 OVER Calibration:
#    Brier: 0.2134 -> 0.1987 (Δ -0.0147)
#    ECE:   0.0532 -> 0.0298 (Δ -0.0234)
#
# 📈 UNDER Calibration:
#    Brier: 0.2287 -> 0.2145 (Δ -0.0142)
#    ECE:   0.0687 -> 0.0421 (Δ -0.0266)
```

---

### 3. CLV Gate Function (`clv_gate.py`)

Side-specific validation using Closing Line Value (CLV).

**Key Features:**
- OVER: Always enabled (baseline)
- UNDER: Requires CLV validation
  - % beat closing > 55% (last 50 bets)
  - Mean CLV > 0% (last 50 bets)
- Side-specific EV thresholds

**Usage:**
```python
from src.calibration.clv_gate import check_clv_gate, get_ev_threshold, filter_bets_by_clv_gate

# CLV ledger
clv_ledger = pd.DataFrame({
    'side': ['UNDER'] * 60,
    'clv_pct': [0.025, 0.032, -0.015, ...],  # CLV as decimal
    'beat_closing': [1, 1, 0, ...]           # 1 = beat, 0 = lost
})

# Check CLV gate
passes, diagnostics = check_clv_gate('UNDER', clv_ledger, min_bets=50)

print(f"UNDER CLV Gate: {'✅ PASS' if passes else '❌ FAIL'}")
print(f"  % Beat Closing: {diagnostics['pct_beat_closing']:.1%}")
print(f"  Avg CLV: {diagnostics['avg_clv']:+.2%}")

# Get EV threshold
ev_threshold, reason = get_ev_threshold('UNDER', clv_ledger, tier=1)
print(f"EV Threshold: {ev_threshold:.0%} - {reason}")

# Filter betting opportunities
bets = pd.DataFrame({
    'side': ['OVER', 'UNDER', 'UNDER'],
    'ev': [0.05, 0.08, 0.04],
    'player': ['A', 'B', 'C'],
    'line': [25, 30, 35]
})

filtered_bets, stats = filter_bets_by_clv_gate(bets, clv_ledger, tier=1)
print(f"Passed: {stats['total_passed']}/{stats['total_opps']}")
```

**EV Thresholds:**

| Side  | Tier 1 (Unvalidated) | Tier 1 (Validated) | Tier 2 | Tier 3 |
|-------|---------------------|-------------------|--------|--------|
| OVER  | 4%                  | 4%                | 4%     | 3%     |
| UNDER | 6%                  | 4%                | 4%     | 3%     |

**CLV Validation Criteria:**
- Minimum 50 bets with CLV data
- % beat closing >= 55%
- Mean CLV >= 0%

---

### 4. Enhanced Bet Logger (`enhanced_bet_logger.py`)

Comprehensive bet logging with calibration and Kelly diagnostics.

**New Columns:**
- `prob_raw`: Uncalibrated probability
- `prob_cal`: Calibrated probability
- `q_novig`: No-vig fair probability
- `ev_pre`: EV before calibration
- `ev_post`: EV after calibration
- `kelly_base`: Full Kelly fraction
- `kelly_mult`: ECE penalty multiplier
- `kelly_frac`: Fraction applied (0.1 or 0.25)
- `stake_final`: Final stake
- `ece_rolling`: Rolling ECE at bet time
- `clv_pct_rolling`: Rolling % positive CLV

**Usage:**
```python
from src.calibration.enhanced_bet_logger import EnhancedBetLogger

# Initialize logger
logger = EnhancedBetLogger(ledger_path='data/enhanced_bet_ledger.csv')

# Kelly diagnostics from calculate_kelly_side_specific()
kelly_diag = {
    'kelly_base': 0.055,
    'kelly_mult': 0.85,
    'kelly_frac': 0.1
}

# Log bet entry
logger.log_bet(
    bet_id='bet_001',
    date='2025-01-15',
    player='LeBron James',
    side='OVER',
    line=35.5,
    entry_odds_dec=1.91,
    prob_raw=0.65,           # Uncalibrated
    prob_cal=0.60,           # Calibrated
    q_novig=0.52,            # No-vig fair prob
    ev_pre=0.08,             # Before calibration
    ev_post=0.05,            # After calibration
    kelly_diagnostics=kelly_diag,
    stake=467.50,
    predicted_pra=38.2,
    player_sigma=6.5,
    bankroll=10000,
    tier=1,
    ece_rolling=0.042,
    clv_pct_rolling=0.58
)

# Log closing line (T-2 minutes before game)
logger.log_closing_line(
    bet_id='bet_001',
    close_line=36.5,
    close_odds_dec=1.87,
    close_time='2025-01-15 19:28:00',
    opposite_close_odds_dec=1.95
)

# Log result (after game)
logger.log_result(bet_id='bet_001', result=42.0)

# Print summary
logger.print_summary()
```

**Summary Output:**
```
================================================================================
ENHANCED BET LEDGER SUMMARY
================================================================================

📊 Overview:
   Total bets logged: 127
   Bets with results: 127
   Bets with CLV: 115

💰 Performance:
   Win Rate: 54.3%
   Total Staked: $12,450.00
   Total Profit: +$687.50
   ROI: +5.5%

📈 Closing Line Value:
   % Beat Closing: 57.4%
   Avg CLV: +2.3%

🎯 Calibration:
   Avg Prob (raw): 0.625
   Avg Prob (cal): 0.587
   Avg EV (pre):   7.2%
   Avg EV (post):  5.1%

🔧 Kelly Sizing:
   Avg ECE Multiplier: 0.87
   Avg Rolling ECE: 0.038
```

---

## Production Integration

### Step-by-Step Workflow

**1. Train Calibrator (once per day/week)**
```python
from src.calibration.beta_calibrator import BetaCalibrator

# Load historical predictions and outcomes
train_data = load_walk_forward_predictions()

# Fit calibrator
cal = BetaCalibrator(min_samples_beta=30, calibrator_type='auto')
cal.fit(
    probs_raw=train_data['prob_raw'],
    outcomes=train_data['won'],
    sides=train_data['side'],
    validate=True
)

# Save for production
cal.save('models/beta_calibrator_2025_01_15.pkl')
```

**2. Daily Betting Pipeline**
```python
from src.calibration.beta_calibrator import BetaCalibrator
from src.calibration.kelly_sizing import calculate_kelly_side_specific
from src.calibration.clv_gate import filter_bets_by_clv_gate
from src.calibration.enhanced_bet_logger import EnhancedBetLogger

# Load components
cal = BetaCalibrator.load('models/beta_calibrator_2025_01_15.pkl')
logger = EnhancedBetLogger(ledger_path='data/enhanced_bet_ledger.csv')
clv_ledger = logger.ledger  # Use same ledger for CLV

# Get today's predictions
predictions = get_todays_predictions()

# Apply calibration (WITH clipping for betting)
predictions['prob_cal'] = cal.predict(
    predictions['prob_raw'],
    predictions['side'],
    clip_for_betting=True  # Clip to [0.5, 0.999]
)

# Calculate EV after calibration
predictions['ev_post'] = calculate_ev(predictions['prob_cal'], predictions['odds'])

# Filter by CLV gate
filtered_bets, stats = filter_bets_by_clv_gate(
    bets=predictions,
    clv_ledger=clv_ledger,
    tier=1,  # Tier 1 = stabilization
    min_bets=50
)

print(f"Betting opportunities: {len(filtered_bets)}/{len(predictions)}")

# Calculate stakes for each bet
for _, bet in filtered_bets.iterrows():
    stake, kelly_diag = calculate_kelly_side_specific(
        ev=bet['ev_post'],
        odds=bet['odds'],
        bankroll=get_current_bankroll(),
        side=bet['side'],
        recent_bets=logger.ledger.tail(100),
        kelly_fraction=0.1,  # Conservative 10% of Kelly
        max_bet_pct=0.05     # Cap at 5% of bankroll
    )

    # Log bet
    logger.log_bet(
        bet_id=generate_bet_id(),
        date=bet['date'],
        player=bet['player'],
        side=bet['side'],
        line=bet['line'],
        entry_odds_dec=bet['odds'],
        prob_raw=bet['prob_raw'],
        prob_cal=bet['prob_cal'],
        q_novig=bet['q_novig'],
        ev_pre=bet['ev_pre'],
        ev_post=bet['ev_post'],
        kelly_diagnostics=kelly_diag,
        stake=stake,
        predicted_pra=bet['predicted_pra'],
        player_sigma=bet['player_sigma'],
        bankroll=get_current_bankroll(),
        tier=1,
        ece_rolling=kelly_diag['ece_rolling'],
        clv_pct_rolling=logger.get_rolling_clv_pct(bet['side'], window=50)
    )

    # Place bet
    if stake > 0:
        place_bet(bet, stake)
```

**3. Capture Closing Lines (T-2 minutes)**
```python
# For each bet placed today
for bet in todays_bets:
    closing_data = get_closing_odds(bet['game_id'], bet['market'])

    logger.log_closing_line(
        bet_id=bet['bet_id'],
        close_line=closing_data['line'],
        close_odds_dec=closing_data['odds'],
        close_time=closing_data['timestamp'],
        opposite_close_odds_dec=closing_data['opposite_odds']
    )
```

**4. Log Results (after games)**
```python
# For each bet with completed game
for bet in completed_bets:
    result = get_game_result(bet['game_id'], bet['player'], bet['market'])

    logger.log_result(
        bet_id=bet['bet_id'],
        result=result
    )
```

**5. Daily Monitoring**
```python
# Generate reports
logger.print_summary()

# Check CLV validation
for side in ['OVER', 'UNDER']:
    passes, diag = check_clv_gate(side, logger.ledger, min_bets=50)
    print(f"\n{side} CLV Gate: {'✅ PASS' if passes else '❌ FAIL'}")
    print(f"  {diag['reason']}")

# Monitor ECE by side
from src.calibration.kelly_sizing import compute_ece_by_side

ece_by_side = compute_ece_by_side(logger.ledger, window=50)
print(f"\nRolling ECE:")
print(f"  OVER:  {ece_by_side['OVER']:.3f}")
print(f"  UNDER: {ece_by_side['UNDER']:.3f}")
```

---

## Testing

Run unit tests:
```bash
uv run pytest tests/test_calibration_system.py -v
```

Run example scripts:
```bash
# Kelly sizing
uv run python src/calibration/kelly_sizing.py

# Beta calibrator
uv run python src/calibration/beta_calibrator.py

# CLV gate
uv run python src/calibration/clv_gate.py

# Enhanced bet logger
uv run python src/calibration/enhanced_bet_logger.py
```

---

## Key Fixes Implemented

### 1. Kelly Sizing (FIXED)

**Old (Buggy):**
```python
kelly_adjusted = kelly_base * (1 - (ece - 0.05)/0.05)
stake = max(min(kelly_adjusted * 0.25 * bankroll, 0.05 * bankroll), 0)
# Problems:
# - Can go negative
# - Uses global ECE (not side-specific)
# - Uses all-time ECE (not rolling)
```

**New (Corrected):**
```python
# 1. Compute rolling ECE by side
ece_rolling = compute_rolling_ece(recent_bets, side, window=50)

# 2. Calculate penalty multiplier (clamped to [0, 1])
if ece_rolling <= ece_threshold:
    kelly_mult = 1.0
else:
    excess_ece = ece_rolling - ece_threshold
    penalty = excess_ece / ece_threshold
    kelly_mult = max(0.0, 1.0 - penalty)

# 3. Apply multiplier
stake = kelly_frac * kelly_mult * bankroll
stake = min(stake, max_bet_pct * bankroll)  # Cap
```

### 2. Beta Calibrator (FIXED)

**Old (Buggy):**
```python
prob_cal = calibrator.predict(prob_raw)
prob_cal = np.clip(prob_cal, 0.5, 0.999)  # ❌ WRONG for metrics
return prob_cal
```

**New (Corrected):**
```python
# For metrics (NO clipping)
prob_cal = calibrator.predict(prob_raw, sides, clip_for_betting=False)
ece = compute_ece(prob_cal, outcomes)

# For betting (WITH clipping)
prob_cal = calibrator.predict(prob_raw, sides, clip_for_betting=True)
```

### 3. Validation Metrics (FIXED)

**Old (Potential Bug):**
```python
probs_cal = np.concatenate([probs_cal_over, probs_cal_under])
outcomes_sorted = np.concatenate([outcomes[over_mask], outcomes[under_mask]])
# Risk: Order mismatch if not careful
```

**New (Corrected):**
```python
# Preserve original row order
probs_cal = np.zeros_like(probs_raw)
probs_cal[over_mask] = calibrator_over.predict(probs_raw[over_mask])
probs_cal[under_mask] = calibrator_under.predict(probs_raw[under_mask])
# Now probs_cal[i] matches outcomes[i]
```

---

## Configuration Recommendations

### Tier 1: Stabilization (first 50 bets)
```python
kelly_fraction = 0.1        # Conservative 10% of Kelly
max_bet_pct = 0.05          # Cap at 5% of bankroll
ece_threshold_over = 0.05   # 5% ECE threshold
ece_threshold_under = 0.06  # 6% ECE threshold (stricter)
ev_threshold_over = 0.04    # 4% EV
ev_threshold_under = 0.06   # 6% EV (unvalidated)
```

### Tier 2: Validation (50-200 bets)
```python
kelly_fraction = 0.15       # Moderate 15% of Kelly
max_bet_pct = 0.05          # Keep cap at 5%
# After UNDER CLV validation:
ev_threshold_under = 0.04   # Drop to 4% EV
```

### Tier 3: Production (200+ bets)
```python
kelly_fraction = 0.25       # Aggressive 25% of Kelly
max_bet_pct = 0.10          # Increase cap to 10%
ev_threshold = 0.03         # Lower to 3% EV (both sides)
# Requires strong CLV validation for both sides
```

---

## Performance Benchmarks

**Target Metrics (Tier 2/3):**
- Win Rate: 55-58%
- ROI: 5-10%
- % Beat Closing: 55-60%
- Avg CLV: +2-3%
- ECE (OVER): < 0.05
- ECE (UNDER): < 0.06

**Red Flags:**
- Win Rate < 52% (after 100 bets)
- % Beat Closing < 50% (after 50 bets)
- ECE > 0.10 (poor calibration)
- ROI < 0% (after 200 bets with positive CLV)

---

## Troubleshooting

### Issue: UNDER bets blocked by CLV gate

**Diagnosis:**
```python
passes, diag = check_clv_gate('UNDER', clv_ledger, min_bets=50)
print(f"Reason: {diag['reason']}")
print(f"% Beat Closing: {diag['pct_beat_closing']:.1%}")
print(f"Avg CLV: {diag['avg_clv']:+.2%}")
```

**Solutions:**
1. If insufficient data: Continue betting OVERs only, accumulate UNDER CLV data
2. If % beat closing < 55%: Model needs recalibration or feature improvement
3. If avg CLV < 0%: UNDER predictions are overconfident, increase ev_threshold

### Issue: ECE penalty reducing stakes too much

**Diagnosis:**
```python
from src.calibration.kelly_sizing import compute_ece_by_side

ece_by_side = compute_ece_by_side(ledger, window=50)
print(f"OVER ECE: {ece_by_side['OVER']:.3f}")
print(f"UNDER ECE: {ece_by_side['UNDER']:.3f}")
```

**Solutions:**
1. If ECE > 0.10: Recalibrate model (Beta/Platt not working)
2. If ECE 0.05-0.10: Acceptable, penalty is correct
3. If ECE < 0.05: Good calibration, no penalty applied

### Issue: Kelly multiplier always 0

**Diagnosis:**
```python
stake, diag = calculate_kelly_side_specific(...)
print(f"ECE Rolling: {diag['ece_rolling']:.3f}")
print(f"ECE Threshold: {diag['ece_threshold']:.3f}")
print(f"Kelly Mult: {diag['kelly_mult']:.2f}")
```

**Cause:** ECE >> threshold (e.g., 0.15 vs 0.05)

**Solution:** Retrain calibrator with more data or switch to Platt scaling

---

## Future Enhancements

1. **Adaptive ECE thresholds** - Learn optimal thresholds from historical data
2. **Multi-tier Kelly fractions** - Increase fraction as CLV improves
3. **Player-specific calibration** - Separate calibrators for high/low variance players
4. **Bankroll tracking** - Integrate with ledger for accurate Kelly sizing
5. **Real-time CLV monitoring** - Alert when CLV drops below threshold

---

## References

1. **Kelly Criterion:**
   - Thorp, E. (2008). "The Kelly Capital Growth Investment Criterion"
   - MacLean, L. et al. (2011). "The Kelly Capital Growth Investment Criterion"

2. **Calibration:**
   - Kull, M. et al. (2017). "Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities with Dirichlet Calibration"
   - Naeini, M. et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning"

3. **CLV:**
   - "Dissecting the Profitability of Closing Line Value" (Pinnacle, 2018)
   - "Why Beating the Closing Line is Key to Profitable Betting" (Pinnacle, 2019)

---

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review test cases in `tests/test_calibration_system.py`
3. Run example scripts to verify setup
4. Check logs for detailed diagnostics

---

**Last Updated:** October 25, 2025
**Version:** 1.0.0
**Status:** Production Ready ✅
