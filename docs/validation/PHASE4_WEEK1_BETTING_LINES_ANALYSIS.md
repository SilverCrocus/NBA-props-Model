# Phase 4 Week 1: Betting Lines Analysis

**Date:** October 15, 2025
**Approach:** Market-based simulation of betting lines
**Result:** INCONCLUSIVE - Cannot determine profitability without real lines
**KEY INSIGHT:** Simulated lines based on own predictions create selection bias

---

## Executive Summary

**Goal:** Evaluate model profitability against betting lines

**Attempted Approach:** Create market-based simulation of sportsbook lines

**Result:** Simulation shows unrealistic profitability (81% win rate, 59% ROI)

**Root Cause:** Simulated lines use our predictions as base → circular logic creates artificial edge

**Real Conclusion:** MUST acquire real betting lines from sportsbooks to evaluate profitability

---

## What We Tried

### Attempt 1: Market Simulation with 85% Efficiency (Data Leakage)

**Method:**
```python
line = 0.90 * prediction + 0.10 * season_avg + 0.85 * (actual - line_base) + noise
```

**Problem:** Incorporated 85% of true outcome → data leakage
**Result:** Lines had MAE 0.92 (way better than our model's 6.10)
**Outcome:** Model showed 0% win rate (lines were too good)

### Attempt 2: Market Simulation with Independent Noise

**Method:**
```python
line = 0.90 * prediction + 0.10 * season_avg + noise(std=2.5)
```

**Problem:** Lines still based on our predictions → selection bias
**Result:** Lines had MAE 6.35 (similar to our model's 6.10)
**Outcome:** Model showed 81% win rate (unrealistically high)

---

## Why the Simulation Fails

### The Circular Logic Problem

**What we did:**
1. Train model → predictions with MAE 6.10
2. Create lines based on predictions + noise
3. Evaluate model against these lines
4. Find profitable opportunities

**What's wrong:**
```
When our model predicts 25 and line is 20 (edge = 5):
  - Line is based on our prediction (25) + noise
  - So line ≈ 25 + noise(2.5) = maybe 22-28
  - When line is low (20), it means noise was negative
  - But our prediction (25) was made independently
  - So we're essentially betting on our prediction vs noise
  - Of course we win more often!
```

**This is selection bias** - we're finding cases where our prediction differs from (our prediction + noise), not cases where we beat a truly independent model.

### Real Sportsbooks Are Different

**Real sharp sportsbooks (Pinnacle, Circa, etc.):**
- Have their own sophisticated models
- Use different features we don't have:
  - Real-time injury reports
  - Lineup data (who's starting)
  - Sharp bettor action (line movement)
  - Public betting patterns
- Might be better than us in some areas
- Might be worse in others
- Are INDEPENDENT of our model

**Our simulation:**
- Uses our predictions as starting point
- Adds random noise
- Not representative of real market

---

## What the Results Actually Mean

### Simulation Results (Grain of Salt)

| Edge Threshold | N Bets | Win Rate | ROI | Realistic? |
|-------|--------|----------|-----|------------|
| 0 points | 25,349 | 57.75% | +10.26% | Maybe |
| 3 points | 5,867 | 68.31% | +30.42% | Unlikely |
| 5 points | 1,150 | 75.04% | +43.26% | No |
| 6.5 points (optimal) | 258 | 81.78% | +59.17% | Definitely not |

### What This Tells Us

**Optimistic interpretation:**
- Model CAN find edges when lines differ from predictions
- Selective betting (high thresholds) improves win rate
- There's signal in the predictions

**Realistic interpretation:**
- Win rates >70% are unrealistic for sports betting
- Real sportsbooks won't set lines as naively
- True win rate likely 50-55% at best (if profitable at all)

**Pessimistic interpretation:**
- Simulation proves nothing
- Could be 48% win rate (losing) against real books
- Need actual data to know

---

## Why We Can't Use This Analysis

### Problem 1: Selection Bias

**Simulation logic:**
1. High edge = our prediction differs a lot from (our prediction + noise)
2. This happens when noise is large
3. But our prediction accuracy is independent of noise
4. So high edge cases have biased outcomes

**Real world:**
1. High edge = our prediction differs from sportsbook's independent model
2. Sportsbook might be right (we lose)
3. Or we might be right (we win)
4. No way to know without comparing models

### Problem 2: Unrealistic Win Rates

**Historical context:**
- Best sports bettors: 54-58% long-term
- Professional syndicates: 55-60%
- Our simulation: 82%

**This suggests:**
- Massive overfitting
- Selection bias
- Or we're the best sports bettors in history (unlikely)

### Problem 3: Can't Validate Approach

**Without real lines, we can't test:**
- Is our edge detection strategy valid?
- What threshold actually works?
- Is Kelly Criterion sizing appropriate?
- Will line shopping help?

---

## What We Actually Learned

### Positive Findings

1. **Model has structure** - predictions correlate with outcomes
2. **Edge detection works conceptually** - higher edges → higher accuracy
3. **CatBoost MAE 6.10 is reasonable** - competitive with simulated "market"

### Critical Gaps

1. **No independent validation** - circular logic invalidates results
2. **No benchmark** - don't know if 6.10 MAE beats real books
3. **No risk assessment** - can't measure true volatility

### Key Insight

**The only way forward is to acquire REAL betting lines from actual sportsbooks.**

---

## Options Moving Forward

### Option 1: Acquire Real Historical Betting Lines (REQUIRED)

**Sources:**
1. **The Odds API** - $99/month for historical data
   - Pro: Comprehensive, reliable, easy to use
   - Con: Expensive for hobby project
   
2. **SportsOddsHistory.com** - Free historical odds
   - Pro: Free
   - Con: Limited data, may not have player props

3. **Web Scraping** - DraftKings/FanDuel archives
   - Pro: Free, direct from books
   - Con: Legal grey area, time-consuming, fragile

4. **OddsPortal** - Community-maintained odds database
   - Pro: Free, decent coverage
   - Con: Data quality issues, manual export

**Recommendation:**  
Try free sources first (SportsOddsHistory, OddsPortal).  
If insufficient, invest in The Odds API for 1 month ($99).

**Timeline:** 3-7 days

**Deliverable:** `real_betting_lines_2024_25.csv` with actual sportsbook lines

---

### Option 2: Paper Trade with Live Lines (Alternative)

**If historical data unavailable:**

**Approach:**
1. Start tracking current games (rest of 2024-25 season)
2. Collect lines from DraftKings/FanDuel daily
3. Make predictions before games
4. Track results over 4-6 weeks
5. Calculate true win rate and ROI

**Pros:**
- Real lines, real validation
- Learn about line movement, availability
- No historical data cost

**Cons:**
- Takes 4-6 weeks
- Smaller sample size (~200-400 bets)
- Season-dependent (playoffs different)

**Timeline:** 4-6 weeks

**Deliverable:** Real win rate, ROI, lessons learned

---

### Option 3: Accept Uncertainty, Improve Model First

**Rationale:**
- Even if lines are favorable, MAE 6.10 might not be enough
- Focus on getting to MAE <5.50 first
- THEN test against real lines

**Approach:**
1. Implement minutes projection features (Phase 2-3 finding)
2. Target MAE 5.20-5.40
3. Re-evaluate profitability afterward

**Pros:**
- Clear improvement path
- Better model = better chance against any lines
- Don't waste time on unprofitable baseline

**Cons:**
- Might be already profitable (we don't know!)
- 2-3 weeks of work before validation
- Could be improving wrong thing

**Timeline:** 2-3 weeks

**Deliverable:** Improved model with MAE <5.50

---

## Recommended Path Forward

### Week 1 (Current): Real Lines Research

**Tasks:**
1. Check SportsOddsHistory.com for NBA player props data
2. Check OddsPortal for historical PRA lines
3. If not available: Budget for The Odds API ($99)

**Decision Point:**
- If real lines available: Proceed to Week 2 (evaluation)
- If not available: Choose Option 2 (paper trade) or Option 3 (improve first)

### Week 2-3 (If Real Lines Found): True Profitability Analysis

**Tasks:**
1. Match real lines to our predictions
2. Calculate actual win rates by edge threshold
3. Simulate betting with Kelly Criterion
4. Measure true ROI, Sharpe ratio, max drawdown

**Decision Point:**
- If win rate >53%: Proceed to paper trading
- If win rate 50-53%: Marginal, consider improvements
- If win rate <50%: Implement minutes features

### Week 4+ (If Profitable): Paper Trading

**Tasks:**
1. Track live lines daily
2. Place "paper bets" (simulated)
3. Track 200+ bets
4. Validate profitability in real conditions

**Decision Point:**
- If still profitable: Deploy with $500-1000 bankroll
- If not profitable: Back to feature engineering

---

## Key Metrics from Simulation (For Reference Only)

### Model Performance

- MAE: 6.10 points
- Simulated Line MAE: 6.35 points
- Model slightly better than simulated market

### Simulated Profitability (UNRELIABLE)

- Overall win rate: 57.75% (likely too high)
- Edge >=5 pts: 75.04% (definitely too high)
- Optimal edge: 6.5 points, 81.78% win rate (impossible)

### What These Numbers Might Mean in Reality

**Optimistic scenario:**
- True win rate: 54-56%
- ROI: 3-6%
- Profitable but volatile

**Realistic scenario:**
- True win rate: 51-53%
- ROI: 0-3%
- Marginal, needs improvements

**Pessimistic scenario:**
- True win rate: 48-50%
- ROI: -2% to 0%
- Not profitable, needs work

---

## Conclusion

**Phase 4 Week 1 Status:** INCONCLUSIVE

**What we know:**
1. Model has predictive power (MAE 6.10)
2. Can identify high-confidence predictions
3. Simulated lines suggest potential profitability

**What we DON'T know:**
1. True win rate against real sportsbooks
2. Actual ROI and risk metrics
3. Whether model is profitable

**Critical Next Step:**
ACQUIRE REAL BETTING LINES

**Without real lines, we cannot:**
- Validate profitability
- Optimize edge thresholds
- Make informed decision on deployment

**Recommended Action:**
1. Spend 3-7 days acquiring real historical betting lines
2. Re-run profitability analysis with real data
3. Make evidence-based decision on next phase

---

## Files Generated

- `scripts/data_collection/fetch_betting_lines.py` - Betting line simulator
- `scripts/analysis/analyze_betting_profitability.py` - Profitability calculator
- `data/results/predictions_with_real_betting_lines.csv` - Simulated lines
- `docs/validation/PHASE4_WEEK1_BETTING_LINES_ANALYSIS.md` - This document

---

**Final Note:** While we can't determine profitability from simulations, we DID learn that our model can identify cases with different risk/reward profiles. The selective betting approach (high edge thresholds) is valid conceptually - we just need real lines to know if it works in practice.

**Next:** Acquire real betting lines within 1 week, or pivot to model improvements.
