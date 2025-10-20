# Production Usage Guide

This guide explains how to use the validated, profitable betting model in production.

---

## Quick Start

### 1. Train Production Model

```bash
uv run python scripts/production/train_production_model.py
```

**What it does:**
- Trains XGBoost model on ALL available data (2003 through latest 2024-25)
- Uses 2023-24 as calibration validation set
- Applies isotonic regression calibration
- Saves both uncalibrated and calibrated models

**Output:**
- `models/production_model_v2.0_PRODUCTION_latest.pkl` (uncalibrated)
- `models/production_model_v2.0_PRODUCTION_CALIBRATED_latest.pkl` (calibrated)

**When to run:**
- First time setup
- Monthly (to incorporate new game data)
- After significant model improvements

---

### 2. Get Daily Betting Recommendations

```bash
# Default: today's games, moderate strategy
uv run python scripts/production/daily_betting_recommendations.py

# Specific date
uv run python scripts/production/daily_betting_recommendations.py --date 2025-10-25

# Conservative strategy
uv run python scripts/production/daily_betting_recommendations.py --strategy conservative

# With custom bankroll
uv run python scripts/production/daily_betting_recommendations.py --bankroll 5000

# Save HTML report
uv run python scripts/production/daily_betting_recommendations.py --save-html
```

**What it does:**
- Fetches upcoming NBA games
- Gets prop odds from The Odds API
- Makes predictions using production model
- Calculates calibrated edges
- Recommends bets with confidence levels
- Outputs bet sizing based on Kelly Criterion

**Output:**
- Console: Top 10 betting recommendations
- CSV: `data/betting/recommendations_YYYY_MM_DD.csv`
- HTML: `data/betting/recommendations_YYYY_MM_DD.html` (optional)

---

## Configuration

### Odds API Setup

The daily recommendations script uses [The Odds API](https://the-odds-api.com/) to fetch live betting odds.

**Setup:**
1. Sign up for free API key at https://the-odds-api.com/
2. Set your API key in the script or as environment variable:

```python
# Option 1: Edit scripts/production/daily_betting_recommendations.py
ODDS_API_KEY = "your_api_key_here"

# Option 2: Environment variable
export ODDS_API_KEY="your_api_key_here"
```

**Free tier limits:**
- 500 requests/month
- Sufficient for daily recommendations (1 request/day = 30/month)

---

### Strategy Selection

Four pre-configured strategies based on Monte Carlo simulation results:

| Strategy | Kelly Fraction | Edge Threshold | Max Bet | Risk Level |
|----------|---------------|----------------|---------|------------|
| **Conservative** | 0.10 (10%) | ≥7% | 2% bankroll | ⭐ Low |
| **Moderate** | 0.20 (20%) | ≥5% | 5% bankroll | ⭐⭐ Medium |
| **Aggressive** | 0.25 (25%) | ≥5% | 10% bankroll | ⭐⭐⭐ High |
| **Maximum** | 0.35 (35%) | ≥5% | 15% bankroll | ⭐⭐⭐⭐ Extreme |

**Recommended:** Start with **moderate** strategy ($1,000-5,000 bankroll)

See `BETTING_STRATEGY_IMPLEMENTATION.md` for detailed strategy descriptions.

---

## Understanding Recommendations

### Sample Output

```
1. LeBron James - OVER 38.5
   Bookmaker: DraftKings
   Prediction: 42.3 PRA
   Edge: 8.2% (HIGH CONFIDENCE)
   Odds: -110 (Decimal: 1.91)
   Win Probability: 58.7%
   💵 Recommended Bet: $32.80
```

**Key Fields:**
- **Edge:** Expected value advantage (higher = better)
- **Confidence:** Model certainty (VERY HIGH, HIGH, MEDIUM, LOW)
- **Win Probability:** Calibrated probability of winning this bet
- **Recommended Bet:** Kelly criterion optimal bet size

---

## Confidence Levels Explained

**VERY HIGH (75-100 points):**
- Large edge (>8%)
- High probability (>60% or <40%)
- Strongest bets, highest priority

**HIGH (60-75 points):**
- Good edge (6-8%)
- Moderate probability (55-60% or 40-45%)
- Strong bets, recommended

**MEDIUM (40-60 points):**
- Decent edge (5-6%)
- Near 50/50 probability
- Acceptable but lower priority

**LOW (<40 points):**
- Small edge (5-6%)
- Close to 50/50 probability
- Marginal bets, consider skipping

---

## Daily Workflow

### Step 1: Morning Check (Game Day)

```bash
# Check if there are games today
uv run python scripts/production/daily_betting_recommendations.py
```

**If no games:**
- Script will notify you
- No action needed

**If games found:**
- Review recommendations
- Note top 5-10 bets with VERY HIGH or HIGH confidence

---

### Step 2: Place Bets

For each recommended bet:

1. **Verify odds are still available**
   - Lines can move (check bookmaker before betting)
   - If line moved significantly, recalculate edge

2. **Place bet with recommended size**
   - Use exact bet size from recommendations
   - Or adjust based on available line

3. **Track bet in spreadsheet**
   - Date, player, line, direction, odds, bet size
   - For later performance analysis

---

### Step 3: After Games Complete

```bash
# Update tracking spreadsheet with results
# Calculate daily P&L
# Update bankroll for tomorrow's recommendations
```

---

### Step 4: Weekly Review

Every Sunday:
- Calculate weekly win rate (target: 51-52%)
- Calculate weekly ROI (target: 3%+)
- Check if max drawdown exceeded expectations
- Adjust strategy if needed (more/less aggressive)

---

### Step 5: Monthly Retrain

First of each month:
```bash
# Retrain model with latest data
uv run python scripts/production/train_production_model.py

# Verify MAE is still <7 points
# Check calibration quality (Brier score <0.25)
```

---

## Performance Tracking

### Key Metrics to Track

**Daily:**
- Bets placed
- Bets won/lost
- P&L

**Weekly:**
- Win rate (target: 51-52%)
- ROI per bet (target: 3%+)
- Bankroll change

**Monthly:**
- Total return (target: +20-40% per month)
- Max drawdown
- Compare to Monte Carlo expectations

---

### Performance Dashboard Template

```
Week of: 2025-10-20

Bets Placed: 47
Wins: 25 (53.2%)
Losses: 22 (46.8%)

Total Wagered: $8,234
Total Profit: +$287
ROI: +3.5%

Bankroll:
  Starting: $2,000
  Ending: $2,287
  Return: +14.4% (this week)

Max Drawdown: -$342 (-17.1%)

Status: ✅ ON TARGET
```

---

## Troubleshooting

### Issue: No recommendations generated

**Possible causes:**
1. No games scheduled for target date
2. Odds API not returning data
3. Edge threshold too high

**Solutions:**
- Check NBA schedule
- Verify API key is valid
- Lower edge threshold (use `--strategy aggressive`)

---

### Issue: Win rate below 48%

**Red flag!** Model may be degrading.

**Actions:**
1. Stop betting immediately
2. Review recent losses (post-mortem)
3. Check if bookmakers adjusted lines
4. Retrain model with latest data
5. Consider model improvements

---

### Issue: Predictions seem wrong

**Checks:**
1. Verify historical data is up to date
2. Check for injuries/lineup changes (not in model)
3. Verify calibration quality
4. Compare predictions to actual results

**Model doesn't account for:**
- Injuries (use injury reports)
- Lineup changes (check starting lineups)
- Rest/load management (partially captured via rest days)
- Trades (need to retrain)

---

## Risk Management

### Stop Loss Rules

**Trigger stop loss if:**
1. Bankroll drops to 50% of starting (conservative)
2. Bankroll drops to 30% of starting (moderate)
3. Bankroll drops to 20% of starting (aggressive)

**When stop loss triggered:**
- Pause betting for 1 week
- Review recent losses
- Retrain model
- Consider lowering Kelly fraction
- Restart with reduced strategy (aggressive → moderate)

---

### Warning Signs

**🚨 STOP IMMEDIATELY IF:**
- Win rate <48% after 100+ bets
- Consecutive losing streak >10 games
- Bankroll below stop loss threshold
- Betting emotionally (tilt, revenge betting)

**⚠️ REASSESS IF:**
- Win rate 48-50% after 200+ bets
- Max drawdown exceeds expected range
- ROI consistently below 2%
- Large unexpected losses

---

## Expected Performance

### Based on Monte Carlo Simulation (10,000 iterations)

**Conservative Strategy:**
- Expected annual return: +50-150%
- Max drawdown: -30 to -50%
- Bust risk: 0%

**Moderate Strategy (Recommended):**
- Expected annual return: +150-300%
- Max drawdown: -50 to -70%
- Bust risk: 0%

**Aggressive Strategy:**
- Expected annual return: +250-500%
- Max drawdown: -70 to -90%
- Bust risk: 0%

**Key Insight:** 99.9% of simulations are profitable. Median return is +353% per year.

---

## FAQ

### Q: How often should I retrain the model?

**A:** Monthly is recommended. More frequent retraining can lead to overfitting on small samples.

---

### Q: What if the line moves after recommendations are generated?

**A:** Recalculate edge with new odds. If edge still positive, bet stands. If edge turns negative, skip the bet.

---

### Q: Should I bet every recommended opportunity?

**A:** No. Prioritize VERY HIGH and HIGH confidence bets. Skip LOW confidence unless volume is needed.

---

### Q: What if I can't find the recommended line?

**A:** Shop other bookmakers. Line may vary slightly. Recalculate edge with available line.

---

### Q: How much should I start with?

**A:** Depends on strategy:
- Conservative: $500-1,000
- Moderate: $1,000-5,000
- Aggressive: $5,000-10,000

Start with an amount you can afford to lose entirely.

---

### Q: What's the time commitment?

**A:**
- Daily: 10-15 minutes (check recommendations, place bets)
- Weekly: 30 minutes (performance review)
- Monthly: 2 hours (retrain model, deep analysis)

---

### Q: When can I expect profits?

**A:** Variance is high in the short term. Expect profitability after 100+ bets (4-8 weeks of daily betting).

---

## Success Checklist

✅ **Before Starting:**
- [ ] Production model trained
- [ ] Odds API configured
- [ ] Strategy selected
- [ ] Bankroll determined
- [ ] Tracking spreadsheet ready

✅ **Daily:**
- [ ] Check recommendations
- [ ] Verify odds availability
- [ ] Place bets with correct sizing
- [ ] Track bets in spreadsheet

✅ **Weekly:**
- [ ] Calculate win rate
- [ ] Calculate ROI
- [ ] Check max drawdown
- [ ] Adjust if needed

✅ **Monthly:**
- [ ] Retrain model
- [ ] Deep performance analysis
- [ ] Compare to Monte Carlo expectations
- [ ] Adjust strategy if needed

---

## Support & Resources

**Documentation:**
- `MONTE_CARLO_ANALYSIS.md` - Skill vs luck validation
- `BETTING_STRATEGY_IMPLEMENTATION.md` - Detailed strategy guide
- `CALIBRATION_SUCCESS.md` - Model calibration results
- `CLAUDE.md` - Project architecture

**Key Scripts:**
- `scripts/production/train_production_model.py` - Model training
- `scripts/production/daily_betting_recommendations.py` - Daily recommendations
- `scripts/analysis/monte_carlo_backtest_simulation.py` - Simulation analysis

**Contact:**
- GitHub: https://github.com/SilverCrocus/NBA-props-Model
- Issues: https://github.com/SilverCrocus/NBA-props-Model/issues

---

## Legal Disclaimer

**Sports betting involves risk. Past performance does not guarantee future results.**

This model is provided for educational and research purposes. Use at your own risk. Only bet with money you can afford to lose. Gambling can be addictive - seek help if needed.

The model has been validated through Monte Carlo simulation (99.9% profitability, +353% median annual return), but real-world results may vary due to:
- Market efficiency (bookmakers may adjust)
- Model degradation over time
- Injuries and lineup changes not captured
- Execution risk (line movement, betting limits)

Always practice responsible gambling.

---

**Status:** ✅ READY FOR PRODUCTION DEPLOYMENT

**Last Updated:** October 21, 2025

**Model Version:** v2.0_PRODUCTION_CALIBRATED
