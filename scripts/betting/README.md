# Betting Tracking Scripts

Track your NBA prop bets from recommendation to settlement and monitor performance over time.

## Overview

This directory contains scripts to:
1. **Record bets** you actually place from daily recommendations
2. **Update results** after games finish using NBA API
3. **Track performance** with detailed analytics and reporting

## Quick Commands

### Simple Interface (Recommended)
```bash
# Record top 5 bets
uv run track_bet.py record 2025-10-21 --top 5

# Update results after games finish
uv run track_bet.py update 2025-10-21

# View performance dashboard
uv run track_bet.py status --detailed
```

### Individual Scripts

#### 1. record_bets.py
Records which bets you actually placed.

```bash
# Record all recommended bets
uv run python scripts/betting/record_bets.py 2025-10-21 --all

# Record top 5 bets
uv run python scripts/betting/record_bets.py 2025-10-21 --top 5

# Interactive selection
uv run python scripts/betting/record_bets.py 2025-10-21

# Custom stakes (if different from recommended)
uv run python scripts/betting/record_bets.py 2025-10-21 --top 5 --stakes 10,15,20,25,30
```

#### 2. update_results.py
Fetches actual game results and calculates bet outcomes.

```bash
# Update results for specific date
uv run python scripts/betting/update_results.py 2025-10-21

# Update all pending bets
uv run python scripts/betting/update_results.py --all-pending

# Force refresh existing results
uv run python scripts/betting/update_results.py 2025-10-21 --force

# Show summary only
uv run python scripts/betting/update_results.py --summary
```

#### 3. betting_dashboard.py
View performance analytics and track ROI.

```bash
# Overall summary
uv run python scripts/betting/betting_dashboard.py

# Detailed breakdown by confidence, edge, bookmaker
uv run python scripts/betting/betting_dashboard.py --detailed

# Performance over time
uv run python scripts/betting/betting_dashboard.py --chronological

# Export to CSV
uv run python scripts/betting/betting_dashboard.py --export
```

## Workflow Example

```bash
# Monday: Get recommendations
uv run nba_today.py 2025-10-21 1000

# Monday: Record bets you placed (e.g., top 5)
uv run track_bet.py record 2025-10-21 --top 5

# Tuesday: Update results after games finish
uv run track_bet.py update 2025-10-21

# Tuesday: Check performance
uv run track_bet.py status --detailed
```

## Data Storage

### Bet Ledger
`/data/betting/bet_ledger.csv` - Main database of all bets

**DO NOT DELETE THIS FILE** - it contains your complete betting history

### Recommendations
`/data/betting/recommendations_YYYY-MM-DD.csv` - Daily recommendations from nba_today.py

## Features

### Performance Tracking
- Win/Loss/Push record
- Win rate %
- Total profit/loss
- ROI %
- Average bet size

### Breakdowns
- By confidence level (VERY HIGH, HIGH, MEDIUM, LOW)
- By bet type (OVER vs UNDER)
- By edge size (0-10%, 10-20%, 20-30%, 30%+)
- By bookmaker
- Chronological performance

### Analytics
- Best/worst individual bets
- Performance trends over time
- Cumulative P/L tracking
- Export capabilities for external analysis

## Tips

1. **Record bets immediately** after placing them to maintain accurate records
2. **Update results the next day** after games finish (NBA API needs time to update)
3. **Review dashboard weekly** to identify performance patterns
4. **Compare to backtest** - expected ROI is ~0.91% based on historical validation

## Troubleshooting

**"No recommendations found"**
- Run `uv run nba_today.py [date]` first to generate recommendations

**"Player not found" during update**
- Player may not have played that game
- NBA API may use different name format (e.g., "Jimmy Butler III" vs "Jimmy Butler")
- Wait a few hours after games end for API to update

**Results still PENDING**
- Games may not have finished yet
- Run update script again later
- Check NBA API is accessible

## See Also

- `BETTING_TRACKER_GUIDE.md` - Complete guide at project root
- `nba_today.py` - Daily recommendations command
- `scripts/production/daily_betting_recommendations.py` - Underlying engine
