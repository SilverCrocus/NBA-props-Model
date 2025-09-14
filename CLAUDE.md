# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Props Model - Machine learning system for NBA player prop predictions (PRA - Points + Rebounds + Assists) using CleaningTheGlass.com premium analytics data.

## Package Management

This project uses `uv` for package management:
- **Install packages**: `uv add <package-name>`
- **Install all dependencies**: `uv sync`
- **Install with dev dependencies**: `uv sync --dev`
- **Run Python files**: `uv run <filename.py>`

## Common Development Commands

### Running the Main Scrapers
```bash
# Resume CTG data collection (auto-detects progress)
uv run ctg_robust_scraper.py

# Run team data collection
uv run team_data_collector_simple.py

# Check collection status
uv run ctg_file_manager.py --status
```

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/unit/test_features.py
```

### Code Quality
```bash
# Format code with black
uv run black src/ scripts/

# Lint with flake8
uv run flake8 src/

# Type checking with mypy
uv run mypy src/
```

## Project Architecture

### Data Collection Pipeline
- **CTG Scraper** (`ctg_robust_scraper.py`): Selenium-based scraper with session persistence, auto-resume capability, and Chrome profile management
- **Team Data Collector** (`team_data_collector_simple.py`): NBA API integration for supplementary team stats
- **File Manager** (`ctg_file_manager.py`): Data organization and progress tracking

### Key Directories
- `data/ctg_data_organized/`: 614 player stat CSV files (93% complete)
- `data/ctg_team_data/`: 270 team stat files (100% complete)
- `src/data/scrapers/`: Web scraping implementations
- `src/features/`: Three-tier feature engineering (Core Performance, Contextual Modulators, Temporal Dynamics)
- `src/models/`: ML model implementations (XGBoost, LightGBM planned)
- `notebooks/`: Jupyter notebooks for analysis and experimentation

### Data Processing Flow
1. **Collection**: CleaningTheGlass.com → Selenium scraper → CSV files
2. **Organization**: Raw data → Structured directories by season/type
3. **Feature Engineering**: Three-tier feature extraction pipeline
4. **Modeling**: PRA prediction using ensemble methods

## Session Management & Error Recovery

The CTG scraper includes robust session management:
- Chrome profile persistence for maintaining login
- Automatic restart every 40 files to prevent memory issues
- JSON-based progress tracking for seamless resume
- Exponential backoff for rate limiting
- Automatic Chrome process cleanup

## Important Implementation Notes

### Web Scraping
- Uses persistent Chrome profile at `~/.ctg_chrome_profile`
- 2-4 second delays between requests for respectful scraping
- Progress tracked in `data/ctg_data_organized/tracking/download_progress.json`

### Feature Engineering Tiers
1. **Core Performance**: Usage rate, scoring efficiency, rebounding rates
2. **Contextual Modulators**: Minutes projection, opponent defense, pace factors
3. **Temporal Dynamics**: Rolling averages, EWMA, volatility measures

### Current Status
- Player data: 614/660 files (93% complete, 46 remaining)
- Team data: 270/270 files (100% complete)
- Data spans 22 seasons (2003-04 to 2024-25)