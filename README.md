# 🏀 NBA Props Model - Advanced PRA Prediction System
*Machine learning system for NBA player prop predictions using premium analytics from CleaningTheGlass.com*

[![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)](https://www.python.org/downloads/)
[![Data Source](https://img.shields.io/badge/data-CleaningTheGlass-orange.svg)](https://cleaningtheglass.com/)
[![Coverage](https://img.shields.io/badge/seasons-22_years-green.svg)]()
[![Progress](https://img.shields.io/badge/data_collection-93%25_complete-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/target-PRA_props-purple.svg)]()

## 🎯 Project Overview

Advanced machine learning pipeline for predicting NBA player prop bets, specifically **PRA (Points + Rebounds + Assists)** combinations. Leverages premium analytics from CleaningTheGlass.com with sophisticated feature engineering to identify value in betting markets.

### Why PRA Props?
- **Reduced Variance**: Combining three categories smooths individual performance volatility
- **Market Inefficiency**: Combo props often present better value than individual stat bets  
- **Predictable Patterns**: Three-tier feature architecture captures player performance reliably

### Key Features
- 🔧 **Robust Web Scraping**: Selenium-based with session persistence and error recovery
- 📊 **Premium Data**: CleaningTheGlass advanced metrics + NBA API temporal features
- 🧠 **Three-Tier Features**: Core performance + contextual modulators + temporal dynamics
- 📈 **Comprehensive Coverage**: 22 seasons of data (2003-04 to 2024-25)
- 🔄 **Resume Capability**: Automatic progress tracking and seamless resumption

## 📊 Current Status

### Data Collection Progress
| Component | Status | Progress | Notes |
|-----------|---------|----------|--------|
| CTG Player Data | 🟢 Active | 614/660 (93%) | 46 files remaining |
| CTG Team Data | ✅ Complete | 270/270 (100%) | All 30 teams |
| Data Organization | ✅ Complete | 100% | Hierarchical structure |
| Session Management | ✅ Complete | 100% | Persistent login |
| Feature Engineering | 🟡 Planned | 0% | Architecture designed |
| Model Development | ⭕ Pending | 0% | Awaiting complete dataset |

### Dataset Statistics
- **Player Files**: 614 CSV files across 15 categories
- **Team Files**: 270 files (30 teams × 9 categories)  
- **Total Records**: ~290,000+ player-season combinations
- **Storage**: 2.9GB organized data

## 🏗️ Technical Architecture

### Data Collection Pipeline
```
CleaningTheGlass.com (Premium Analytics)
    ↓ [Selenium + Chrome Profile Persistence]
    ↓ [Session Management + Error Recovery]
    ↓ [Progress Tracking + Auto-Resume]
    ↓
Local Storage (887 CSV files)
    ↓ [Three-Tier Feature Engineering]
    ↓
NBA API (Contextual Features)
    ↓ [Data Fusion]
    ↓
ML Model → PRA Predictions
```

### Data Sources & Coverage

#### CleaningTheGlass.com - Primary Source
**Categories Scraped (15 total):**
- Player offensive metrics (usage, scoring efficiency, shot selection)
- Defense and rebounding rates
- On/Off court impact statistics  
- Team shooting and efficiency when player is on/off court

**Temporal Coverage:**
- 22 seasons (2003-04 through 2024-25)
- Regular Season + Playoffs
- 660 total combinations possible

#### NBA API - Supplementary Source  
- Team pace and efficiency ratings
- Defensive matchup statistics
- Opponent PRA allowed by position
- Rest days and scheduling factors

## ⚙️ Three-Tier Feature Engineering

### 🎯 Tier 1: Core Performance Engine
*Player's fundamental basketball abilities*

- **Usage Rate (USG%)**: Volume predictor for touches and shots
- **Points Per Shot Attempt (PSA)**: Scoring efficiency metric
- **Assist Rate & AST:Usage**: Playmaking relative to offensive load
- **Rebounding (fgOR%, fgDR%)**: Opportunity-adjusted rebounding
- **Advanced Metrics**: PER, Win Shares, Box Plus/Minus

### 🎮 Tier 2: Contextual Modulators  
*Game-specific environmental factors*

- **Minutes Projection**: L5 game average opportunity
- **Opponent Defense**: Position-specific defensive ratings
- **Pace Factors**: Team/opponent pace differential
- **Situational Context**: Rest, back-to-backs, home/away
- **Injury Impact**: On/Off usage delta for opportunity shifts

### 📈 Tier 3: Temporal Dynamics
*Recent performance and trends*

- **Rolling Averages**: 5/10/15 game windows  
- **EWMA**: Exponentially weighted recent performance
- **Volatility Measures**: Consistency and variance metrics
- **Trend Detection**: Hot/cold streak identification

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12+
- Google Chrome browser
- CleaningTheGlass.com premium subscription

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/nba_props_model.git
cd nba_props_model

# Install dependencies with uv (recommended)
uv sync

# Alternative: pip installation
pip install -r requirements.txt
```

### Data Collection Setup
```bash
# Resume data collection (automatic progress detection)
python ctg_robust_scraper.py

# Check collection status
python ctg_file_manager.py --status

# Run with specific parameters
python ctg_robust_scraper.py --season 2024-25 --type regular
```

## 📖 Usage Examples

### Loading Player Data
```python
import pandas as pd
from pathlib import Path

# Load player advanced stats
data_path = Path("data/ctg_data_organized/players/2024-25/regular_season")
offensive_data = pd.read_csv(data_path / "offensive_overview.csv")

# Get high usage players
top_usage = offensive_data.nlargest(20, 'USG%')
print(f"Top usage players:\n{top_usage[['Player', 'Team', 'USG%']]}")
```

### Team Matchup Analysis
```python
# Load team pace data
team_data = pd.read_csv("data/team_data/pace/team_advanced_2023-24.csv")

# Identify fast-paced matchups
fast_teams = team_data[team_data['PACE'] > 100]
print(f"Fast-paced teams: {fast_teams['TEAM'].tolist()}")
```

## 📂 Project Structure
```
nba_props_model/
├── data/
│   ├── ctg_data_organized/       # Player stats (614 files)
│   │   └── players/
│   │       ├── 2024-25/
│   │       │   ├── regular_season/
│   │       │   └── playoffs/
│   │       └── [21 more seasons]
│   ├── ctg_team_data/            # Team stats (270 files)
│   └── chrome_profile_ctg/       # Browser session
├── ctg_robust_scraper.py         # Main scraper with resume
├── team_data_collector_simple.py # NBA API collector
├── ctg_file_manager.py          # Data organization
├── features_plan.md             # Feature documentation
└── pyproject.toml              # Dependencies
```

## 🔧 Technical Implementation

### Web Scraping Architecture
- **Session Persistence**: Chrome profile maintains login across runs
- **Error Recovery**: Automatic retry with exponential backoff
- **Rate Limiting**: Respectful 2-4 second delays
- **Progress Tracking**: JSON-based resume capability
- **Memory Management**: Auto-restart every 40 files

### Data Quality Assurance
- **Completeness Checks**: Automated file count validation
- **Schema Validation**: Consistent columns across seasons
- **Duplicate Detection**: Hash-based integrity checks
- **Missing Data**: Graceful handling of incomplete seasons

## 🗺️ Development Roadmap

### Phase 1: Data Foundation ✅
- [x] CleaningTheGlass integration
- [x] Robust scraping infrastructure  
- [x] Data organization system
- [x] Session management

### Phase 2: Feature Engineering 🔄 (Current)
- [ ] Complete CTG collection (46 files remaining)
- [ ] NBA API temporal features
- [ ] Three-tier pipeline implementation
- [ ] Feature validation

### Phase 3: Model Development 📋
- [ ] Baseline models (Linear, Random Forest)
- [ ] Advanced models (XGBoost, Neural Networks)
- [ ] Hyperparameter optimization
- [ ] Feature importance analysis

### Phase 4: Production 🚀
- [ ] Real-time predictions
- [ ] Backtesting framework
- [ ] Performance monitoring
- [ ] Web interface

### Future Enhancements 🔮
- Multi-target models (individual props)
- Live in-game predictions
- Injury report integration
- Odds movement tracking
- Arbitrage detection

## 📈 Performance Metrics

### Data Collection
- **Scraping Speed**: ~40 files per session
- **Session Duration**: 2-3 hours continuous
- **Error Rate**: <2% with retry logic
- **Storage**: 2.9GB compressed

### Model Targets
*To be updated upon completion*
- **Baseline**: Beat closing line
- **Target ROI**: >5% on closing odds
- **Confidence**: 95% intervals
- **Updates**: Daily predictions

## 🤝 Contributing

### Development Setup
```bash
# Install dev dependencies
uv sync --dev

# Run tests
python -m pytest tests/

# Format code
black ctg_*.py
```

### Code Standards
- Type hints for all functions
- Comprehensive docstrings
- Robust error handling
- Detailed logging

## ⚖️ Legal Disclaimer

**Educational Purpose Only**: This project is for educational and research purposes in sports analytics and machine learning.

**Betting Risks**: Sports betting involves substantial risk. Past performance doesn't guarantee future results. Never bet more than you can afford to lose.

**Data Usage**: Ensure compliance with CleaningTheGlass.com terms of service and NBA data policies.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CleaningTheGlass.com**: Premium NBA analytics platform
- **NBA.com**: Official statistics and game data
- **Open Source Community**: Tools and libraries enabling this project

---

*Built with dedication to the intersection of data science and basketball analytics*