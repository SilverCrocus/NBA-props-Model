# NBA Props Model - PRA Prediction

## Project Overview
Building a player props model to predict PRA (Points + Rebounds + Assists) using data from CleaningTheGlass.com.

## ✅ SOLUTION CONFIRMED WORKING!

### Page Structure (Confirmed via Inspection)
- **Select #0**: Category dropdown (15 options)
- **Select #1**: Season dropdown (22 options) 
- **Select #2**: Season Type dropdown (2 options)
- **Download button**: `<a class='download_button'>Download this table</a>`

## Data Collection

### CTG Automated Downloader
Use `ctg_downloader_final.py` to automatically download all player data.

**Usage:**
```bash
python ctg_downloader_final.py
```

Choose:
1. **Test mode** - Downloads 2 files to verify setup
2. **Full download** - Downloads all 660 files (~2 hours)

**Features:**
- Uses correct select indices (0=Category, 1=Season, 2=Type)
- Clicks the download button for each combination
- Handles file organization and renaming
- Shows progress and time estimates
- Robust error handling

**Output Structure:**
```
data/ctg_data/
├── offensive_overview_2024_25_regular_season.csv
├── offensive_overview_2024_25_playoffs.csv
├── shooting_overall_2023_24_regular_season.csv
└── ... (657 more files)
```

## Model Features (Planned)

### Three-Tier Feature Architecture:
1. **Core Performance Engine** - Usage%, PSA, AST%, Rebounding
2. **Contextual Modulators** - Minutes, Opponent, Pace, Rest
3. **Temporal Dynamics** - Rolling averages, EWMA, Volatility

See `features_plan.md` for complete feature strategy.

## Files in Project
- `ctg_downloader_final.py` - **THE WORKING DOWNLOADER**
- `features_plan.md` - PRA model feature specifications
- `data/ctg_data/` - Downloaded CSV files (after running scraper)

## Next Steps
1. ✅ Set up project structure
2. ✅ Create automated scraper
3. ✅ Fix element detection issues
4. ⏳ Run full data collection (660 files)
5. ⏳ Build feature engineering pipeline
6. ⏳ Develop PRA prediction model