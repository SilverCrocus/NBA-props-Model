"""
NBA Props Model - Data Collection & Quality Analysis
Comprehensive analysis of CTG player data, team data, and game logs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class DataQualityAnalyzer:
    """Analyzes data collection completeness and quality for NBA props model"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.player_data_dir = self.base_dir / "data" / "ctg_data_organized" / "players"
        self.team_data_dir = self.base_dir / "data" / "ctg_team_data"
        self.game_logs_dir = self.base_dir / "data" / "game_logs"
        self.results = {}

    def analyze_player_data_structure(self):
        """Analyze player data directory structure and file organization"""
        print("\n" + "="*80)
        print("PLAYER DATA STRUCTURE ANALYSIS")
        print("="*80)

        seasons = sorted([d.name for d in self.player_data_dir.iterdir() if d.is_dir()])
        print(f"\nSeasons available: {len(seasons)}")
        print(f"Season range: {seasons[0]} to {seasons[-1]}")

        # Analyze file types by season
        file_structure = {}
        total_files = 0

        for season in seasons:
            season_dir = self.player_data_dir / season
            season_files = {}

            for split in ['regular_season', 'playoffs']:
                split_dir = season_dir / split
                if not split_dir.exists():
                    continue

                # Get all CSV files recursively
                csv_files = list(split_dir.rglob('*.csv'))
                season_files[split] = len(csv_files)
                total_files += len(csv_files)

            file_structure[season] = season_files

        print(f"\nTotal player data files: {total_files}")

        # Sample structure from most recent season
        recent_season = seasons[-1]
        print(f"\nFile structure for {recent_season}:")
        for split in ['regular_season', 'playoffs']:
            split_dir = self.player_data_dir / recent_season / split
            if split_dir.exists():
                print(f"\n  {split}:")
                for category_dir in sorted(split_dir.iterdir()):
                    if category_dir.is_dir():
                        csv_files = list(category_dir.glob('*.csv'))
                        if csv_files:
                            print(f"    - {category_dir.name}: {len(csv_files)} file(s)")

        self.results['player_data'] = {
            'total_files': total_files,
            'seasons': seasons,
            'file_structure': file_structure
        }

        return file_structure

    def analyze_data_categories(self):
        """Identify all unique data categories collected"""
        print("\n" + "="*80)
        print("DATA CATEGORIES AVAILABLE")
        print("="*80)

        categories = set()

        # Scan all seasons for unique categories
        for season_dir in self.player_data_dir.iterdir():
            if not season_dir.is_dir():
                continue
            for split in ['regular_season', 'playoffs']:
                split_dir = season_dir / split
                if not split_dir.exists():
                    continue

                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        # Handle on/off subdirectories
                        if category_dir.name == 'on_off':
                            for sub_cat in category_dir.iterdir():
                                if sub_cat.is_dir():
                                    categories.add(f"on_off/{sub_cat.name}")
                        else:
                            categories.add(category_dir.name)

        print(f"\nTotal unique data categories: {len(categories)}")
        print("\nCategories collected:")
        for i, cat in enumerate(sorted(categories), 1):
            print(f"  {i:2d}. {cat}")

        self.results['categories'] = sorted(categories)
        return categories

    def sample_data_schemas(self):
        """Examine schemas of different data types"""
        print("\n" + "="*80)
        print("DATA SCHEMA ANALYSIS")
        print("="*80)

        schemas = {}

        # Get most recent season
        recent_season = sorted([d.name for d in self.player_data_dir.iterdir() if d.is_dir()])[-1]
        season_dir = self.player_data_dir / recent_season / 'regular_season'

        # Sample key data categories
        key_categories = [
            'offensive_overview',
            'shooting_accuracy',
            'shooting_frequency',
            'defense_rebounding',
            'foul_drawing',
            'on_off/efficiency_four_factors',
            'on_off/team_shooting_frequency'
        ]

        for category in key_categories:
            category_path = season_dir / category
            csv_file = category_path / f"{category.split('/')[-1]}.csv"

            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, nrows=5)
                    schemas[category] = {
                        'columns': list(df.columns),
                        'num_columns': len(df.columns),
                        'sample_rows': len(df)
                    }

                    print(f"\n{category}:")
                    print(f"  Columns: {len(df.columns)}")
                    print(f"  Fields: {', '.join(df.columns[:10])}" +
                          (f", ... (+{len(df.columns)-10} more)" if len(df.columns) > 10 else ""))
                except Exception as e:
                    print(f"\n{category}: Error reading - {e}")

        self.results['schemas'] = schemas
        return schemas

    def analyze_data_quality(self):
        """Check data quality metrics across multiple files"""
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)

        quality_metrics = {
            'files_analyzed': 0,
            'total_rows': 0,
            'total_columns': 0,
            'missing_data': [],
            'data_issues': []
        }

        # Sample files from most recent season
        recent_season = sorted([d.name for d in self.player_data_dir.iterdir() if d.is_dir()])[-1]
        season_dir = self.player_data_dir / recent_season / 'regular_season'

        # Analyze several key files
        csv_files = list(season_dir.rglob('*.csv'))[:5]  # Sample first 5 files

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                quality_metrics['files_analyzed'] += 1
                quality_metrics['total_rows'] += len(df)
                quality_metrics['total_columns'] += len(df.columns)

                # Check for missing data
                missing_pct = (df.isnull().sum() / len(df) * 100)
                high_missing = missing_pct[missing_pct > 5]

                if len(high_missing) > 0:
                    quality_metrics['missing_data'].append({
                        'file': csv_file.name,
                        'columns_with_missing': dict(high_missing)
                    })

                # Check for duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    quality_metrics['data_issues'].append({
                        'file': csv_file.name,
                        'issue': f'{duplicates} duplicate rows'
                    })

            except Exception as e:
                quality_metrics['data_issues'].append({
                    'file': csv_file.name,
                    'issue': f'Read error: {str(e)}'
                })

        print(f"\nFiles analyzed: {quality_metrics['files_analyzed']}")
        print(f"Total rows sampled: {quality_metrics['total_rows']:,}")
        print(f"Average columns per file: {quality_metrics['total_columns'] // quality_metrics['files_analyzed']}")

        if quality_metrics['missing_data']:
            print(f"\nFiles with >5% missing data: {len(quality_metrics['missing_data'])}")
            for item in quality_metrics['missing_data'][:3]:
                print(f"  - {item['file']}")
        else:
            print("\nNo significant missing data detected in sampled files")

        if quality_metrics['data_issues']:
            print(f"\nData issues found: {len(quality_metrics['data_issues'])}")
            for issue in quality_metrics['data_issues'][:3]:
                print(f"  - {issue['file']}: {issue['issue']}")
        else:
            print("\nNo data quality issues detected")

        self.results['quality_metrics'] = quality_metrics
        return quality_metrics

    def analyze_team_data(self):
        """Analyze team data collection"""
        print("\n" + "="*80)
        print("TEAM DATA ANALYSIS")
        print("="*80)

        team_dirs = [d for d in self.team_data_dir.iterdir() if d.is_dir()]
        print(f"\nTeams collected: {len(team_dirs)}")

        # Analyze file structure
        if team_dirs:
            sample_team = team_dirs[0]
            csv_files = list(sample_team.glob('*.csv'))
            print(f"\nFiles per team: {len(csv_files)}")
            print("\nTeam data categories:")
            for csv_file in sorted(csv_files):
                df = pd.read_csv(csv_file, nrows=1)
                print(f"  - {csv_file.stem}: {len(df.columns)} columns")

        total_team_files = sum(len(list(d.glob('*.csv'))) for d in team_dirs)
        print(f"\nTotal team data files: {total_team_files}")

        self.results['team_data'] = {
            'teams': len(team_dirs),
            'total_files': total_team_files
        }

    def analyze_game_logs(self):
        """Analyze game logs data"""
        print("\n" + "="*80)
        print("GAME LOGS ANALYSIS")
        print("="*80)

        if not self.game_logs_dir.exists():
            print("\nGame logs directory not found")
            return

        game_log_files = list(self.game_logs_dir.glob('game_logs_*.csv'))
        print(f"\nGame log files: {len(game_log_files)}")

        if game_log_files:
            # Check combined file
            combined_file = self.game_logs_dir / 'all_game_logs_combined.csv'
            if combined_file.exists():
                df = pd.read_csv(combined_file)
                print(f"\nCombined game logs:")
                print(f"  Total games: {len(df):,}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Sample columns: {', '.join(df.columns[:8])}")

                if 'SEASON_YEAR' in df.columns:
                    seasons = df['SEASON_YEAR'].unique()
                    print(f"  Seasons covered: {len(seasons)} ({min(seasons)} to {max(seasons)})")

        self.results['game_logs'] = {
            'files': len(game_log_files),
            'combined_exists': (self.game_logs_dir / 'all_game_logs_combined.csv').exists()
        }

    def check_tracking_progress(self):
        """Analyze download progress tracking"""
        print("\n" + "="*80)
        print("DATA COLLECTION PROGRESS")
        print("="*80)

        tracking_file = self.player_data_dir.parent / 'tracking' / 'download_progress.json'

        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                progress = json.load(f)

            completed = len(progress.get('completed', []))
            total = progress.get('total_expected', 660)
            completion_rate = progress.get('completion_rate', '0%')
            last_updated = progress.get('last_updated', 'Unknown')

            print(f"\nDownload Progress:")
            print(f"  Completed: {completed}/{total} files")
            print(f"  Completion rate: {completion_rate}")
            print(f"  Last updated: {last_updated}")
            print(f"  Missing: {total - completed} files")

            self.results['progress'] = {
                'completed': completed,
                'total': total,
                'completion_rate': completion_rate,
                'missing': total - completed
            }
        else:
            print("\nTracking file not found")

    def identify_missing_data(self):
        """Identify specific gaps in data collection"""
        print("\n" + "="*80)
        print("MISSING DATA IDENTIFICATION")
        print("="*80)

        tracking_file = self.player_data_dir.parent / 'tracking' / 'download_progress.json'

        if not tracking_file.exists():
            print("\nCannot identify missing data - tracking file not found")
            return

        with open(tracking_file, 'r') as f:
            progress = json.load(f)

        # Expected structure: 22 seasons x 2 splits x 15 categories = 660 files
        expected_seasons = [f"{y:04d}-{(y+1)%100:02d}" for y in range(2003, 2025)]
        expected_splits = ['Regular Season', 'Playoffs']

        # Categories based on what we've seen
        expected_categories = [
            'Offensive Overview',
            'Shooting: Overall',
            'Shooting: Frequency',
            'Shooting: Accuracy',
            'Foul Drawing',
            'Defense and Rebounding',
            'Defense & Rebounding',
            'On/Off Efficiency & Four Factors',
            'On/Off Team Shooting: Frequency',
            'On/Off Team Shooting: Accuracy',
            'On/Off Team Halfcourt & Putbacks',
            'On/Off Team Transition',
            'On/Off Opponent Shooting: Frequency',
            'On/Off Opponent Shooting: Accuracy',
            'On/Off Opponent Halfcourt & Putbacks',
            'On/Off Opponent Transition'
        ]

        completed = set()
        for item in progress.get('completed', []):
            if len(item) == 3:
                completed.add(tuple(item))

        # Find missing combinations
        missing = []
        for season in expected_seasons[-5:]:  # Check last 5 seasons
            for split in expected_splits:
                for category in expected_categories:
                    if (category, season, split) not in completed:
                        missing.append((season, split, category))

        if missing:
            print(f"\nRecent missing files (last 5 seasons): {len(missing)}")
            print("\nSample missing entries:")
            for item in missing[:10]:
                print(f"  - {item[0]} {item[1]}: {item[2]}")
        else:
            print("\nNo missing files detected in recent seasons")

        self.results['missing_files'] = missing[:20]  # Store sample

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)

        print("\n### DATA INVENTORY")
        print(f"  Player data files: {self.results.get('player_data', {}).get('total_files', 0)}")
        print(f"  Team data files: {self.results.get('team_data', {}).get('total_files', 0)}")
        print(f"  Data categories: {len(self.results.get('categories', []))}")
        print(f"  Seasons covered: {len(self.results.get('player_data', {}).get('seasons', []))}")

        print("\n### COLLECTION STATUS")
        progress = self.results.get('progress', {})
        print(f"  Completion: {progress.get('completed', 0)}/{progress.get('total', 660)} ({progress.get('completion_rate', 'N/A')})")
        print(f"  Missing files: {progress.get('missing', 0)}")

        print("\n### DATA QUALITY")
        quality = self.results.get('quality_metrics', {})
        print(f"  Files analyzed: {quality.get('files_analyzed', 0)}")
        print(f"  Data issues: {len(quality.get('data_issues', []))}")
        print(f"  Missing data warnings: {len(quality.get('missing_data', []))}")

        print("\n### READINESS FOR MODELING")

        readiness_score = 0
        max_score = 4

        # Check completeness
        if progress.get('completion_rate', '0%').replace('%', '') >= '90':
            print("  ✓ Data collection >90% complete")
            readiness_score += 1
        else:
            print("  ✗ Data collection incomplete")

        # Check variety
        if len(self.results.get('categories', [])) >= 10:
            print("  ✓ Sufficient data categories collected")
            readiness_score += 1
        else:
            print("  ✗ Insufficient data variety")

        # Check quality
        if len(quality.get('data_issues', [])) == 0:
            print("  ✓ No major data quality issues")
            readiness_score += 1
        else:
            print("  ✗ Data quality issues detected")

        # Check game logs
        if self.results.get('game_logs', {}).get('combined_exists'):
            print("  ✓ Game logs available")
            readiness_score += 1
        else:
            print("  ✗ Game logs incomplete")

        print(f"\nOverall Readiness: {readiness_score}/{max_score}")

        if readiness_score >= 3:
            print("Status: READY for feature engineering and modeling")
        elif readiness_score >= 2:
            print("Status: MOSTLY READY - address minor issues")
        else:
            print("Status: NOT READY - complete data collection first")

        print("\n### RECOMMENDATIONS")

        if progress.get('missing', 0) > 0:
            print(f"\n1. Complete data collection")
            print(f"   - {progress.get('missing', 0)} files remaining")
            print(f"   - Run: uv run ctg_robust_scraper.py")

        if len(quality.get('data_issues', [])) > 0:
            print(f"\n2. Address data quality issues")
            print(f"   - Review and clean problematic files")

        print(f"\n3. Next steps for modeling:")
        print(f"   - Develop feature engineering pipeline")
        print(f"   - Merge player stats with game logs")
        print(f"   - Create PRA (Points + Rebounds + Assists) target variable")
        print(f"   - Build train/test split by season")

        # Save results to file
        output_file = self.base_dir / 'data_quality_report.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n### Detailed report saved to: {output_file}")


def main():
    """Run comprehensive data analysis"""
    base_dir = '/Users/diyagamah/Documents/nba_props_model'

    analyzer = DataQualityAnalyzer(base_dir)

    # Run all analyses
    analyzer.analyze_player_data_structure()
    analyzer.analyze_data_categories()
    analyzer.sample_data_schemas()
    analyzer.analyze_data_quality()
    analyzer.analyze_team_data()
    analyzer.analyze_game_logs()
    analyzer.check_tracking_progress()
    analyzer.identify_missing_data()
    analyzer.generate_summary_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
