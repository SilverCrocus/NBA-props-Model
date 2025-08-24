#!/usr/bin/env python3
"""
CTG File Manager - Utility for organized file saving
Provides standardized path generation and file management for CTG data
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import os


class CTGFileManager:
    """Manages file paths and operations for organized CTG data structure"""
    
    def __init__(self, base_dir="/Users/diyagamah/Documents/nba_props_model/data/ctg_data_organized"):
        self.base_dir = Path(base_dir)
        
        # Stat type mappings - maps scraper names to folder structure
        self.stat_mappings = {
            # Main categories
            'defense_and_rebounding': 'defense_rebounding',
            'foul_drawing': 'foul_drawing',
            'offensive_overview': 'offensive_overview',
            'shooting_accuracy': 'shooting_accuracy',
            'shooting_frequency': 'shooting_frequency',
            'shooting_overall': 'shooting_overall',
            
            # On/Off subcategories
            'on_off_efficiency_four_factors': 'on_off/efficiency_four_factors',
            'on_off_opponent_halfcourt_putbacks': 'on_off/opponent_halfcourt_putbacks',
            'on_off_opponent_shooting_accuracy': 'on_off/opponent_shooting_accuracy',
            'on_off_opponent_shooting_frequency': 'on_off/opponent_shooting_frequency',
            'on_off_opponent_transition': 'on_off/opponent_transition',
            'on_off_team_halfcourt_putbacks': 'on_off/team_halfcourt_putbacks',
            'on_off_team_shooting_accuracy': 'on_off/team_shooting_accuracy',
            'on_off_team_shooting_frequency': 'on_off/team_shooting_frequency',
            'on_off_team_transition': 'on_off/team_transition'
        }
    
    def get_file_path(self, stat_type, season, season_type='regular_season', entity_type='players'):
        """
        Generate standardized file path for CTG data
        
        Args:
            stat_type: Type of statistics (e.g., 'defense_and_rebounding')
            season: Season string (e.g., '2024-25')
            season_type: 'regular_season' or 'playoffs'
            entity_type: 'players' or 'teams'
        
        Returns:
            Path object for the file
        """
        # Normalize inputs
        season = season.replace('_', '-')  # Convert 2024_25 to 2024-25
        
        if stat_type not in self.stat_mappings:
            raise ValueError(f"Unknown stat type: {stat_type}. Available types: {list(self.stat_mappings.keys())}")
        
        folder_path = self.stat_mappings[stat_type]
        
        # Build path components
        path_components = [entity_type, season, season_type]
        
        # Add stat type folder structure
        if '/' in folder_path:
            path_components.extend(folder_path.split('/')[:-1])  # Add parent folders
        
        # Create directory path
        dir_path = self.base_dir / '/'.join(path_components)
        
        # Generate filename
        if '/' in folder_path:
            filename = folder_path.split('/')[-1] + '.csv'
        else:
            filename = folder_path + '.csv'
        
        return dir_path / filename
    
    def save_dataframe(self, df, stat_type, season, season_type='regular_season', 
                      entity_type='players', backup_existing=True):
        """
        Save dataframe to organized structure
        
        Args:
            df: pandas DataFrame to save
            stat_type: Type of statistics
            season: Season string
            season_type: 'regular_season' or 'playoffs'
            entity_type: 'players' or 'teams'
            backup_existing: Whether to backup existing files
        
        Returns:
            Path where file was saved
        """
        file_path = self.get_file_path(stat_type, season, season_type, entity_type)
        
        # Create directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if it exists
        if backup_existing and file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_suffix(f'.backup_{timestamp}.csv')
            file_path.rename(backup_path)
            print(f"📋 Backed up existing file to: {backup_path}")
        
        # Save new file
        df.to_csv(file_path, index=False)
        print(f"💾 Saved {len(df)} rows to: {file_path}")
        
        # Update tracking
        self._update_tracking(file_path, len(df))
        
        return file_path
    
    def load_dataframe(self, stat_type, season, season_type='regular_season', entity_type='players'):
        """
        Load dataframe from organized structure
        
        Returns:
            pandas DataFrame or None if file doesn't exist
        """
        file_path = self.get_file_path(stat_type, season, season_type, entity_type)
        
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"📖 Loaded {len(df)} rows from: {file_path}")
        return df
    
    def list_available_data(self, entity_type='players', season=None, season_type=None):
        """
        List all available data files matching criteria
        
        Returns:
            List of dictionaries with file information
        """
        base_path = self.base_dir / entity_type
        
        if not base_path.exists():
            return []
        
        files = []
        for csv_file in base_path.rglob("*.csv"):
            # Skip backup files
            if '.backup' in csv_file.name:
                continue
            
            # Parse path components
            relative_path = csv_file.relative_to(base_path)
            path_parts = relative_path.parts
            
            if len(path_parts) < 3:
                continue
            
            file_season = path_parts[0]
            file_season_type = path_parts[1]
            
            # Apply filters
            if season and file_season != season:
                continue
            if season_type and file_season_type != season_type:
                continue
            
            # Build stat type from path
            if len(path_parts) == 3:
                # Direct file: players/2024-25/regular_season/defense_rebounding.csv
                stat_category = csv_file.stem
            else:
                # Nested file: players/2024-25/regular_season/on_off/efficiency_four_factors.csv
                parent_category = path_parts[2]
                stat_category = f"{parent_category}/{csv_file.stem}"
            
            files.append({
                'entity_type': entity_type,
                'season': file_season,
                'season_type': file_season_type,
                'stat_type': stat_category,
                'file_path': str(csv_file),
                'file_size': csv_file.stat().st_size,
                'modified_time': datetime.fromtimestamp(csv_file.stat().st_mtime)
            })
        
        return sorted(files, key=lambda x: (x['season'], x['season_type'], x['stat_type']))
    
    def _update_tracking(self, file_path, row_count):
        """Update tracking information for saved files"""
        tracking_dir = self.base_dir / 'tracking'
        tracking_dir.mkdir(parents=True, exist_ok=True)
        
        tracking_file = tracking_dir / 'file_operations.json'
        
        # Load existing tracking data
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)
        else:
            tracking_data = {'operations': []}
        
        # Add new operation
        operation = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'save',
            'file_path': str(file_path.relative_to(self.base_dir)),
            'row_count': row_count,
            'file_size': file_path.stat().st_size
        }
        
        tracking_data['operations'].append(operation)
        
        # Keep only last 1000 operations
        tracking_data['operations'] = tracking_data['operations'][-1000:]
        
        # Save tracking data
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
    
    def get_data_summary(self):
        """Get summary of all available data"""
        summary = {
            'players': {},
            'teams': {}
        }
        
        for entity_type in ['players', 'teams']:
            files = self.list_available_data(entity_type)
            
            # Group by season
            by_season = {}
            for file_info in files:
                season = file_info['season']
                if season not in by_season:
                    by_season[season] = {'regular_season': [], 'playoffs': []}
                
                by_season[season][file_info['season_type']].append({
                    'stat_type': file_info['stat_type'],
                    'size_mb': round(file_info['file_size'] / 1024 / 1024, 2),
                    'modified': file_info['modified_time'].strftime('%Y-%m-%d %H:%M')
                })
            
            summary[entity_type] = by_season
        
        return summary


# Example usage functions for scrapers
def save_player_stats(df, stat_type, season, season_type='regular_season'):
    """Convenience function for scrapers to save player data"""
    manager = CTGFileManager()
    return manager.save_dataframe(df, stat_type, season, season_type, 'players')


def load_player_stats(stat_type, season, season_type='regular_season'):
    """Convenience function to load player data"""
    manager = CTGFileManager()
    return manager.load_dataframe(stat_type, season, season_type, 'players')


def get_available_player_data():
    """Get summary of all available player data"""
    manager = CTGFileManager()
    return manager.list_available_data('players')


if __name__ == "__main__":
    # Demo usage
    manager = CTGFileManager()
    
    print("CTG File Manager - Available Data Summary")
    print("=" * 50)
    
    summary = manager.get_data_summary()
    
    for entity_type, seasons in summary.items():
        if seasons:  # Only show if data exists
            print(f"\n📊 {entity_type.upper()}")
            for season, season_data in seasons.items():
                print(f"  🏀 {season}")
                for season_type, files in season_data.items():
                    if files:
                        print(f"    📅 {season_type}: {len(files)} stat types")
                        for file_info in files[:3]:  # Show first 3
                            print(f"       • {file_info['stat_type']} ({file_info['size_mb']}MB)")
                        if len(files) > 3:
                            print(f"       • ... and {len(files) - 3} more")