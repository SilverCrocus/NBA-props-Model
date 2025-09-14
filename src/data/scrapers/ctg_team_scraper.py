#!/usr/bin/env python3
"""
CTG Team Stats Scraper
Separate script for scraping team-level statistics from CleaningTheGlass.com
Designed to not interfere with the player scraper
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
import logging
import psutil
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CTGTeamScraper:
    """Scraper for CTG team statistics - separate from player scraper"""
    
    def __init__(self, base_dir="/Users/diyagamah/Documents/nba_props_model/data"):
        self.base_dir = Path(base_dir)
        self.teams_dir = self.base_dir / "ctg_team_data"
        self.teams_dir.mkdir(exist_ok=True, parents=True)
        
        # Create tracking directory
        self.tracking_dir = self.teams_dir / "tracking"
        self.tracking_dir.mkdir(exist_ok=True)
        
        # Use the same Chrome profile as the player scraper (already has CTG login)
        self.profile_dir = Path.home() / ".ctg_chrome_profile"
        
        # Team mappings
        self.teams = {
            1: "Atlanta Hawks", 2: "Boston Celtics", 3: "Brooklyn Nets",
            4: "Charlotte Hornets", 5: "Chicago Bulls", 6: "Cleveland Cavaliers",
            7: "Dallas Mavericks", 8: "Denver Nuggets", 9: "Detroit Pistons",
            10: "Golden State Warriors", 11: "Houston Rockets", 12: "Indiana Pacers",
            13: "Los Angeles Clippers", 14: "Los Angeles Lakers", 15: "Memphis Grizzlies",
            16: "Miami Heat", 17: "Milwaukee Bucks", 18: "Minnesota Timberwolves",
            19: "New Orleans Pelicans", 20: "New York Knicks", 21: "Oklahoma City Thunder",
            22: "Orlando Magic", 23: "Philadelphia 76ers", 24: "Phoenix Suns",
            25: "Portland Trail Blazers", 26: "Sacramento Kings", 27: "San Antonio Spurs",
            28: "Toronto Raptors", 29: "Utah Jazz", 30: "Washington Wizards"
        }
        
        # Team stat categories (both offense and defense)
        self.categories = [
            "Team Efficiency and Four Factors",
            "Offense: Shooting Frequency",
            "Offense: Shooting Accuracy",
            "Offense: Play Context: Halfcourt and Putbacks",
            "Offense: Play Context: Transition",
            "Defense: Shooting Frequency",
            "Defense: Shooting Accuracy", 
            "Defense: Play Context: Halfcourt and Putbacks",
            "Defense: Play Context: Transition"
        ]
        
        # Seasons to scrape
        self.seasons = [
            "2024-25", "2023-24", "2022-23", "2021-22", "2020-21",
            "2019-20", "2018-19", "2017-18", "2016-17", "2015-16",
            "2014-15", "2013-14", "2012-13", "2011-12", "2010-11",
            "2009-10", "2008-09", "2007-08", "2006-07", "2005-06",
            "2004-05", "2003-04"
        ]
        
        # We'll always show playoff stats to get both in one download
        self.include_playoffs = True
        
        # Load progress
        self.progress_file = self.tracking_dir / "team_download_progress.json"
        self.completed = self.load_progress()
        
        self.driver = None
        self.wait = None
        self.download_dir = str(Path.home() / "Downloads")
        
    def kill_chrome_processes(self):
        """Kill any hanging Chrome processes to free up the profile"""
        logger.info("Checking for stuck Chrome processes...")
        killed_count = 0
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'chrome' in proc.info['name'].lower() or 'chromedriver' in proc.info['name'].lower():
                    proc.terminate()
                    killed_count += 1
            except:
                pass
        
        if killed_count > 0:
            time.sleep(2)
            logger.info(f"Killed {killed_count} Chrome processes")
    
    def load_progress(self):
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded progress: {len(data.get('completed', []))} files already downloaded")
                return data.get('completed', [])
        return []
    
    def save_progress(self):
        """Save progress to file"""
        progress_data = {
            'completed': self.completed,
            'total_expected': len(self.teams) * len(self.categories),  # 30 teams × 9 categories = 270
            'last_updated': datetime.now().isoformat(),
            'note': 'Each file contains both regular season and playoff data'
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved: {len(self.completed)} files completed")
    
    def setup_driver(self, restart=False):
        """Setup Chrome driver with user's main Chrome profile
        
        This uses your actual Chrome profile, so:
        - You'll already be logged in to sites you use
        - Your bookmarks and extensions will be available
        - Close any other Chrome windows before running
        """
        if restart and self.driver:
            logger.info("Restarting browser...")
            self.driver.quit()
            time.sleep(2)
        
        # Kill any stuck processes
        self.kill_chrome_processes()
        
        options = webdriver.ChromeOptions()
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Additional options to fix Chrome startup issues
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Download preferences
        prefs = {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        
        logger.info("Chrome driver initialized with persistent profile")
    
    def navigate_to_team_stats(self, team_id):
        """Navigate to a specific team's stats page
        
        URL structure:
        - /stats/team/{id}/player - PLAYER STATS tab
        - /stats/team/{id}/onoff - ON/OFF STATS tab  
        - /stats/team/{id}/team - TEAM STATS tab (what we want)
        - /stats/team/{id}/gamelogs - GAME LOGS tab
        - /stats/team/{id}/lineups - LINEUPS tab
        - /stats/team/{id}/salaries - SALARIES tab
        """
        url = f"https://cleaningtheglass.com/stats/team/{team_id}/team"
        logger.info(f"Navigating to {self.teams[team_id]} stats page...")
        self.driver.get(url)
        time.sleep(2)
        
        # Verify we're on the TEAM STATS tab, click it if not
        try:
            # Check if TEAM STATS is already active
            active_tab = self.driver.find_element(
                By.XPATH, "//a[contains(@class, 'active') and contains(text(), 'Team Stats')]"
            )
            logger.info("✅ Already on TEAM STATS tab")
        except:
            # If not active, click the TEAM STATS tab
            try:
                team_stats_tab = self.driver.find_element(
                    By.XPATH, "//a[contains(text(), 'Team Stats')]"
                )
                team_stats_tab.click()
                logger.info("📊 Clicked TEAM STATS tab")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Could not verify/click TEAM STATS tab: {e}")
    
    def select_category(self, category):
        """Select a stat category from the dropdown"""
        try:
            # Try multiple methods to find the dropdown
            dropdown = None
            
            # Method 1: Try by ID
            try:
                dropdown = self.driver.find_element(By.ID, "stat-tab-select")
            except:
                pass
            
            # Method 2: Try by class and tag
            if not dropdown:
                try:
                    dropdowns = self.driver.find_elements(By.TAG_NAME, "select")
                    # Find the dropdown that contains our category options
                    for dd in dropdowns:
                        options_text = dd.text
                        if "Team Efficiency" in options_text or "Shooting" in options_text:
                            dropdown = dd
                            break
                except:
                    pass
            
            if not dropdown:
                logger.error(f"Could not find category dropdown on page")
                return False
            
            select = Select(dropdown)
            
            # Map category names to dropdown options
            category_map = {
                "Team Efficiency and Four Factors": "Team Efficiency and Four Factors",
                "Offense: Shooting Frequency": "Shooting Frequency",
                "Offense: Shooting Accuracy": "Shooting Accuracy",
                "Offense: Play Context: Halfcourt and Putbacks": "Play Context: Halfcourt and Putbacks",
                "Offense: Play Context: Transition": "Play Context: Transition",
                "Defense: Shooting Frequency": "Shooting Frequency",
                "Defense: Shooting Accuracy": "Shooting Accuracy", 
                "Defense: Play Context: Halfcourt and Putbacks": "Play Context: Halfcourt and Putbacks",
                "Defense: Play Context: Transition": "Play Context: Transition"
            }
            
            mapped_name = category_map.get(category, category)
            
            # For defense categories, we need special handling
            if "Defense:" in category:
                # Get all options
                options = select.options
                defense_found = False
                offense_count = 0
                
                # Count offense occurrences first
                for option in options:
                    if option.text == mapped_name and "Defense" not in str(option.get_attribute("value")):
                        offense_count += 1
                
                # Now find the defense occurrence (after offense)
                occurrence_count = 0
                for i, option in enumerate(options):
                    if option.text == mapped_name:
                        occurrence_count += 1
                        if occurrence_count > offense_count:  # This is the defense one
                            select.select_by_index(i)
                            defense_found = True
                            break
                
                if not defense_found:
                    logger.warning(f"Could not find defense category: {category}, trying by text")
                    select.select_by_visible_text(mapped_name)
            else:
                # For offense categories, select by visible text
                select.select_by_visible_text(mapped_name)
            
            # Wait for page to update after category change
            time.sleep(5)  # Give more time for AJAX update
            
            # Wait for table to be present
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except:
                pass
            
            logger.info(f"Selected category: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error selecting category {category}: {str(e)}")
            logger.error(f"Full error: {type(e).__name__}: {e}")
            return False
    
    def show_playoff_stats(self):
        """Click 'Show Playoff Stats' to include both regular season and playoff data"""
        try:
            # Look for the "Show Playoff Stats" link
            playoff_link = self.driver.find_element(
                By.XPATH, "//a[contains(text(), 'Show Playoff Stats')]"
            )
            playoff_link.click()
            time.sleep(2)
            logger.info("✅ Playoff stats added to table")
            return True
        except Exception as e:
            # If not found, playoffs might already be shown or not available
            logger.info(f"Note: Could not add playoff stats (may already be shown or unavailable)")
            return False
    
    def download_current_table(self, team_name, category):
        """Download the currently displayed table (contains both regular season and playoff data)"""
        try:
            # Try multiple approaches to find the download link
            download_link = None
            
            # Method 1: Look for "Download this table" text
            try:
                download_link = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Download this table')]"))
                )
            except:
                logger.warning("Could not find download link by text")
            
            # Method 2: Look for download icon link
            if not download_link:
                try:
                    links = self.driver.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        if "download" in link.get_attribute("class").lower() or \
                           "download" in (link.get_attribute("href") or "").lower():
                            download_link = link
                            break
                except:
                    pass
            
            if not download_link:
                logger.error("Could not find any download link on the page")
                # Take a screenshot for debugging
                try:
                    self.driver.save_screenshot(f"debug_no_download_{team_name}_{category}.png")
                except:
                    pass
                return None
            
            # Make sure element is visible and clickable
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", download_link)
            time.sleep(2)  # Wait for scroll to complete
            
            # Try to click the download link
            click_success = False
            
            # Try regular click first
            try:
                download_link.click()
                click_success = True
                logger.info(f"Clicked download link using regular click")
            except Exception as e1:
                logger.warning(f"Regular click failed: {e1}")
                
                # Try JavaScript click
                try:
                    self.driver.execute_script("arguments[0].click();", download_link)
                    click_success = True
                    logger.info(f"Clicked download link using JavaScript")
                except Exception as e2:
                    logger.error(f"JavaScript click also failed: {e2}")
            
            if not click_success:
                return None
            
            # Wait for download to complete
            time.sleep(4)  # Give more time for download
            
            # Find and move the downloaded file
            newest_file = self.find_newest_csv()
            if newest_file:
                # Create organized path
                safe_team_name = team_name.replace(" ", "_").lower()
                safe_category = category.replace(":", "").replace(" ", "_").lower()
                
                team_dir = self.teams_dir / safe_team_name
                team_dir.mkdir(exist_ok=True, parents=True)
                
                final_path = team_dir / f"{safe_category}_all_seasons.csv"
                shutil.move(newest_file, final_path)
                
                logger.info(f"✅ Downloaded: {team_name} - {category} (includes playoffs)")
                return str(final_path)
            else:
                logger.error(f"❌ Download failed: no new file found for {team_name} - {category}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading table for {team_name} - {category}: {str(e)}")
            logger.error(f"Full error details: {type(e).__name__}: {e}")
            # Take a screenshot for debugging
            try:
                self.driver.save_screenshot(f"debug_error_{team_name}_{category}.png")
            except:
                pass
            return None
    
    def find_newest_csv(self):
        """Find the most recently downloaded CSV file"""
        download_path = Path(self.download_dir)
        csv_files = list(download_path.glob("*.csv"))
        
        if not csv_files:
            return None
        
        # Get the newest file
        newest = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        # Check if it was downloaded in the last 10 seconds
        if time.time() - newest.stat().st_mtime < 10:
            return str(newest)
        
        return None
    
    def scrape_all_teams(self):
        """Main scraping function for all teams"""
        self.setup_driver()
        
        # Navigate to stats page first
        logger.info("Navigating to CTG stats page...")
        self.driver.get("https://cleaningtheglass.com/stats")
        time.sleep(3)
        
        # Check if logged in - try different methods
        logged_in = False
        
        # Method 1: Check for Logout link
        try:
            logout_link = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Logout')]")
            logger.info("✅ Logged in successfully (found Logout link)!")
            logged_in = True
        except:
            pass
        
        # Method 2: Check if we can access team pages directly
        if not logged_in:
            try:
                # Try to navigate to a team page to check access
                test_url = "https://cleaningtheglass.com/stats/team/1/team"
                self.driver.get(test_url)
                time.sleep(2)
                
                # Check if we're on the team page (not redirected to login)
                if "team/1" in self.driver.current_url:
                    logger.info("✅ Logged in successfully (can access team pages)!")
                    logged_in = True
                    # Navigate back to main stats page
                    self.driver.get("https://cleaningtheglass.com/stats")
                    time.sleep(2)
            except:
                pass
        
        if not logged_in:
            logger.warning("⚠️ Could not verify login status, proceeding anyway...")
            logger.warning("If you see errors, please ensure you're logged in to CTG")
        
        total_files = len(self.teams) * len(self.categories)
        logger.info(f"Starting team data collection: {total_files} total files")
        logger.info("Each file will contain BOTH regular season and playoff data")
        
        files_downloaded = 0
        
        # Iterate through all teams
        for team_id, team_name in self.teams.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {team_name}")
            logger.info(f"{'='*60}")
            
            # Navigate to team page
            self.navigate_to_team_stats(team_id)
            
            # Show playoff stats once (this adds playoff rows to all tables)
            self.show_playoff_stats()
            
            for category in self.categories:
                # Check if already downloaded
                combo_key = f"{team_name}_{category}"
                if combo_key in self.completed:
                    logger.info(f"⏭️ Skipping (already done): {combo_key}")
                    continue
                
                # Select category
                if not self.select_category(category):
                    logger.error(f"Failed to select {category}")
                    continue
                
                # Download the table (contains both regular season and playoff data)
                file_path = self.download_current_table(team_name, category)
                
                if file_path:
                    self.completed.append(combo_key)
                    files_downloaded += 1
                    self.save_progress()
                    
                    # Progress update
                    total_completed = len(self.completed)
                    progress_pct = (total_completed / total_files) * 100
                    logger.info(f"Progress: {total_completed}/{total_files} ({progress_pct:.1f}%)")
                
                # Small delay between downloads
                time.sleep(1)
            
            # Restart browser every 10 teams to avoid memory issues
            if team_id % 10 == 0:
                logger.info("Restarting browser to clear memory...")
                self.setup_driver(restart=True)
                time.sleep(3)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ TEAM DATA COLLECTION COMPLETE!")
        logger.info(f"Downloaded: {files_downloaded} new files")
        logger.info(f"Total collected: {len(self.completed)}/{total_files} files")
        logger.info(f"{'='*60}")
        
        # Cleanup
        if self.driver:
            self.driver.quit()
    
    def verify_data_structure(self):
        """Verify the downloaded data structure"""
        logger.info("\n📊 Verifying team data structure...")
        
        total_files = 0
        for team_dir in self.teams_dir.glob("*/"):
            if team_dir.is_dir() and team_dir.name != "tracking":
                team_files = list(team_dir.glob("**/*.csv"))
                total_files += len(team_files)
                logger.info(f"  {team_dir.name}: {len(team_files)} files")
        
        logger.info(f"\n✅ Total team data files: {total_files}")
        return total_files


if __name__ == "__main__":
    print("=" * 60)
    print("🏀 CTG TEAM STATS SCRAPER")
    print("=" * 60)
    print("\nThis script will download team-level statistics from CTG.")
    print("It uses the same Chrome profile as the player scraper.")
    print("(Should auto-login if you've already logged in before)")
    print("\nData categories to collect:")
    print("  - Team Efficiency and Four Factors")
    print("  - Shooting Frequency/Accuracy (Offense & Defense)")
    print("  - Play Context: Halfcourt/Transition (Offense & Defense)")
    print("\n📊 Each file contains BOTH regular season AND playoff data")
    print("   (CTG shows both when 'Show Playoff Stats' is clicked)")
    print("\nTotal files: 30 teams × 9 categories = 270 files")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: Close all other Chrome windows before starting!")
    print("   (This uses your main Chrome profile)")
    
    # Auto-start without confirmation for automated runs
    print("\n▶️ Starting scraper...")
    time.sleep(2)  # Brief pause to allow reading the message
    
    # Kill any existing Chrome processes first
    print("\n🔧 Cleaning up any existing Chrome processes...")
    import subprocess
    try:
        subprocess.run(['pkill', '-f', 'Chrome'], capture_output=True)
        time.sleep(2)
    except:
        pass
    
    scraper = CTGTeamScraper()
    
    # Main scraping
    scraper.scrape_all_teams()
    
    # Verify results
    scraper.verify_data_structure()
    
    print("\n✅ Team scraping complete!")
    print("Data saved to: data/ctg_team_data/")
    print("\nNext steps:")
    print("1. Integrate team data with player features")
    print("2. Build feature engineering pipeline")
    print("3. Create PRA prediction model")