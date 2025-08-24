#!/usr/bin/env python3
"""
CTG ROBUST SCRAPER WITH SESSION MANAGEMENT
Handles session timeouts, resumes from failures, and organizes files properly
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException, InvalidSessionIdException
import time
import os
import shutil
import json
import subprocess
import psutil
from datetime import datetime
from pathlib import Path

class CTGRobustScraper:
    def __init__(self):
        self.base_url = "https://cleaningtheglass.com/stats/players"
        self.data_dir = Path("/Users/diyagamah/Documents/nba_props_model/data/ctg_data_organized")
        self.tracking_dir = self.data_dir / "tracking"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Chrome profile for persistent login
        self.profile_dir = Path.home() / ".ctg_chrome_profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking file
        self.progress_file = self.tracking_dir / "download_progress.json"
        self.session_crashes_file = self.tracking_dir / "session_crashes.json"
        
        # Session management
        self.max_files_per_session = 40  # Restart browser every 40 files
        self.current_session_files = 0
        self.driver = None
        
        # All configurations
        self.categories = [
            "Offensive Overview",
            "Shooting: Overall",
            "Shooting: Frequency", 
            "Shooting: Accuracy",
            "Foul Drawing",
            "Defense and Rebounding",
            "On/Off Efficiency & Four Factors",
            "On/Off Team Shooting: Frequency",
            "On/Off Team Shooting: Accuracy",
            "On/Off Team Halfcourt & Putbacks",
            "On/Off Team Transition",
            "On/Off Opponent Shooting: Frequency",
            "On/Off Opponent Shooting: Accuracy",
            "On/Off Opponent Halfcourt & Putbacks",
            "On/Off Opponent Transition"
        ]
        
        self.seasons = [
            "2024-25", "2023-24", "2022-23", "2021-22", "2020-21",
            "2019-20", "2018-19", "2017-18", "2016-17", "2015-16",
            "2014-15", "2013-14", "2012-13", "2011-12", "2010-11",
            "2009-10", "2008-09", "2007-08", "2006-07", "2005-06",
            "2004-05", "2003-04"
        ]
        
        self.season_types = ["Regular Season", "Playoffs"]
        
        # Load progress
        self.completed_files = self.load_progress()
        
    def load_progress(self):
        """Load previously completed downloads"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                return set(tuple(x) for x in data.get('completed', []))
        return set()
        
    def save_progress(self):
        """Save download progress"""
        progress = {
            'completed': list(self.completed_files),
            'total_expected': len(self.categories) * len(self.seasons) * len(self.season_types),
            'last_updated': datetime.now().isoformat(),
            'completion_rate': f"{len(self.completed_files) * 100 / (len(self.categories) * len(self.seasons) * len(self.season_types)):.1f}%"
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    def log_session_crash(self, error_msg, last_successful=None):
        """Log session crashes for debugging"""
        crashes = []
        if self.session_crashes_file.exists():
            with open(self.session_crashes_file, 'r') as f:
                crashes = json.load(f)
                
        crashes.append({
            'timestamp': datetime.now().isoformat(),
            'error': str(error_msg)[:200],
            'last_successful': last_successful,
            'files_completed': len(self.completed_files)
        })
        
        with open(self.session_crashes_file, 'w') as f:
            json.dump(crashes, f, indent=2)
            
    def setup_driver(self):
        """Setup Chrome with robust options and persistent profile"""
        print("\n🔄 Starting new browser session...")
        
        # Kill any existing Chrome processes first
        self.kill_chrome_processes()
        
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True)
        
        # Use persistent Chrome profile to maintain login
        print(f"📁 Using Chrome profile: {self.profile_dir}")
        options.add_argument(f"--user-data-dir={self.profile_dir}")
        options.add_argument("--profile-directory=CTG_Profile")
        
        # Memory management to prevent crashes
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=VizDisplayCompositor')
        
        # Set download directory - use home directory instead of /tmp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_dir = str(Path.home() / f"ctg_downloads_{timestamp}")
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"📥 Download directory: {self.temp_dir}")
        
        # More comprehensive download preferences
        prefs = {
            "download.default_directory": self.temp_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
            "safebrowsing.disable_download_protection": True,
            "profile.default_content_setting_values.automatic_downloads": 1
        }
        options.add_experimental_option("prefs", prefs)
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.maximize_window()
        self.current_session_files = 0
        
        return self.driver
        
    def kill_chrome_processes(self):
        """Kill any hanging Chrome processes to free up the profile"""
        killed = False
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'chrome' in proc.info['name'].lower() or 'chromium' in proc.info['name'].lower():
                    try:
                        proc.terminate()
                        killed = True
                    except:
                        pass
            if killed:
                time.sleep(2)  # Give processes time to terminate
                # Force kill if still running
                for proc in psutil.process_iter(['pid', 'name']):
                    if 'chrome' in proc.info['name'].lower() or 'chromium' in proc.info['name'].lower():
                        try:
                            proc.kill()
                        except:
                            pass
        except:
            # Fallback to subprocess if psutil fails
            try:
                subprocess.run(['pkill', '-f', 'Chrome'], capture_output=True)
                subprocess.run(['pkill', '-f', 'chromedriver'], capture_output=True)
                killed = True
            except:
                pass
        
        if killed:
            print("   🔨 Killed hanging Chrome processes")
            time.sleep(2)
    
    def restart_browser_session(self):
        """Restart browser to prevent session timeout with proper cleanup"""
        print("\n🔄 Restarting browser session to prevent timeout...")
        
        # First, try to close driver normally
        if self.driver:
            try:
                self.driver.quit()
                print("   ✅ Closed browser normally")
            except Exception as e:
                print(f"   ⚠️ Normal close failed: {str(e)[:50]}")
        
        # Wait a bit for normal close
        time.sleep(3)
        
        # Kill any hanging Chrome processes
        self.kill_chrome_processes()
        
        # Try to start new browser with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.setup_driver()
                self.login()
                print("   ✅ Browser restarted successfully")
                return
            except Exception as e:
                if "user data directory is already in use" in str(e):
                    print(f"   ⚠️ Chrome profile still locked (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        # More aggressive cleanup
                        self.kill_chrome_processes()
                        time.sleep(5)  # Longer wait
                    else:
                        # Last resort: use a different profile directory
                        print("   🔄 Switching to temporary profile...")
                        self.profile_dir = Path.home() / f".ctg_chrome_profile_temp_{int(time.time())}"
                        self.profile_dir.mkdir(parents=True, exist_ok=True)
                        self.setup_driver()
                        self.login()
                        print("   ✅ Using temporary profile")
                        return
                else:
                    print(f"   ❌ Restart failed: {str(e)[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        raise
        
    def login(self):
        """Navigate and check login status"""
        self.driver.get(self.base_url)
        time.sleep(3)
        
        # Check if already logged in by looking for user-specific elements
        try:
            # Try to find a dropdown menu (they should be visible if logged in)
            selects = self.driver.find_elements(By.TAG_NAME, "select")
            if len(selects) >= 3:
                print("✅ Already logged in! Using saved session.")
                return
        except:
            pass
        
        # If not logged in, prompt for manual login
        print("\n" + "="*60)
        print("Please log into CleaningTheGlass")
        print("NOTE: Your login will be saved for future sessions!")
        print("="*60)
        input("Press Enter when logged in and ready...")
        
    def is_file_already_downloaded(self, category, season, season_type):
        """Check if file already exists in organized structure"""
        season_folder = "regular_season" if season_type == "Regular Season" else "playoffs"
        # Replace & first, then spaces, then clean up multiple underscores
        category_clean = category.lower().replace(" & ", "_").replace("&", "_").replace(":", "").replace("/", "_").replace(" ", "_")
        # Clean up any multiple underscores
        while "__" in category_clean:
            category_clean = category_clean.replace("__", "_")
        
        # Handle on_off subcategories
        if category_clean.startswith("on_off_"):
            subfolder_name = category_clean.replace("on_off_", "")
            file_path = self.data_dir / "players" / season / season_folder / "on_off" / subfolder_name / f"{subfolder_name}.csv"
        else:
            # Regular categories
            file_path = self.data_dir / "players" / season / season_folder / category_clean / f"{category_clean}.csv"
        
        return file_path.exists() or (category, season, season_type) in self.completed_files
        
    def download_file_with_retry(self, category, season, season_type, max_retries=3):
        """Download with retry logic and session management"""
        
        for attempt in range(max_retries):
            try:
                # Check if driver exists and is valid
                if not hasattr(self, 'driver') or self.driver is None:
                    print("  ⚠️ No browser session, starting new one...")
                    self.setup_driver()
                    self.login()
                
                # Check for invalid session
                try:
                    _ = self.driver.current_url
                except (InvalidSessionIdException, AttributeError):
                    print("  ⚠️ Session expired, restarting browser...")
                    self.restart_browser_session()
                    
                # Get fresh element references
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                if len(selects) != 3:
                    print(f"  ⚠️ Expected 3 selects, found {len(selects)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False
                    
                # Select dropdowns
                Select(selects[0]).select_by_visible_text(category)
                time.sleep(1.5)
                
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                Select(selects[1]).select_by_visible_text(season)
                time.sleep(1.5)
                
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                Select(selects[2]).select_by_visible_text(season_type)
                time.sleep(3)  # Wait for table
                
                # Click download button
                download_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[@class='download_button']"))
                )
                self.driver.execute_script("arguments[0].click();", download_btn)
                
                # Wait for download
                time.sleep(4)
                
                # Check if download actually happened
                try:
                    temp_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
                except (FileNotFoundError, AttributeError) as e:
                    print(f"  ❌ Temp directory issue: {e}")
                    print(f"      Temp dir: {self.temp_dir if hasattr(self, 'temp_dir') else 'NOT SET'}")
                    return False
                    
                initial_count = len(temp_files)
                print(f"  📁 Checking: {self.temp_dir}")
                time.sleep(6)  # Longer wait for download
                
                temp_files_after = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
                new_count = len(temp_files_after)
                
                # Also check default Downloads folder as fallback
                downloads_dir = Path.home() / "Downloads"
                recent_downloads = []
                try:
                    all_downloads = list(downloads_dir.glob("*.csv"))
                    # Get files modified in last 10 seconds
                    current_time = time.time()
                    recent_downloads = [f for f in all_downloads if (current_time - f.stat().st_mtime) < 10]
                    if recent_downloads:
                        print(f"  📥 Found in Downloads folder: {recent_downloads[0].name}")
                        # Move the file to our temp directory
                        for file in recent_downloads:
                            shutil.move(str(file), self.temp_dir)
                            new_count += 1
                            print(f"  ✅ Moved from Downloads to temp dir")
                except Exception as e:
                    pass  # Ignore errors checking Downloads
                
                if new_count <= initial_count:
                    print(f"  ⚠️ No file downloaded to {self.temp_dir}")
                    # Still mark as completed to avoid retrying files that don't exist
                    self.completed_files.add((category, season, season_type))
                    return False
                
                # Move and organize file
                if self.organize_downloaded_file(category, season, season_type):
                    self.completed_files.add((category, season, season_type))
                    self.current_session_files += 1
                    print(f"  ✅ Downloaded: {category} - {season} - {season_type}")
                    return True
                else:
                    print(f"  ❌ Failed to organize file")
                    
            except (StaleElementReferenceException, TimeoutException) as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Retry {attempt + 1}/{max_retries}: {str(e)[:50]}")
                    time.sleep(2)
                else:
                    print(f"  ❌ Failed after {max_retries} attempts")
                    return False
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                return False
                
        return False
        
    def organize_downloaded_file(self, category, season, season_type):
        """Move file to organized structure"""
        try:
            # Find the downloaded file
            files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
            if not files:
                return False
                
            latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(self.temp_dir, f)))
            source = os.path.join(self.temp_dir, latest_file)
            
            # Create organized path
            season_folder = "regular_season" if season_type == "Regular Season" else "playoffs"
            # Replace & first, then spaces, then clean up multiple underscores
            category_clean = category.lower().replace(" & ", "_").replace("&", "_").replace(":", "").replace("/", "_").replace(" ", "_")
            # Clean up any multiple underscores
            while "__" in category_clean:
                category_clean = category_clean.replace("__", "_")
            
            # Handle on_off subcategories
            if category_clean.startswith("on_off_"):
                # Remove on_off prefix for subfolder name
                subfolder_name = category_clean.replace("on_off_", "")
                dest_dir = self.data_dir / "players" / season / season_folder / "on_off" / subfolder_name
                filename = f"{subfolder_name}.csv"
            else:
                # Regular categories
                dest_dir = self.data_dir / "players" / season / season_folder / category_clean
                filename = f"{category_clean}.csv"
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / filename
            
            # Move file
            shutil.move(source, str(dest_file))
            return True
            
        except Exception as e:
            print(f"    Could not organize file: {e}")
            return False
            
    def run_full_scrape(self):
        """Run complete scraping with resume capability"""
        
        # Calculate what needs to be downloaded
        all_combinations = [
            (cat, season, stype) 
            for season in self.seasons 
            for stype in self.season_types 
            for cat in self.categories
        ]
        
        # Filter out already completed
        needed = [combo for combo in all_combinations if not self.is_file_already_downloaded(*combo)]
        
        print("\n" + "="*60)
        print("CTG ROBUST SCRAPER STATUS")
        print("="*60)
        print(f"Total files: {len(all_combinations)}")
        print(f"Already completed: {len(self.completed_files)}")
        print(f"Still needed: {len(needed)}")
        print(f"Session restart every: {self.max_files_per_session} files")
        print("="*60)
        
        if not needed:
            print("\n✅ All files already downloaded!")
            return
            
        # Start downloading
        self.setup_driver()
        self.login()
        
        successful = 0
        failed = []
        start_time = datetime.now()
        
        for i, (category, season, season_type) in enumerate(needed, 1):
            print(f"\n[{i}/{len(needed)}] Downloading: {season} - {season_type} - {category}")
            
            # Check if we need to restart browser
            if self.current_session_files >= self.max_files_per_session:
                print("\n" + "="*40)
                print("Session limit reached - Restarting browser")
                print("="*40)
                self.restart_browser_session()
                
            # Download file
            if self.download_file_with_retry(category, season, season_type):
                successful += 1
            else:
                failed.append((category, season, season_type))
                
            # Save progress every 5 files
            if successful % 5 == 0:
                self.save_progress()
                
            # Progress update
            if i % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(needed) - i) / rate if rate > 0 else 0
                
                print(f"\n{'='*40}")
                print(f"PROGRESS: {i}/{len(needed)} ({successful} successful)")
                print(f"Time remaining: {int(remaining/60)} minutes")
                print(f"Session files: {self.current_session_files}/{self.max_files_per_session}")
                print(f"{'='*40}")
                
            # Rate limiting
            time.sleep(2)
            
        # Final save
        self.save_progress()
        
        # Final report
        print("\n" + "="*60)
        print("SCRAPING COMPLETE!")
        print("="*60)
        print(f"Total processed: {len(needed)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {successful*100/len(needed):.1f}%")
        print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
        
        if failed:
            print(f"\nFailed downloads ({len(failed)}):")
            for item in failed[:10]:
                print(f"  - {item[0]} / {item[1]} / {item[2]}")
                
        # Clean up
        if self.driver:
            self.driver.quit()
            

def main():
    scraper = CTGRobustScraper()
    
    print("\n" + "="*60)
    print("CTG ROBUST SCRAPER WITH SESSION MANAGEMENT")
    print("="*60)
    print("\n✅ Features:")
    print("  • 🔐 Persistent Chrome profile (stay logged in!)")
    print("  • Automatic browser restart every 40 files")
    print("  • Resume from last successful download")
    print("  • Organized folder structure (season/type)")
    print("  • Progress tracking with JSON")
    print("  • Session crash recovery")
    print("\n1. Resume downloading (continues from last point)")
    print("2. Check progress status")
    print("3. Exit")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        scraper.run_full_scrape()
    elif choice == "2":
        if scraper.progress_file.exists():
            with open(scraper.progress_file, 'r') as f:
                progress = json.load(f)
                print(f"\nProgress: {progress['completion_rate']}")
                print(f"Completed: {len(scraper.completed_files)} files")
                print(f"Last updated: {progress['last_updated']}")
    else:
        print("Exiting...")
        

if __name__ == "__main__":
    main()