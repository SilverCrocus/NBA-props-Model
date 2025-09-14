#!/bin/bash

echo "🚀 CTG Scraper Runner"
echo "===================="
echo ""

# First, ensure progress is up to date
echo "📊 Updating progress tracker..."
python3 rebuild_progress.py
echo ""

# Kill any stuck Chrome processes
echo "🔨 Cleaning up Chrome processes..."
python3 kill_chrome.py
echo ""

# Run the scraper
echo "🏀 Starting CTG scraper..."
echo "   This will resume from your last position (83/660 files)"
echo ""
python3 ctg_robust_scraper.py