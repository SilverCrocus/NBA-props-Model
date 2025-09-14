#!/usr/bin/env python3
"""Quick test script to verify the reorganized structure works"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    # Test imports
    print("Testing imports after reorganization...")
    print("-" * 50)
    
    # Test data imports
    from data.scrapers import ctg_scraper
    print("✓ Imported ctg_scraper")
    
    from data.utils import file_manager
    print("✓ Imported file_manager")
    
    # Test features imports
    from features import engineering, selection
    print("✓ Imported feature modules")
    
    # Test preprocessing
    from preprocessing import preprocessor
    print("✓ Imported preprocessor")
    
    # Test pipelines
    from pipelines import training_pipeline
    print("✓ Imported training_pipeline")
    
    print("-" * 50)
    print("✅ All imports successful! Structure is working correctly.")
    
    # Test instantiation
    print("\nTesting class instantiation...")
    print("-" * 50)
    
    # Test preprocessor
    dp = preprocessor.NBADataPreprocessor()
    print("✓ Created NBADataPreprocessor instance")
    
    # Test feature engineer
    fe = engineering.NBAFeatureEngineer()
    print("✓ Created NBAFeatureEngineer instance")
    
    # Test feature selector
    fs = selection.FeatureSelector()
    print("✓ Created FeatureSelector instance")
    
    print("-" * 50)
    print("✅ All classes instantiated successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("You may need to install the package in development mode:")
    print("  cd /Users/diyagamah/Documents/nba_props_model")
    print("  uv pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)