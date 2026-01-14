#!/usr/bin/env python3
"""Simple script to run all scrapers"""

import subprocess
import sys
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_scraper(scraper_name: str) -> bool:
    """Run a single scraper script"""
    try:
        logger.info(f"Running {scraper_name}...")
        # Run from project root directory where config is available
        project_root = Path(__file__).parent.parent
        scraper_script = project_root / f"scrape_{scraper_name}.py"
        
        # FIXED: Set environment variables to avoid config issues
        env = os.environ.copy()
        env['DEBUG'] = 'False'
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # FIXED: Don't capture output - let it stream to console
        result = subprocess.run(
            [sys.executable, str(scraper_script)],
            check=True,
            cwd=project_root,  # Run from project root
            env=env
        )
        
        logger.info(f"{scraper_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{scraper_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"{scraper_name} failed: {str(e)}")
        return False

def main():
    """Run all scrapers"""
    logger.info("Starting all scrapers...")

    scrapers = ["mytek", "tunisianet", "ldlc"]
    success_count = 0
    failed_scrapers = []

    for scraper in scrapers:
        if run_scraper(scraper):
            success_count += 1
        else:
            failed_scrapers.append(scraper)

    logger.info(f"Completed: {success_count}/{len(scrapers)} scrapers")
    
    if failed_scrapers:
        logger.warning(f"Failed scrapers: {', '.join(failed_scrapers)}")
    
    return success_count == len(scrapers)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)