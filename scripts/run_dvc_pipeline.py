#!/usr/bin/env python3
"""Script to run the DVC pipeline with proper error handling and logging"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command with error handling"""
    try:
        logger.info(f"🚀 Running: {command}")
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(f"📝 Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {command}")
        logger.error(f"💥 Error: {e.stderr}")
        return False

def main():
    """Main pipeline execution"""
    logger.info("🚀 Starting DVC pipeline execution")
    logger.info("=" * 50)

    # Step 1: Initialize DVC (if not already done)
    if not Path(".dvc").exists():
        if not run_command("dvc init"):
            return False

    # Step 2: Add data to DVC tracking
    data_paths = [
        "data/raw/",
        "data/processed/",
        "data/final/"
    ]

    for data_path in data_paths:
        if Path(data_path).exists():
            if not run_command(f"dvc add {data_path}"):
                logger.warning(f"⚠️  Could not add {data_path} to DVC")
        else:
            logger.warning(f"⚠️  {data_path} does not exist")

    # Step 3: Run the DVC pipeline
    logger.info("\n🔗 Running DVC pipeline...")
    if not run_command("dvc repro"):
        return False

    # Step 4: Push to remote storage
    logger.info("\n💾 Pushing data to remote storage...")
    if not run_command("dvc push"):
        logger.warning("⚠️  Could not push to remote storage")

    logger.info("\n" + "=" * 50)
    logger.info("🎉 DVC pipeline completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)