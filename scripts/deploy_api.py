"""API Deployment Script"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_api():
    """Deploy API to production"""
    logger.info("🚀 Starting API deployment...")

    try:
        # Check if requirements are installed
        logger.info("📋 Checking requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "list"
        ], capture_output=True, text=True)

        if "fastapi" not in result.stdout:
            logger.error("❌ FastAPI not installed")
            return False

        # Run QA checks
        logger.info("🔍 Running QA checks...")
        qa_result = subprocess.run([
            sys.executable, "tests/qa_checks.py"
        ], capture_output=True, text=True)

        if qa_result.returncode != 0:
            logger.error(f"❌ QA checks failed: {qa_result.stderr}")
            return False

        # Test API
        logger.info("🧪 Testing API...")
        test_result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_api.py", "-v"
        ], capture_output=True, text=True)

        if test_result.returncode != 0:
            logger.error(f"❌ API tests failed: {test_result.stderr}")
            return False

        logger.info("✅ API deployment completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        return False

if __name__ == "__main__":
    success = deploy_api()
    sys.exit(0 if success else 1)