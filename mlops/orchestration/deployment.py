"""
Prefect Deployment Configuration for PriceCheckTN MLOps Pipeline
"""

import subprocess
import sys
from pathlib import Path

def deploy_pipeline():
    """Deploy the Prefect pipeline using the new CLI approach"""

    # Check if prefect CLI is available
    try:
        result = subprocess.run(["python", "-m", "prefect", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Prefect CLI not found. Please ensure Prefect is installed.")
            return False
        print(f"🎉 Prefect CLI found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Prefect CLI not found. Please ensure Prefect is installed.")
        return False

    # Deploy using prefect CLI
    try:
        print("🚀 Deploying Prefect pipeline...")

        # Build deployment using prefect CLI
        deploy_cmd = [
            "python", "-m", "prefect", "deploy",
            "--name", "pricechecktn-mlops-production",
            "--version", "1.0.0",
            "--work-pool", "default-agent-pool",
            "--cron", "0 2 * * *",  # Daily at 2 AM UTC
            "--tag", "mlops",
            "--tag", "production",
            "--tag", "daily",
            "--description", "Daily MLOps pipeline for PriceCheckTN",
            "mlops/orchestration/pipeline.py:pricechecktn_mlops_pipeline"
        ]

        result = subprocess.run(deploy_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Deployment created successfully!")
            print(f"📊 Output: {result.stdout}")
            return True
        else:
            print("❌ Deployment failed:")
            print(f"🔴 Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Unexpected error during deployment: {e}")
        return False

if __name__ == "__main__":
    success = deploy_pipeline()
    sys.exit(0 if success else 1)
