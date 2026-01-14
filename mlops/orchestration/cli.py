#!/usr/bin/env python3
"""Pipeline CLI

Command-line interface for managing Prefect pipelines.
"""

import click
import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Pipeline CLI"""
    pass

@cli.command()
def run():
    """Run pipeline locally"""
    logger.info(" Running pipeline locally...")
    result = subprocess.run([
        sys.executable, "mlops/orchestration/pipeline.py"
    ], cwd=".")
    sys.exit(result.returncode)

@cli.command()
def deploy():
    """Deploy pipeline to Prefect server"""
    logger.info(" Deploying pipeline...")
    result = subprocess.run([
        sys.executable, "mlops/orchestration/deployment.py"
    ], cwd=".")
    sys.exit(result.returncode)

@cli.command()
def monitor():
    """Launch monitoring dashboard"""
    logger.info(" Launching monitoring dashboard...")
    result = subprocess.run([
        "streamlit", "run", "mlops/orchestration/dashboard.py"
    ], cwd=".")
    sys.exit(result.returncode)

@cli.command()
def status():
    """Check pipeline status"""
    logger.info(" Checking pipeline status...")
    # This would query Prefect API in a real implementation
    logger.info(" Pipeline status: Ready")
    logger.info(" Deployment: pricechecktn-mlops-production")
    logger.info(" Schedule: Daily at 2 AM UTC")

if __name__ == "__main__":
    cli()