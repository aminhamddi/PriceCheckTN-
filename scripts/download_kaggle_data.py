#!/usr/bin/env python3
"""
Download fake reviews dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
"""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset(dataset_name: str, output_dir: str = "data/raw") -> str:
    """
    Download dataset from Kaggle and extract it.

    Args:
        dataset_name: Kaggle dataset name (owner/dataset-name)
        output_dir: Directory to save the dataset

    Returns:
        Path to extracted dataset directory
    """
    try:
        # Initialize Kaggle API
        logger.info("Initializing Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Download dataset
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

        logger.info(f"Dataset downloaded to: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def explore_downloaded_data(data_dir: str):
    """Explore and validate the downloaded dataset."""
    data_path = Path(data_dir)

    # Find CSV files
    csv_files = list(data_path.glob("**/*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in downloaded dataset")
        return None

    logger.info(f"Found {len(csv_files)} CSV files:")

    for csv_file in csv_files:
        logger.info(f"  - {csv_file.name}")

        # Read and show basic info
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"    Shape: {df.shape}")
            logger.info(f"    Columns: {list(df.columns)}")
            logger.info(f"    Sample rows: {len(df)}")
            logger.info(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            # Show first few rows
            logger.info("    First 3 rows:")
            logger.info(f"{df.head(3).to_string()}")

        except Exception as e:
            logger.error(f"    Error reading {csv_file.name}: {e}")

    return csv_files

def move_to_processed(csv_files, processed_dir: str = "data/processed"):
    """Move and rename downloaded files to processed directory."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    moved_files = []

    for csv_file in csv_files:
        # Create a standardized name
        if 'fake' in csv_file.name.lower():
            new_name = "reviews_cleaned.csv"
        else:
            new_name = f"kaggle_{csv_file.name}"

        new_path = processed_path / new_name

        # Copy file
        import shutil
        shutil.copy2(csv_file, new_path)
        moved_files.append(new_path)

        logger.info(f"Moved {csv_file.name} -> {new_path}")

    return moved_files

def main():
    print("🚀 Downloading Fake Reviews Dataset from Kaggle")
    print("="*60)

    # Dataset details
    dataset_name = "mexwell/fake-reviews-dataset"
    raw_dir = "data/raw"

    try:
        # Download dataset
        download_path = download_kaggle_dataset(dataset_name, raw_dir)

        # Explore downloaded data
        print("\n📊 Exploring Downloaded Data...")
        csv_files = explore_downloaded_data(download_path)

        if csv_files:
            # Move to processed directory
            print("\n📁 Moving Files to Processed Directory...")
            processed_files = move_to_processed(csv_files)

            print("\n✅ Dataset Download Complete!")
            print("="*60)
            print(f"📂 Raw data location: {download_path}")
            print(f"📂 Processed data location: data/processed")
            print(f"📊 Files processed: {len(processed_files)}")

            for file_path in processed_files:
                df = pd.read_csv(file_path)
                print(f"  - {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")

        else:
            print("❌ No CSV files found in the downloaded dataset")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 To fix this:")
        print("1. Install Kaggle API: pip install kaggle")
        print("2. Get Kaggle API token from https://www.kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ or %USERPROFILE%\\.kaggle\\")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
        sys.exit(1)

if __name__ == "__main__":
    main()
