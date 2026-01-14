"""
Script 1: Consolidation des produits scrapés
--------------------------------------------
Objectif: Unifier tous les fichiers CSV de produits (MyTek, Tunisianet, LDLC)
         en un seul dataset propre et standardisé

Author: PriceCheck TN Team
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.product_normalizer import ProductNormalizer

# Configure logger
logger.add(
    "logs/consolidation_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class ProductConsolidator:
    """Consolidate scraped products from multiple sources"""

    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.final_dir = Path("data/final")
        self.final_dir.mkdir(exist_ok=True, parents=True)

        self.normalizer = ProductNormalizer()

        # Statistics
        self.stats = {
            'total_loaded': 0,
            'by_source': {},
            'duplicates_removed': 0,
            'invalid_removed': 0,
            'duplicate_columns_removed': 0,
            'final_count': 0
        }

        logger.info(" ProductConsolidator initialized")

    def load_latest_csv(self, source: str) -> pd.DataFrame:
        """
        Load latest CSV file for a given source

        Args:
            source: 'mytek', 'tunisianet', or 'ldlc'

        Returns:
            DataFrame with products
        """
        latest_file = self.raw_dir / f"{source}_latest.csv"

        if not latest_file.exists():
            logger.warning(f" Latest file not found: {latest_file}")
            logger.info(f" Searching for timestamped files...")

            # Find most recent timestamped file
            pattern = f"{source}_products_*.csv"
            files = sorted(self.raw_dir.glob(pattern), reverse=True)

            if not files:
                logger.error(f" No files found for source: {source}")
                return pd.DataFrame()

            latest_file = files[0]
            logger.info(f" Found: {latest_file.name}")

        try:
            df = pd.read_csv(latest_file, encoding='utf-8')
            logger.info(f" Loaded {len(df)} products from {source}")
            return df

        except Exception as e:
            logger.error(f" Error loading {latest_file}: {e}")

            # Try with different encoding
            try:
                df = pd.read_csv(latest_file, encoding='latin-1')
                logger.warning(f"  Loaded with latin-1 encoding")
                return df
            except Exception as e2:
                logger.error(f" Failed with latin-1: {e2}")
                return pd.DataFrame()

    def standardize_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Standardize column names across different sources
        """
        column_mapping = {
            'title': 'product_name',
            'name': 'product_name',
            'product_title': 'product_name',
            'prix': 'price_value',
            'price': 'price_value',
            'url': 'product_url',
            'link': 'product_url',
            'image': 'image_url',
            'img': 'image_url',
            'category': 'category',
            'categorie': 'category',
            'in_stock': 'in_stock',
            'stock': 'in_stock',
            'description': 'description',
            'specs': 'specifications_raw',
            'scraped_at': 'scraped_at',
            'date': 'scraped_at'
        }

        # Rename columns
        df_renamed = df.rename(columns=column_mapping)

        # Remove any 'id' column from source data to avoid conflicts
        if 'id' in df_renamed.columns:
            df_renamed = df_renamed.drop(columns=['id'])
            logger.info(f"   Removed 'id' column from {source}")

        # Ensure required columns
        required_columns = ['product_name', 'price_value', 'product_url']
        for col in required_columns:
            if col not in df_renamed.columns:
                logger.warning(f"  Missing column: {col} in {source}")
                df_renamed[col] = None

        # Add metadata
        df_renamed['source'] = source
        df_renamed['country'] = 'Tunisia' if source in ['mytek', 'tunisianet'] else 'France'
        df_renamed['currency'] = 'TND' if source in ['mytek', 'tunisianet'] else 'EUR'

        if 'scraped_at' not in df_renamed.columns or df_renamed['scraped_at'].isna().all():
            df_renamed['scraped_at'] = datetime.now().isoformat()

        logger.info(f" Standardized columns for {source}")
        return df_renamed

    def clean_price(self, price) -> Optional[float]:
        """Clean and normalize price values"""
        if pd.isna(price):
            return None

        price_str = str(price)

        # Remove currency symbols and text
        price_str = re.sub(r'[^\d.,\s-]', '', price_str)

        # Remove spaces
        price_str = price_str.replace(' ', '')

        # Handle European format
        if ',' in price_str and '.' not in price_str:
            price_str = price_str.replace(',', '.')
        elif ',' in price_str and '.' in price_str:
            if price_str.index(',') > price_str.index('.'):
                # European: 1.234,56
                price_str = price_str.replace('.', '').replace(',', '.')
            else:
                # US: 1,234.56
                price_str = price_str.replace(',', '')

        try:
            price_float = float(price_str)

            # Validation
            if price_float < 0 or price_float > 1_000_000:
                return None

            return round(price_float, 2)

        except ValueError:
            logger.warning(f"  Could not parse price: {price}")
            return None

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate products based on URL"""
        initial_count = len(df)

        # Sort by scraped_at descending
        df_sorted = df.sort_values('scraped_at', ascending=False)

        # Remove duplicates based on URL
        df_dedup = df_sorted.drop_duplicates(subset=['product_url'], keep='first')

        duplicates_removed = initial_count - len(df_dedup)
        self.stats['duplicates_removed'] += duplicates_removed

        if duplicates_removed > 0:
            logger.info(f"  Removed {duplicates_removed} duplicates")

        return df_dedup.reset_index(drop=True)

    def validate_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid products"""
        initial_count = len(df)

        # Remove products without name
        df = df[df['product_name'].notna() & (df['product_name'] != '')]

        # Remove products without valid price
        df = df[df['price_value'].notna() & (df['price_value'] > 0)]

        # Remove products without URL
        df = df[df['product_url'].notna() & (df['product_url'] != '')]

        invalid_removed = initial_count - len(df)
        self.stats['invalid_removed'] += invalid_removed

        if invalid_removed > 0:
            logger.info(f"  Removed {invalid_removed} invalid products")

        return df.reset_index(drop=True)

    def enrich_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich products with extracted data"""
        logger.info(" Enriching products...")

        enriched_data = []

        for idx, row in df.iterrows():
            product_name = row['product_name']

            # Normalize name
            normalized_name = self.normalizer.normalize_product_name(product_name)

            # Extract brand
            brand = self.normalizer.extract_brand(product_name)

            # Extract model
            model = self.normalizer.extract_model(product_name, brand)

            # Categorize
            category = self.normalizer.categorize_product(product_name)

            # Extract specs
            specs = self.normalizer.extract_specs(
                product_name,
                row.get('description', '') or ''
            )

            # Create signature
            signature = self.normalizer.create_signature(brand, model, specs)

            enriched_data.append({
                'product_name_normalized': normalized_name,
                'brand': brand,
                'model': model,
                'category': category,
                'specifications': json.dumps(specs),
                'product_signature': signature
            })

            if (idx + 1) % 100 == 0:
                logger.info(f"   Processed {idx + 1}/{len(df)} products")

        # Add enriched columns
        enriched_df = pd.DataFrame(enriched_data)
        df_enriched = pd.concat([df.reset_index(drop=True), enriched_df], axis=1)

        logger.info(f" Enrichment complete")
        return df_enriched

    def add_product_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add unique product IDs"""
        df['product_id'] = [f"prod_{i+1:05d}" for i in range(len(df))]

        # Reorder columns (product_id first)
        cols = ['product_id'] + [col for col in df.columns if col != 'product_id']
        return df[cols]

    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate column names"""
        initial_cols = len(df.columns)

        # Get duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()

        if duplicate_cols:
            logger.warning(f"  Found duplicate columns: {duplicate_cols}")

        # Keep only first occurrence of each column
        df = df.loc[:, ~df.columns.duplicated()]

        final_cols = len(df.columns)
        removed = initial_cols - final_cols

        if removed > 0:
            self.stats['duplicate_columns_removed'] = removed
            logger.info(f" Removed {removed} duplicate columns")
        else:
            logger.info(" No duplicate columns found")

        return df

    def consolidate_all(self) -> pd.DataFrame:
        """Main consolidation pipeline"""
        logger.info("="*60)
        logger.info(" STARTING PRODUCT CONSOLIDATION")
        logger.info("="*60)

        all_products = []
        sources = ['mytek', 'tunisianet', 'ldlc']

        # Step 1: Load all sources
        logger.info("\n STEP 1: Loading data from all sources")
        for source in sources:
            logger.info(f"\n--- Processing {source.upper()} ---")

            df = self.load_latest_csv(source)

            if df.empty:
                logger.warning(f"  No data for {source}, skipping...")
                continue

            # Standardize
            df = self.standardize_columns(df, source)

            # Clean prices
            df['price_value'] = df['price_value'].apply(self.clean_price)

            # Track stats
            self.stats['by_source'][source] = len(df)
            self.stats['total_loaded'] += len(df)

            all_products.append(df)

        if not all_products:
            logger.error(" No products loaded from any source!")
            return pd.DataFrame()

        # Step 2: Combine all sources
        logger.info("\n STEP 2: Combining all sources")
        df_combined = pd.concat(all_products, ignore_index=True)
        logger.info(f" Combined: {len(df_combined)} products")

        # Step 3: Remove duplicates
        logger.info("\n  STEP 3: Removing duplicates")
        df_dedup = self.remove_duplicates(df_combined)
        logger.info(f" After deduplication: {len(df_dedup)} products")

        # Step 4: Validate products
        logger.info("\n STEP 4: Validating products")
        df_valid = self.validate_products(df_dedup)
        logger.info(f" Valid products: {len(df_valid)}")

        # Step 5: Enrich products
        logger.info("\n STEP 5: Enriching products")
        df_enriched = self.enrich_products(df_valid)

        # Step 6: Add product IDs
        logger.info("\n STEP 6: Adding product IDs")
        df_final = self.add_product_ids(df_enriched)

        # Step 7: Remove duplicate columns (FIX)
        logger.info("\n STEP 7: Cleaning duplicate columns")
        df_final = self.remove_duplicate_columns(df_final)

        # Update final stats
        self.stats['final_count'] = len(df_final)

        return df_final

    def save_outputs(self, df: pd.DataFrame):
        """Save consolidated dataset in multiple formats"""
        logger.info("\n Saving outputs...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV
        csv_path = self.final_dir / "products_unified.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f" Saved CSV: {csv_path}")

        # Save JSON
        json_path = self.final_dir / "products_unified.json"
        df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        logger.info(f" Saved JSON: {json_path}")

        # Save timestamped backup
        backup_path = self.final_dir / f"products_unified_{timestamp}.csv"
        df.to_csv(backup_path, index=False, encoding='utf-8')
        logger.info(f" Saved backup: {backup_path}")

    def generate_report(self, df: pd.DataFrame):
        """Generate consolidation report"""
        logger.info("\n Generating consolidation report...")

        report_lines = []
        report_lines.append("="*60)
        report_lines.append("PRODUCT CONSOLIDATION REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*60)
        report_lines.append("")

        # Overall stats
        report_lines.append(" OVERALL STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total products loaded:     {self.stats['total_loaded']:,}")
        report_lines.append(f"Duplicates removed:        {self.stats['duplicates_removed']:,}")
        report_lines.append(f"Invalid entries removed:   {self.stats['invalid_removed']:,}")
        report_lines.append(f"Duplicate columns removed: {self.stats['duplicate_columns_removed']:,}")
        report_lines.append(f"Final product count:       {self.stats['final_count']:,}")
        retention_rate = (self.stats['final_count'] / self.stats['total_loaded'] * 100) if self.stats['total_loaded'] > 0 else 0
        report_lines.append(f"Retention rate:            {retention_rate:.1f}%")
        report_lines.append("")

        # By source
        report_lines.append(" BY SOURCE")
        report_lines.append("-" * 40)
        for source, count in self.stats['by_source'].items():
            pct = (count / self.stats['total_loaded'] * 100) if self.stats['total_loaded'] > 0 else 0
            report_lines.append(f"{source.capitalize():15} {count:6,} ({pct:5.1f}%)")
        report_lines.append("")

        # By country
        report_lines.append(" BY COUNTRY")
        report_lines.append("-" * 40)
        country_counts = df['country'].value_counts()
        for country, count in country_counts.items():
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"{country:15} {count:6,} ({pct:5.1f}%)")
        report_lines.append("")

        # By category
        report_lines.append(" BY CATEGORY")
        report_lines.append("-" * 40)
        category_counts = df['category'].value_counts()
        for category, count in category_counts.head(10).items():
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"{category:20} {count:6,} ({pct:5.1f}%)")
        report_lines.append("")

        # Price statistics
        report_lines.append(" PRICE STATISTICS")
        report_lines.append("-" * 40)

        # Tunisia prices
        tn_products = df[df['country'] == 'Tunisia']
        if not tn_products.empty:
            report_lines.append(f"\nTunisia (TND):")
            report_lines.append(f"  Count:        {len(tn_products):,}")
            report_lines.append(f"  Average:      {tn_products['price_value'].mean():,.2f} TND")
            report_lines.append(f"  Median:       {tn_products['price_value'].median():,.2f} TND")
            report_lines.append(f"  Min:          {tn_products['price_value'].min():,.2f} TND")
            report_lines.append(f"  Max:          {tn_products['price_value'].max():,.2f} TND")

        # France prices
        fr_products = df[df['country'] == 'France']
        if not fr_products.empty:
            report_lines.append(f"\nFrance (EUR):")
            report_lines.append(f"  Count:        {len(fr_products):,}")
            report_lines.append(f"  Average:      {fr_products['price_value'].mean():,.2f} EUR")
            report_lines.append(f"  Median:       {fr_products['price_value'].median():,.2f} EUR")
            report_lines.append(f"  Min:          {fr_products['price_value'].min():,.2f} EUR")
            report_lines.append(f"  Max:          {fr_products['price_value'].max():,.2f} EUR")
        report_lines.append("")

        # Data quality
        report_lines.append(" DATA QUALITY")
        report_lines.append("-" * 40)
        report_lines.append(f"Products with image:       {df['image_url'].notna().sum():,} ({df['image_url'].notna().mean()*100:.1f}%)")
        report_lines.append(f"Products with category:    {df['category'].notna().sum():,} ({df['category'].notna().mean()*100:.1f}%)")
        report_lines.append(f"Products with brand:       {df['brand'].notna().sum():,} ({df['brand'].notna().mean()*100:.1f}%)")
        report_lines.append("")

        # Top brands
        report_lines.append("  TOP BRANDS")
        report_lines.append("-" * 40)
        brand_counts = df['brand'].value_counts()
        for brand, count in brand_counts.head(10).items():
            if brand and brand != 'Unknown':
                report_lines.append(f"{brand:20} {count:6,}")
        report_lines.append("")

        report_lines.append("="*60)
        report_lines.append(" CONSOLIDATION COMPLETE")
        report_lines.append("="*60)

        # Print report (fix encoding)
        report_text = "\n".join(report_lines)
        try:
            print("\n" + report_text)
        except UnicodeEncodeError:
            # Fallback for Windows console
            print("\n" + report_text.encode('cp1252', errors='ignore').decode('cp1252'))

        # Save report
        report_path = self.final_dir / "consolidation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f" Report saved: {report_path}")

        # Save stats as JSON
        stats_path = self.final_dir / "consolidation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'by_country': country_counts.to_dict(),
                'by_category': category_counts.to_dict(),
                'top_brands': brand_counts.head(10).to_dict()
            }, f, indent=2, ensure_ascii=False)

        logger.info(f" Stats JSON saved: {stats_path}")


def main():
    """Main execution"""
    try:
        # Create consolidator
        consolidator = ProductConsolidator()

        # Run consolidation
        df_final = consolidator.consolidate_all()

        if df_final.empty:
            logger.error(" No products to save!")
            return

        # Save outputs
        consolidator.save_outputs(df_final)

        # Generate report
        consolidator.generate_report(df_final)

        logger.info("\n" + "="*60)
        logger.info(" SUCCESS! Product consolidation complete")
        logger.info("="*60)

    except Exception as e:
        logger.exception(f" Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()