"""
Script 4: Génération Metadata & Statistiques
--------------------------------------------
Objectif: Générer statistiques complètes et metadata pour tous les datasets

Author: PriceCheck TN Team
Date: 2026-01-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger.add(
    "logs/metadata_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class MetadataGenerator:
    """Generate comprehensive metadata and statistics"""

    def __init__(self):
        self.final_dir = Path("data/final")
        self.metadata_dir = self.final_dir / "metadata"
        self.viz_dir = self.metadata_dir / "visualizations"

        self.metadata_dir.mkdir(exist_ok=True, parents=True)
        self.viz_dir.mkdir(exist_ok=True, parents=True)

        # Global metadata
        self.metadata = {
            'generated_at': datetime.now().isoformat(),
            'project': 'PriceCheck TN',
            'version': '1.0.0',
            'datasets': {}
        }

        logger.info("🚀 MetadataGenerator initialized")

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all final datasets"""
        logger.info("\n📂 Loading all datasets...")

        datasets = {}

        files = {
            'products': 'products_unified.csv',
            'price_comparison': 'price_comparison.csv',
            'reviews': 'reviews_with_predictions.csv'
        }

        for name, filename in files.items():
            filepath = self.final_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, encoding='utf-8')
                datasets[name] = df
                logger.info(f"✅ Loaded {name}: {len(df)} rows")
            else:
                logger.warning(f"⚠️  {name} not found: {filepath}")
                datasets[name] = pd.DataFrame()

        return datasets

    def generate_products_stats(self, df: pd.DataFrame) -> Dict:
        """Generate products statistics"""
        logger.info("\n📦 Generating products statistics...")

        stats = {
            'total_products': len(df),
            'by_country': {},
            'by_source': {},
            'by_category': {},
            'by_brand': {},
            'price_stats': {},
            'data_quality': {}
        }

        if df.empty:
            return stats

        # By country
        if 'country' in df.columns:
            stats['by_country'] = df['country'].value_counts().to_dict()

        # By source
        if 'source' in df.columns:
            stats['by_source'] = df['source'].value_counts().to_dict()

        # By category
        if 'category' in df.columns:
            stats['by_category'] = df['category'].value_counts().head(10).to_dict()

        # By brand
        if 'brand' in df.columns:
            stats['by_brand'] = df['brand'].value_counts().head(10).to_dict()

        # Price statistics
        if 'price_value' in df.columns:
            tn_products = df[df['country'] == 'Tunisia']
            fr_products = df[df['country'] == 'France']

            if not tn_products.empty:
                stats['price_stats']['tunisia_tnd'] = {
                    'count': len(tn_products),
                    'mean': float(tn_products['price_value'].mean()),
                    'median': float(tn_products['price_value'].median()),
                    'min': float(tn_products['price_value'].min()),
                    'max': float(tn_products['price_value'].max()),
                    'std': float(tn_products['price_value'].std())
                }

            if not fr_products.empty:
                stats['price_stats']['france_eur'] = {
                    'count': len(fr_products),
                    'mean': float(fr_products['price_value'].mean()),
                    'median': float(fr_products['price_value'].median()),
                    'min': float(fr_products['price_value'].min()),
                    'max': float(fr_products['price_value'].max()),
                    'std': float(fr_products['price_value'].std())
                }

        # Data quality
        if 'image_url' in df.columns:
            stats['data_quality']['with_image'] = int(df['image_url'].notna().sum())
            stats['data_quality']['with_image_pct'] = float(df['image_url'].notna().mean() * 100)

        if 'category' in df.columns:
            stats['data_quality']['with_category'] = int(df['category'].notna().sum())
            stats['data_quality']['with_category_pct'] = float(df['category'].notna().mean() * 100)

        if 'brand' in df.columns:
            stats['data_quality']['with_brand'] = int(df['brand'].notna().sum())
            stats['data_quality']['with_brand_pct'] = float(df['brand'].notna().mean() * 100)

        logger.info(f"✅ Products stats generated")
        return stats

    def generate_price_comparison_stats(self, df: pd.DataFrame) -> Dict:
        """Generate price comparison statistics"""
        logger.info("\n💰 Generating price comparison statistics...")

        stats = {
            'total_matches': len(df),
            'price_differences': {},
            'cheaper_country': {},
            'by_category': {},
            'top_savings': []
        }

        if df.empty:
            return stats

        # Price differences
        if 'price_diff_pct' in df.columns:
            stats['price_differences'] = {
                'mean': float(df['price_diff_pct'].mean()),
                'median': float(df['price_diff_pct'].median()),
                'min': float(df['price_diff_pct'].min()),
                'max': float(df['price_diff_pct'].max()),
                'std': float(df['price_diff_pct'].std())
            }

        # Cheaper country
        if 'cheaper_country' in df.columns:
            stats['cheaper_country'] = df['cheaper_country'].value_counts().to_dict()

        # By category
        if 'price_diff_pct' in df.columns:
            # Try to get category from tn_product_name or similar
            category_col = None
            for col in df.columns:
                if 'category' in col.lower():
                    category_col = col
                    break

            if category_col:
                category_stats = df.groupby(category_col)['price_diff_pct'].agg(['count', 'mean']).to_dict('index')
                stats['by_category'] = {k: {'count': int(v['count']), 'avg_diff_pct': float(v['mean'])}
                                        for k, v in category_stats.items()}

        # Top savings
        if 'savings_eur' in df.columns and 'tn_product_name' in df.columns:
            top_10 = df.nlargest(10, 'savings_eur')
            stats['top_savings'] = [
                {
                    'product': row['tn_product_name'][:60],
                    'savings_eur': float(row['savings_eur']),
                    'cheaper_in': row.get('cheaper_country', 'N/A')
                }
                for _, row in top_10.iterrows()
            ]

        logger.info(f"✅ Price comparison stats generated")
        return stats

    def generate_reviews_stats(self, df: pd.DataFrame) -> Dict:
        """Generate reviews statistics"""
        logger.info("\n💬 Generating reviews statistics...")

        stats = {
            'total_reviews': len(df),
            'fake_detection': {},
            'by_rating': {},
            'by_language': {},
            'sentiment': {},
            'confidence_levels': {},
            'model_performance': {}
        }

        if df.empty:
            return stats

        # Fake detection
        if 'is_fake' in df.columns:
            fake_count = int(df['is_fake'].sum())
            stats['fake_detection'] = {
                'fake_count': fake_count,
                'fake_pct': float(fake_count / len(df) * 100),
                'real_count': int(len(df) - fake_count),
                'real_pct': float((len(df) - fake_count) / len(df) * 100)
            }

        # By rating
        if 'rating' in df.columns:
            stats['by_rating'] = df['rating'].value_counts().sort_index().to_dict()

            # Fake by rating
            if 'is_fake' in df.columns:
                fake_by_rating = df[df['is_fake'] == True].groupby('rating').size().to_dict()
                stats['fake_by_rating'] = fake_by_rating

        # By language
        if 'language' in df.columns:
            stats['by_language'] = df['language'].value_counts().to_dict()

        # Sentiment
        if 'sentiment' in df.columns:
            stats['sentiment'] = df['sentiment'].value_counts().to_dict()

        # Confidence levels
        if 'confidence' in df.columns:
            stats['confidence_levels'] = df['confidence'].value_counts().to_dict()

        # Model performance
        if 'xgboost_score' in df.columns:
            stats['model_performance']['xgboost'] = {
                'mean_score': float(df['xgboost_score'].mean()),
                'median_score': float(df['xgboost_score'].median()),
                'std_score': float(df['xgboost_score'].std())
            }

        if 'bert_score' in df.columns:
            stats['model_performance']['bert'] = {
                'mean_score': float(df['bert_score'].mean()),
                'median_score': float(df['bert_score'].median()),
                'std_score': float(df['bert_score'].std())
            }

        if 'ensemble_score' in df.columns:
            stats['model_performance']['ensemble'] = {
                'mean_score': float(df['ensemble_score'].mean()),
                'median_score': float(df['ensemble_score'].median()),
                'std_score': float(df['ensemble_score'].std())
            }

        logger.info(f"✅ Reviews stats generated")
        return stats

    def create_visualizations(self, datasets: Dict[str, pd.DataFrame]):
        """Create visualizations"""
        logger.info("\n📊 Creating visualizations...")

        plt.style.use('seaborn-v0_8-darkgrid')

        # 1. Products by country
        if not datasets['products'].empty and 'country' in datasets['products'].columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            country_counts = datasets['products']['country'].value_counts()
            ax.bar(country_counts.index, country_counts.values, color=['#2ecc71', '#3498db'])
            ax.set_title('Products by Country', fontsize=16, fontweight='bold')
            ax.set_xlabel('Country')
            ax.set_ylabel('Number of Products')
            for i, v in enumerate(country_counts.values):
                ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.viz_dir / 'products_by_country.png', dpi=300)
            plt.close()
            logger.info("✅ Created: products_by_country.png")

        # 2. Price distribution
        if not datasets['products'].empty and 'price_value' in datasets['products'].columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            tn_prices = datasets['products'][datasets['products']['country'] == 'Tunisia']['price_value']
            fr_prices = datasets['products'][datasets['products']['country'] == 'France']['price_value']

            if not tn_prices.empty:
                axes[0].hist(tn_prices, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
                axes[0].set_title('Tunisia Prices (TND)', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Price (TND)')
                axes[0].set_ylabel('Frequency')

            if not fr_prices.empty:
                axes[1].hist(fr_prices, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
                axes[1].set_title('France Prices (EUR)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Price (EUR)')
                axes[1].set_ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(self.viz_dir / 'price_distribution.png', dpi=300)
            plt.close()
            logger.info("✅ Created: price_distribution.png")

        # 3. Fake reviews by rating
        if not datasets['reviews'].empty and 'rating' in datasets['reviews'].columns and 'is_fake' in datasets[
            'reviews'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            fake_by_rating = datasets['reviews'][datasets['reviews']['is_fake'] == True].groupby('rating').size()
            real_by_rating = datasets['reviews'][datasets['reviews']['is_fake'] == False].groupby('rating').size()

            x = np.arange(1, 6)
            width = 0.35

            ax.bar(x - width / 2, [real_by_rating.get(i, 0) for i in x], width, label='Real', color='#2ecc71')
            ax.bar(x + width / 2, [fake_by_rating.get(i, 0) for i in x], width, label='Fake', color='#e74c3c')

            ax.set_title('Reviews by Rating (Real vs Fake)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_xticks(x)
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.viz_dir / 'fake_reviews_by_rating.png', dpi=300)
            plt.close()
            logger.info("✅ Created: fake_reviews_by_rating.png")

        # 4. Price difference histogram
        if not datasets['price_comparison'].empty and 'price_diff_pct' in datasets['price_comparison'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(datasets['price_comparison']['price_diff_pct'], bins=20,
                    color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
            ax.set_title('Price Difference Distribution (TN vs FR)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Price Difference (%)')
            ax.set_ylabel('Frequency')
            ax.legend()

            plt.tight_layout()
            plt.savefig(self.viz_dir / 'price_diff_histogram.png', dpi=300)
            plt.close()
            logger.info("✅ Created: price_diff_histogram.png")

        logger.info("✅ All visualizations created")

    def generate_summary_report(self, all_stats: Dict):
        """Generate comprehensive summary report"""
        logger.info("\n📝 Generating summary report...")

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("PRICECHECK TN - COMPREHENSIVE DATA REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)
        report_lines.append("")

        # Products section
        if 'products' in all_stats:
            prod_stats = all_stats['products']
            report_lines.append("📦 PRODUCTS OVERVIEW")
            report_lines.append("-" * 70)
            report_lines.append(f"Total Products:              {prod_stats.get('total_products', 0):,}")

            if 'by_country' in prod_stats:
                report_lines.append("\nBy Country:")
                for country, count in prod_stats['by_country'].items():
                    pct = count / prod_stats['total_products'] * 100 if prod_stats['total_products'] > 0 else 0
                    report_lines.append(f"  {country:15} {count:6,} ({pct:5.1f}%)")

            if 'price_stats' in prod_stats and 'tunisia_tnd' in prod_stats['price_stats']:
                tn_stats = prod_stats['price_stats']['tunisia_tnd']
                report_lines.append(f"\nTunisia Prices (TND):")
                report_lines.append(f"  Average:               {tn_stats['mean']:,.2f} TND")
                report_lines.append(f"  Median:                {tn_stats['median']:,.2f} TND")
                report_lines.append(f"  Range:                 {tn_stats['min']:,.2f} - {tn_stats['max']:,.2f} TND")

            if 'price_stats' in prod_stats and 'france_eur' in prod_stats['price_stats']:
                fr_stats = prod_stats['price_stats']['france_eur']
                report_lines.append(f"\nFrance Prices (EUR):")
                report_lines.append(f"  Average:               {fr_stats['mean']:,.2f} EUR")
                report_lines.append(f"  Median:                {fr_stats['median']:,.2f} EUR")
                report_lines.append(f"  Range:                 {fr_stats['min']:,.2f} - {fr_stats['max']:,.2f} EUR")

            report_lines.append("")

        # Price comparison section
        if 'price_comparison' in all_stats:
            comp_stats = all_stats['price_comparison']
            report_lines.append("💰 PRICE COMPARISON ANALYSIS")
            report_lines.append("-" * 70)
            report_lines.append(f"Total Matches Found:         {comp_stats.get('total_matches', 0):,}")

            if 'price_differences' in comp_stats:
                diff_stats = comp_stats['price_differences']
                report_lines.append(f"\nPrice Differences:")
                report_lines.append(f"  Average Difference:    {diff_stats.get('mean', 0):+.1f}%")
                report_lines.append(f"  Median Difference:     {diff_stats.get('median', 0):+.1f}%")
                report_lines.append(
                    f"  Range:                 {diff_stats.get('min', 0):+.1f}% to {diff_stats.get('max', 0):+.1f}%")

            if 'cheaper_country' in comp_stats:
                report_lines.append(f"\nCheaper Country:")
                for country, count in comp_stats['cheaper_country'].items():
                    pct = count / comp_stats['total_matches'] * 100 if comp_stats['total_matches'] > 0 else 0
                    report_lines.append(f"  {country:15} {count:6,} ({pct:5.1f}%)")

            report_lines.append("")

        # Reviews section
        if 'reviews' in all_stats:
            rev_stats = all_stats['reviews']
            report_lines.append("💬 REVIEWS ANALYSIS")
            report_lines.append("-" * 70)
            report_lines.append(f"Total Reviews:               {rev_stats.get('total_reviews', 0):,}")

            if 'fake_detection' in rev_stats:
                fake_stats = rev_stats['fake_detection']
                report_lines.append(f"\nFake Detection Results:")
                report_lines.append(
                    f"  Fake Reviews:          {fake_stats.get('fake_count', 0):,} ({fake_stats.get('fake_pct', 0):.1f}%)")
                report_lines.append(
                    f"  Real Reviews:          {fake_stats.get('real_count', 0):,} ({fake_stats.get('real_pct', 0):.1f}%)")

            if 'confidence_levels' in rev_stats:
                report_lines.append(f"\nConfidence Levels:")
                for level, count in rev_stats['confidence_levels'].items():
                    pct = count / rev_stats['total_reviews'] * 100 if rev_stats['total_reviews'] > 0 else 0
                    report_lines.append(f"  {level.capitalize():15} {count:6,} ({pct:5.1f}%)")

            report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("✅ END OF REPORT")
        report_lines.append("=" * 70)

        # Print and save
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        report_path = self.metadata_dir / "full_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"✅ Summary report saved: {report_path}")

    def run(self):
        """Main execution"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING METADATA GENERATION")
        logger.info("=" * 60)

        # Load datasets
        datasets = self.load_datasets()

        # Generate stats for each dataset
        all_stats = {}

        logger.info("\n📊 Generating statistics...")
        all_stats['products'] = self.generate_products_stats(datasets['products'])
        all_stats['price_comparison'] = self.generate_price_comparison_stats(datasets['price_comparison'])
        all_stats['reviews'] = self.generate_reviews_stats(datasets['reviews'])

        # Add to metadata
        self.metadata['datasets'] = all_stats

        # Create visualizations
        self.create_visualizations(datasets)

        # Save metadata JSON
        logger.info("\n💾 Saving metadata...")
        metadata_path = self.metadata_dir / "full_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Metadata saved: {metadata_path}")

        # Generate summary report
        self.generate_summary_report(all_stats)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 SUCCESS! Metadata generation complete")
        logger.info("=" * 60)


def main():
    """Main execution"""
    try:
        generator = MetadataGenerator()
        generator.run()

    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()