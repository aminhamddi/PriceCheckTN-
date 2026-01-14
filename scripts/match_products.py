"""
Script 2: Matching Produits Tunisia ↔ France
--------------------------------------------
Objectif: Matcher les produits tunisiens avec les français
         pour comparaison de prix

Author: PriceCheck TN Team
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from loguru import logger
import sys
from fuzzywuzzy import fuzz
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.fuzzy_matcher import ProductMatcher
from utils.currency_converter import CurrencyConverter

# Configure logger
logger.add(
    "logs/matching_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class ProductMatchingPipeline:
    """Match Tunisia products with France products for price comparison"""

    def __init__(self):
        self.final_dir = Path("data/final")

        self.matcher = ProductMatcher()
        self.currency_converter = CurrencyConverter()

        # Statistics
        self.stats = {
            'total_tn_products': 0,
            'total_fr_products': 0,
            'total_matches': 0,
            'match_rate': 0.0,
            'high_confidence_matches': 0,
            'medium_confidence_matches': 0,
            'low_confidence_matches': 0,
            'avg_price_diff_pct': 0.0,
            'cheaper_in_tn': 0,
            'cheaper_in_fr': 0
        }

        logger.info(" ProductMatchingPipeline initialized")

    def load_products(self) -> pd.DataFrame:
        """Load consolidated products"""
        products_file = self.final_dir / "products_unified.csv"

        if not products_file.exists():
            logger.error(f" Products file not found: {products_file}")
            logger.error(" Run script 1 first: python scripts/1_consolidate_products.py")
            raise FileNotFoundError(f"Missing: {products_file}")

        df = pd.read_csv(products_file, encoding='utf-8')
        logger.info(f" Loaded {len(df)} products")

        return df

    def split_by_country(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split products by country"""
        tn_products = df[df['country'] == 'Tunisia'].copy()
        fr_products = df[df['country'] == 'France'].copy()

        self.stats['total_tn_products'] = len(tn_products)
        self.stats['total_fr_products'] = len(fr_products)

        logger.info(f"🇹🇳 Tunisia products: {len(tn_products)}")
        logger.info(f"🇫🇷 France products: {len(fr_products)}")

        return tn_products, fr_products

    def match_products(self, tn_products: pd.DataFrame,
                       fr_products: pd.DataFrame) -> List[Dict]:
        """
        Match Tunisia products with France products
        Uses multi-level matching strategy
        """
        logger.info("\n Starting product matching...")

        matches = []

        for idx, tn_product in tn_products.iterrows():
            if (idx + 1) % 50 == 0:
                logger.info(f"   Processed {idx + 1}/{len(tn_products)} TN products")

            # Get best match
            best_match = self.matcher.find_best_match(
                tn_product,
                fr_products
            )

            if best_match:
                matches.append(best_match)

        logger.info(f" Found {len(matches)} matches")
        return matches

    def calculate_price_differences(self, matches: List[Dict]) -> List[Dict]:
        """
        Calculate price differences and add comparison data
        """
        logger.info("\n Calculating price differences...")

        enriched_matches = []

        for match in matches:
            try:
                # Get exchange rate
                exchange_rate = self.currency_converter.get_rate('TND', 'EUR')

                # Convert TN price to EUR
                tn_price_tnd = match['tn_product']['price_value']
                tn_price_eur = self.currency_converter.convert(
                    tn_price_tnd, 'TND', 'EUR'
                )

                fr_price_eur = match['fr_product']['price_value']

                # Calculate differences
                price_diff_eur = tn_price_eur - fr_price_eur
                price_diff_pct = ((tn_price_eur - fr_price_eur) / fr_price_eur) * 100

                # Determine cheaper country
                cheaper_country = 'France' if price_diff_eur > 0 else 'Tunisia'
                savings_eur = abs(price_diff_eur)
                savings_tnd = self.currency_converter.convert(savings_eur, 'EUR', 'TND')

                # Enrich match
                match['tn_product']['price_eur'] = round(tn_price_eur, 2)

                match['comparison'] = {
                    'price_diff_eur': round(price_diff_eur, 2),
                    'price_diff_pct': round(price_diff_pct, 2),
                    'cheaper_country': cheaper_country,
                    'savings_eur': round(savings_eur, 2),
                    'savings_tnd': round(savings_tnd, 2)
                }

                match['exchange_rate'] = {
                    'rate': exchange_rate,
                    'source': 'exchangerate.host',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }

                # Update stats
                if cheaper_country == 'Tunisia':
                    self.stats['cheaper_in_tn'] += 1
                else:
                    self.stats['cheaper_in_fr'] += 1

                enriched_matches.append(match)

            except Exception as e:
                logger.error(f" Error calculating price for match: {e}")
                continue

        logger.info(f" Calculated prices for {len(enriched_matches)} matches")
        return enriched_matches

    def categorize_by_confidence(self, matches: List[Dict]):
        """Categorize matches by confidence level"""
        for match in matches:
            confidence = match['match_quality']['confidence']

            if confidence == 'high':
                self.stats['high_confidence_matches'] += 1
            elif confidence == 'medium':
                self.stats['medium_confidence_matches'] += 1
            else:
                self.stats['low_confidence_matches'] += 1

    def save_matches(self, matches: List[Dict]):
        """Save matched products"""
        logger.info("\n Saving matches...")

        # Convert to DataFrame
        df_matches = pd.DataFrame(matches)

        # Save CSV (flattened)
        csv_data = []
        for match in matches:
            row = {
                'match_id': match['match_id'],
                'tn_product_id': match['tn_product']['product_id'],
                'tn_product_name': match['tn_product']['product_name'],
                'tn_price_tnd': match['tn_product']['price_value'],
                'tn_price_eur': match['tn_product'].get('price_eur'),
                'tn_source': match['tn_product']['source'],
                'tn_url': match['tn_product']['product_url'],
                'fr_product_id': match['fr_product']['product_id'],
                'fr_product_name': match['fr_product']['product_name'],
                'fr_price_eur': match['fr_product']['price_value'],
                'fr_source': match['fr_product']['source'],
                'fr_url': match['fr_product']['product_url'],
                'price_diff_eur': match['comparison']['price_diff_eur'],
                'price_diff_pct': match['comparison']['price_diff_pct'],
                'cheaper_country': match['comparison']['cheaper_country'],
                'savings_eur': match['comparison']['savings_eur'],
                'savings_tnd': match['comparison']['savings_tnd'],
                'match_score': match['match_quality']['match_score'],
                'confidence': match['match_quality']['confidence'],
                'match_method': match['match_quality']['match_method'],
                'exchange_rate': match['exchange_rate']['rate']
            }
            csv_data.append(row)

        df_csv = pd.DataFrame(csv_data)

        csv_path = self.final_dir / "price_comparison.csv"
        df_csv.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f" Saved CSV: {csv_path}")

        # Save JSON (full structure)
        json_path = self.final_dir / "price_comparison.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        logger.info(f" Saved JSON: {json_path}")

    def save_unmatched(self, tn_products: pd.DataFrame,
                       fr_products: pd.DataFrame,
                       matched_tn_ids: List[str],
                       matched_fr_ids: List[str]):
        """Save unmatched products"""
        logger.info("\n Saving unmatched products...")

        # Unmatched TN products
        unmatched_tn = tn_products[~tn_products['product_id'].isin(matched_tn_ids)]
        unmatched_tn_path = self.final_dir / "unmatched_tn_products.csv"
        unmatched_tn.to_csv(unmatched_tn_path, index=False, encoding='utf-8')
        logger.info(f" Saved {len(unmatched_tn)} unmatched TN products")

        # Unmatched FR products
        unmatched_fr = fr_products[~fr_products['product_id'].isin(matched_fr_ids)]
        unmatched_fr_path = self.final_dir / "unmatched_fr_products.csv"
        unmatched_fr.to_csv(unmatched_fr_path, index=False, encoding='utf-8')
        logger.info(f" Saved {len(unmatched_fr)} unmatched FR products")

    def generate_report(self, matches: List[Dict]):
        """Generate matching report"""
        logger.info("\n Generating matching report...")

        # Calculate final stats
        if matches:
            price_diffs = [m['comparison']['price_diff_pct'] for m in matches]
            self.stats['avg_price_diff_pct'] = np.mean(price_diffs)

        self.stats['total_matches'] = len(matches)

        if self.stats['total_tn_products'] > 0:
            self.stats['match_rate'] = (
                    self.stats['total_matches'] / self.stats['total_tn_products'] * 100
            )

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PRODUCT MATCHING REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")

        report_lines.append(" OVERALL STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total TN products:         {self.stats['total_tn_products']:,}")
        report_lines.append(f"Total FR products:         {self.stats['total_fr_products']:,}")
        report_lines.append(f"Total matches found:       {self.stats['total_matches']:,}")
        report_lines.append(f"Match rate:                {self.stats['match_rate']:.1f}%")
        report_lines.append("")

        report_lines.append(" MATCH QUALITY")
        report_lines.append("-" * 40)
        report_lines.append(f"High confidence (>0.85):   {self.stats['high_confidence_matches']:,}")
        report_lines.append(f"Medium confidence (0.70-0.85): {self.stats['medium_confidence_matches']:,}")
        report_lines.append(f"Low confidence (<0.70):    {self.stats['low_confidence_matches']:,}")
        report_lines.append("")

        report_lines.append(" PRICE COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append(f"Average price difference:  {self.stats['avg_price_diff_pct']:+.1f}%")
        report_lines.append(f"Cheaper in Tunisia:        {self.stats['cheaper_in_tn']:,}")
        report_lines.append(f"Cheaper in France:         {self.stats['cheaper_in_fr']:,}")
        report_lines.append("")

        # Top 10 best deals
        if matches:
            report_lines.append(" TOP 10 BEST DEALS (Biggest Savings)")
            report_lines.append("-" * 40)

            sorted_matches = sorted(
                matches,
                key=lambda x: x['comparison']['savings_eur'],
                reverse=True
            )

            for i, match in enumerate(sorted_matches[:10], 1):
                tn_name = match['tn_product']['product_name'][:40]
                savings = match['comparison']['savings_eur']
                cheaper = match['comparison']['cheaper_country']

                report_lines.append(
                    f"{i:2}. {tn_name:40} | Save {savings:7.2f}€ in {cheaper}"
                )
            report_lines.append("")

        # Category breakdown
        if matches:
            report_lines.append(" BY CATEGORY")
            report_lines.append("-" * 40)

            category_stats = {}
            for match in matches:
                category = match['tn_product'].get('category', 'Unknown')
                if category not in category_stats:
                    category_stats[category] = {
                        'count': 0,
                        'total_diff': 0
                    }
                category_stats[category]['count'] += 1
                category_stats[category]['total_diff'] += match['comparison']['price_diff_pct']

            for category, stats in sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                avg_diff = stats['total_diff'] / stats['count']
                report_lines.append(f"{category:20} {stats['count']:4} matches | Avg diff: {avg_diff:+6.1f}%")
            report_lines.append("")

        report_lines.append("=" * 60)
        report_lines.append(" MATCHING COMPLETE")
        report_lines.append("=" * 60)

        # Print report
        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        # Save report
        report_path = self.final_dir / "matching_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f" Report saved: {report_path}")

        # Save stats JSON
        stats_path = self.final_dir / "matching_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f" Stats saved: {stats_path}")

    def run(self):
        """Main pipeline execution"""
        logger.info("=" * 60)
        logger.info(" STARTING PRODUCT MATCHING")
        logger.info("=" * 60)

        # Step 1: Load products
        logger.info("\n STEP 1: Loading products")
        df_products = self.load_products()

        # Step 2: Split by country
        logger.info("\n STEP 2: Splitting by country")
        tn_products, fr_products = self.split_by_country(df_products)

        # Step 3: Match products
        logger.info("\n STEP 3: Matching products")
        matches = self.match_products(tn_products, fr_products)

        if not matches:
            logger.warning("  No matches found!")
            # Create empty files to satisfy DVC
            empty_csv = self.final_dir / "price_comparison.csv"
            pd.DataFrame().to_csv(empty_csv, index=False)
            empty_json = self.final_dir / "price_comparison.json"
            with open(empty_json, 'w') as f:
                json.dump([], f)
            unmatched_tn = self.final_dir / "unmatched_tn_products.csv"
            tn_products.to_csv(unmatched_tn, index=False)
            unmatched_fr = self.final_dir / "unmatched_fr_products.csv"
            fr_products.to_csv(unmatched_fr, index=False)
            return

        # Step 4: Calculate prices
        logger.info("\n STEP 4: Calculating price differences")
        matches = self.calculate_price_differences(matches)

        # Step 5: Categorize by confidence
        logger.info("\n STEP 5: Categorizing matches")
        self.categorize_by_confidence(matches)

        # Step 6: Save matches
        logger.info("\n STEP 6: Saving results")
        self.save_matches(matches)

        # Step 7: Save unmatched
        matched_tn_ids = [m['tn_product']['product_id'] for m in matches]
        matched_fr_ids = [m['fr_product']['product_id'] for m in matches]
        self.save_unmatched(tn_products, fr_products, matched_tn_ids, matched_fr_ids)

        # Step 8: Generate report
        logger.info("\n STEP 8: Generating report")
        self.generate_report(matches)

        logger.info("\n" + "=" * 60)
        logger.info(" SUCCESS! Product matching complete")
        logger.info("=" * 60)


def main():
    """Main execution"""
    try:
        pipeline = ProductMatchingPipeline()
        pipeline.run()

    except Exception as e:
        logger.exception(f" Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()