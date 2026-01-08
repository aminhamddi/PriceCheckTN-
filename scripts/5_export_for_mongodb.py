"""
Script 5: Export for MongoDB
----------------------------
Objectif: Exporter tous les datasets en format MongoDB-ready
         Créer collections JSON avec schema approprié

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
import hashlib

# Configure logger
logger.add(
    "logs/mongodb_export_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO"
)


class MongoDBExporter:
    """Export datasets to MongoDB-ready JSON format"""

    def __init__(self):
        self.final_dir = Path("data/final")
        self.mongodb_dir = self.final_dir / "mongodb"
        self.mongodb_dir.mkdir(exist_ok=True, parents=True)

        self.stats = {
            'collections_created': 0,
            'total_documents': 0,
            'by_collection': {}
        }

        logger.info("🚀 MongoDBExporter initialized")

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all final datasets"""
        logger.info("\n📂 Loading datasets...")

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

    def generate_object_id(self, prefix: str, index: int) -> str:
        """Generate MongoDB-like ObjectId"""
        # Simple deterministic ID generation
        return f"{prefix}_{index:06d}"

    def clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj

    def export_users_collection(self) -> List[Dict]:
        """Create sample users collection"""
        logger.info("\n👥 Creating Users collection...")

        users = [
            {
                '_id': self.generate_object_id('user', 1),
                'email': 'admin@pricecheck.tn',
                'password': '$2b$10$dummyhash',  # Hashed password (example)
                'role': 'admin',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('user', 2),
                'email': 'user@example.com',
                'password': '$2b$10$dummyhash',
                'role': 'user',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        ]

        logger.info(f"✅ Created {len(users)} sample users")
        return users

    def export_user_profiles_collection(self) -> List[Dict]:
        """Create user profiles (1-to-1 with User)"""
        logger.info("\n📋 Creating UserProfiles collection...")

        profiles = [
            {
                '_id': self.generate_object_id('profile', 1),
                'user_id': self.generate_object_id('user', 1),
                'full_name': 'Admin User',
                'location': 'Tunis, Tunisia',
                'language': 'fr',
                'currency_preference': 'TND',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('profile', 2),
                'user_id': self.generate_object_id('user', 2),
                'full_name': 'Regular User',
                'location': 'Paris, France',
                'language': 'fr',
                'currency_preference': 'EUR',
                'created_at': datetime.now().isoformat()
            }
        ]

        logger.info(f"✅ Created {len(profiles)} user profiles")
        return profiles

    def export_categories_collection(self) -> List[Dict]:
        """Create categories collection"""
        logger.info("\n🏷️  Creating Categories collection...")

        categories = [
            {
                '_id': self.generate_object_id('cat', 1),
                'name': 'Laptop',
                'name_fr': 'Ordinateurs Portables',
                'name_ar': 'حواسيب محمولة',
                'slug': 'laptop',
                'icon': '💻',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 2),
                'name': 'Desktop',
                'name_fr': 'Ordinateurs de Bureau',
                'name_ar': 'حواسيب مكتبية',
                'slug': 'desktop',
                'icon': '🖥️',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 3),
                'name': 'Monitor',
                'name_fr': 'Écrans',
                'name_ar': 'شاشات',
                'slug': 'monitor',
                'icon': '🖥️',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 4),
                'name': 'Graphics Card',
                'name_fr': 'Cartes Graphiques',
                'name_ar': 'بطاقات الرسومات',
                'slug': 'graphics-card',
                'icon': '🎮',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 5),
                'name': 'Storage',
                'name_fr': 'Stockage',
                'name_ar': 'تخزين',
                'slug': 'storage',
                'icon': '💾',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 6),
                'name': 'Keyboard',
                'name_fr': 'Claviers',
                'name_ar': 'لوحات مفاتيح',
                'slug': 'keyboard',
                'icon': '⌨️',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 7),
                'name': 'Mouse',
                'name_fr': 'Souris',
                'name_ar': 'فأرة',
                'slug': 'mouse',
                'icon': '🖱️',
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('cat', 8),
                'name': 'Other',
                'name_fr': 'Autre',
                'name_ar': 'أخرى',
                'slug': 'other',
                'icon': '📦',
                'created_at': datetime.now().isoformat()
            }
        ]

        logger.info(f"✅ Created {len(categories)} categories")
        return categories

    def export_stores_collection(self) -> List[Dict]:
        """Create stores collection"""
        logger.info("\n🏪 Creating Stores collection...")

        stores = [
            {
                '_id': self.generate_object_id('store', 1),
                'name': 'MyTek',
                'slug': 'mytek',
                'country': 'Tunisia',
                'url': 'https://www.mytek.tn',
                'logo': None,
                'trust_score': 4.2,
                'total_reviews': 1523,
                'avg_delivery_days': 3,
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('store', 2),
                'name': 'Tunisianet',
                'slug': 'tunisianet',
                'country': 'Tunisia',
                'url': 'https://www.tunisianet.com.tn',
                'logo': None,
                'trust_score': 4.5,
                'total_reviews': 2845,
                'avg_delivery_days': 2,
                'created_at': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('store', 3),
                'name': 'LDLC',
                'slug': 'ldlc',
                'country': 'France',
                'url': 'https://www.ldlc.com',
                'logo': None,
                'trust_score': 4.7,
                'total_reviews': 15234,
                'avg_delivery_days': 5,
                'created_at': datetime.now().isoformat()
            }
        ]

        logger.info(f"✅ Created {len(stores)} stores")
        return stores

    def export_products_collection(self, df: pd.DataFrame, categories: List[Dict]) -> List[Dict]:
        """Convert products DataFrame to MongoDB collection"""
        logger.info("\n📦 Creating Products collection...")

        if df.empty:
            logger.warning("⚠️  No products to export")
            return []

        # Create category lookup
        category_map = {cat['name']: cat['_id'] for cat in categories}

        products = []

        for idx, row in df.iterrows():
            # Get category ID
            category_name = self.clean_for_json(row.get('category', 'Other'))
            category_id = category_map.get(category_name, category_map['Other'])

            # Parse specifications
            specs_str = row.get('specifications', '{}')
            try:
                specs = json.loads(specs_str) if isinstance(specs_str, str) else {}
            except:
                specs = {}

            product = {
                '_id': self.clean_for_json(row.get('product_id', self.generate_object_id('prod', idx))),
                'name': self.clean_for_json(row.get('product_name', '')),
                'name_normalized': self.clean_for_json(row.get('product_name_normalized', '')),
                'brand': self.clean_for_json(row.get('brand', 'Unknown')),
                'model': self.clean_for_json(row.get('model')),
                'category_id': category_id,
                'price': self.clean_for_json(row.get('price_value')),
                'currency': self.clean_for_json(row.get('currency', 'TND')),
                'country': self.clean_for_json(row.get('country', 'Tunisia')),
                'source': self.clean_for_json(row.get('source', '')),
                'url': self.clean_for_json(row.get('product_url', '')),
                'image_url': self.clean_for_json(row.get('image_url')),
                'specifications': specs,
                'product_signature': self.clean_for_json(row.get('product_signature', '')),
                'in_stock': True,
                'scraped_at': self.clean_for_json(row.get('scraped_at', datetime.now().isoformat())),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            products.append(product)

            if (idx + 1) % 100 == 0:
                logger.info(f"   Processed {idx + 1}/{len(df)} products")

        logger.info(f"✅ Created {len(products)} products")
        return products

    def export_reviews_collection(self, df: pd.DataFrame, products: List[Dict]) -> List[Dict]:
        """Convert reviews DataFrame to MongoDB collection"""
        logger.info("\n💬 Creating Reviews collection...")

        if df.empty:
            logger.warning("⚠️  No reviews to export")
            return []

        # Create product lookup (use first N products)
        available_product_ids = [p['_id'] for p in products[:min(len(products), len(df))]]

        reviews = []

        for idx, row in df.iterrows():
            # Assign to a product (round-robin or random)
            product_id = available_product_ids[idx % len(available_product_ids)] if available_product_ids else None

            review = {
                '_id': self.generate_object_id('review', idx),
                'product_id': product_id,
                'text': self.clean_for_json(row.get('text', '')),
                'rating': self.clean_for_json(row.get('rating', 5)),
                'author': self.clean_for_json(row.get('author', 'Anonymous')),
                'language': self.clean_for_json(row.get('language', 'fr')),
                'is_fake': bool(self.clean_for_json(row.get('is_fake', False))),
                'fake_probability': self.clean_for_json(row.get('fake_probability', 0.0)),
                'xgboost_score': self.clean_for_json(row.get('xgboost_score', 0.0)),
                'bert_score': self.clean_for_json(row.get('bert_score', 0.0)),
                'ensemble_score': self.clean_for_json(row.get('ensemble_score', 0.0)),
                'confidence': self.clean_for_json(row.get('confidence', 'low')),
                'sentiment': self.clean_for_json(row.get('sentiment', 'neutral')),
                'predicted_at': self.clean_for_json(row.get('predicted_at', datetime.now().isoformat())),
                'created_at': datetime.now().isoformat()
            }

            reviews.append(review)

            if (idx + 1) % 500 == 0:
                logger.info(f"   Processed {idx + 1}/{len(df)} reviews")

        logger.info(f"✅ Created {len(reviews)} reviews")
        return reviews

    def export_price_history_collection(self, products: List[Dict]) -> List[Dict]:
        """Generate price history for products"""
        logger.info("\n📈 Creating PriceHistory collection...")

        price_history = []
        history_id = 0

        # Generate sample history for first 50 products
        for product in products[:min(50, len(products))]:
            base_price = product.get('price', 100)

            # Generate 5 historical prices
            for days_ago in [30, 21, 14, 7, 0]:
                # Simulate price variation
                variation = np.random.uniform(-0.05, 0.05)
                historical_price = base_price * (1 + variation)

                history_entry = {
                    '_id': self.generate_object_id('history', history_id),
                    'product_id': product['_id'],
                    'price': round(historical_price, 2),
                    'currency': product.get('currency', 'TND'),
                    'source': product.get('source', ''),
                    'date': (datetime.now() - pd.Timedelta(days=days_ago)).isoformat(),
                    'created_at': datetime.now().isoformat()
                }

                price_history.append(history_entry)
                history_id += 1

        logger.info(f"✅ Created {len(price_history)} price history entries")
        return price_history

    def export_search_history_collection(self) -> List[Dict]:
        """Create sample search history"""
        logger.info("\n🔍 Creating SearchHistory collection...")

        search_history = [
            {
                '_id': self.generate_object_id('search', 1),
                'user_id': self.generate_object_id('user', 2),
                'query': 'laptop HP',
                'results_count': 45,
                'timestamp': datetime.now().isoformat()
            },
            {
                '_id': self.generate_object_id('search', 2),
                'user_id': self.generate_object_id('user', 2),
                'query': 'gaming monitor',
                'results_count': 23,
                'timestamp': datetime.now().isoformat()
            }
        ]

        logger.info(f"✅ Created {len(search_history)} search history entries")
        return search_history

    def save_collection(self, collection_name: str, documents: List[Dict]):
        """Save collection to JSON file"""
        if not documents:
            logger.warning(f"⚠️  No documents to save for {collection_name}")
            return

        filepath = self.mongodb_dir / f"{collection_name}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        self.stats['collections_created'] += 1
        self.stats['by_collection'][collection_name] = len(documents)
        self.stats['total_documents'] += len(documents)

        logger.info(f"✅ Saved {collection_name}.json ({len(documents)} documents)")

    def generate_import_script(self):
        """Generate MongoDB import script"""
        logger.info("\n📝 Generating import script...")

        script_lines = []
        script_lines.append("#!/bin/bash")
        script_lines.append("# MongoDB Import Script")
        script_lines.append("# Run this script to import all collections into MongoDB")
        script_lines.append("")
        script_lines.append("DB_NAME='pricecheck_tn'")
        script_lines.append("")
        script_lines.append("echo '🚀 Starting MongoDB import...'")
        script_lines.append("echo ''")
        script_lines.append("")

        collections = [
            'users', 'user_profiles', 'categories', 'stores',
            'products', 'reviews', 'price_history', 'search_history'
        ]

        for collection in collections:
            script_lines.append(f"echo '📦 Importing {collection}...'")
            script_lines.append(
                f"mongoimport --db $DB_NAME --collection {collection} "
                f"--file data/final/mongodb/{collection}.json --jsonArray"
            )
            script_lines.append("echo ''")
            script_lines.append("")

        script_lines.append("echo '✅ Import complete!'")
        script_lines.append("echo ''")
        script_lines.append("echo 'To verify:'")
        script_lines.append("echo '  mongo pricecheck_tn --eval \"db.stats()\"'")

        script_path = self.mongodb_dir / "import.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_lines))

        # Windows batch script
        bat_lines = []
        bat_lines.append("@echo off")
        bat_lines.append("REM MongoDB Import Script for Windows")
        bat_lines.append("SET DB_NAME=pricecheck_tn")
        bat_lines.append("echo Starting MongoDB import...")
        bat_lines.append("")

        for collection in collections:
            bat_lines.append(f"echo Importing {collection}...")
            bat_lines.append(
                f"mongoimport --db %DB_NAME% --collection {collection} "
                f"--file data\\final\\mongodb\\{collection}.json --jsonArray"
            )
            bat_lines.append("")

        bat_lines.append("echo Import complete!")
        bat_lines.append("pause")

        bat_path = self.mongodb_dir / "import.bat"
        with open(bat_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(bat_lines))

        logger.info(f"✅ Import scripts created")
        logger.info(f"   Linux/Mac: {script_path}")
        logger.info(f"   Windows: {bat_path}")

    def generate_report(self):
        """Generate export report"""
        logger.info("\n📊 Generating export report...")

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("MONGODB EXPORT REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")

        report_lines.append("📊 SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Collections created:       {self.stats['collections_created']}")
        report_lines.append(f"Total documents:           {self.stats['total_documents']:,}")
        report_lines.append("")

        report_lines.append("📦 BY COLLECTION")
        report_lines.append("-" * 40)
        for collection, count in self.stats['by_collection'].items():
            report_lines.append(f"{collection:20} {count:8,} documents")
        report_lines.append("")

        report_lines.append("📁 FILES CREATED")
        report_lines.append("-" * 40)
        report_lines.append(f"Location: data/final/mongodb/")
        for collection in self.stats['by_collection'].keys():
            report_lines.append(f"  ✅ {collection}.json")
        report_lines.append(f"  ✅ import.sh")
        report_lines.append(f"  ✅ import.bat")
        report_lines.append("")

        report_lines.append("🚀 NEXT STEPS")
        report_lines.append("-" * 40)
        report_lines.append("1. Install MongoDB locally or use MongoDB Atlas")
        report_lines.append("2. Run import script:")
        report_lines.append("   Linux/Mac: bash data/final/mongodb/import.sh")
        report_lines.append("   Windows:   data\\final\\mongodb\\import.bat")
        report_lines.append("3. Verify import:")
        report_lines.append("   mongo pricecheck_tn --eval 'db.products.count()'")
        report_lines.append("")

        report_lines.append("=" * 60)
        report_lines.append("✅ EXPORT COMPLETE")
        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        report_path = self.mongodb_dir / "export_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"✅ Report saved: {report_path}")

    def run(self):
        """Main execution"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING MONGODB EXPORT")
        logger.info("=" * 60)

        # Load datasets
        datasets = self.load_datasets()

        # Create collections
        logger.info("\n📦 Creating MongoDB collections...")

        users = self.export_users_collection()
        self.save_collection('users', users)

        profiles = self.export_user_profiles_collection()
        self.save_collection('user_profiles', profiles)

        categories = self.export_categories_collection()
        self.save_collection('categories', categories)

        stores = self.export_stores_collection()
        self.save_collection('stores', stores)

        products = self.export_products_collection(datasets['products'], categories)
        self.save_collection('products', products)

        reviews = self.export_reviews_collection(datasets['reviews'], products)
        self.save_collection('reviews', reviews)

        price_history = self.export_price_history_collection(products)
        self.save_collection('price_history', price_history)

        search_history = self.export_search_history_collection()
        self.save_collection('search_history', search_history)

        # Generate import script
        self.generate_import_script()

        # Generate report
        self.generate_report()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 SUCCESS! MongoDB export complete")
        logger.info("=" * 60)


def main():
    """Main execution"""
    try:
        exporter = MongoDBExporter()
        exporter.run()

    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()