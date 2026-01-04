"""
Data loader - Load all scraped data
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger


class DataLoader:
    """Load scraped data from multiple sources"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load_json(self, filename: str) -> List[Dict]:
        """Load a single JSON file"""
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"✓ Loaded {len(data)} items from {filename}")
            return data

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

    def load_all_sources(self) -> pd.DataFrame:
        """Load data from all sources and combine"""

        logger.info("📦 Loading data from all sources...")

        all_data = []

        # Load Tunisianet
        tunisianet_data = self.load_json('tunisianet_latest.json')
        all_data.extend(tunisianet_data)

        # Load MyTek
        mytek_data = self.load_json('mytek_latest.json')
        all_data.extend(mytek_data)

        # Load LDLC
        ldlc_data = self.load_json('ldlc_latest.json')
        all_data.extend(ldlc_data)

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        logger.success(f"✅ Loaded {len(df)} total products from {len(all_data)} sources")

        return df

    def load_by_source(self) -> Dict[str, pd.DataFrame]:
        """Load data separated by source"""

        sources = {}

        # Tunisianet
        tunisianet_data = self.load_json('tunisianet_latest.json')
        if tunisianet_data:
            sources['tunisianet'] = pd.DataFrame(tunisianet_data)

        # MyTek
        mytek_data = self.load_json('mytek_latest.json')
        if mytek_data:
            sources['mytek'] = pd.DataFrame(mytek_data)

        # LDLC
        ldlc_data = self.load_json('ldlc_latest.json')
        if ldlc_data:
            sources['ldlc'] = pd.DataFrame(ldlc_data)

        return sources

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get data summary statistics"""

        summary = {
            'total_products': len(df),
            'sources': df['source'].value_counts().to_dict(),
            'countries': df['country'].value_counts().to_dict(),
            'currencies': df['currency'].value_counts().to_dict(),
            'missing_prices': df['price'].isna().sum(),
            'missing_titles': df['title'].isna().sum(),
            'date_range': {
                'min': df['scraped_at'].min() if 'scraped_at' in df.columns else None,
                'max': df['scraped_at'].max() if 'scraped_at' in df.columns else None,
            }
        }

        return summary