"""
Complete Tunisianet scraper with pagination support
"""
from bs4 import BeautifulSoup

from scrapers.simple_scraper import SimpleBaseScraper
from utils.robots_checker import robots_checker
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
import re
import json
from pathlib import Path


class TunisianetScraper(SimpleBaseScraper):
    """
    Scraper for Tunisianet.com.tn with full pagination support
    """

    def __init__(self, config):
        # Store original config for file paths
        self.original_config = config

        # Create a dict from the config object for compatibility
        config_dict = {
            'base_url': 'https://www.tunisianet.com.tn',
            'headers': {
                'User-Agent': config.SCRAPING_USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            },
            'rate_limit': {
                'min_delay': config.SCRAPING_RATE_LIMIT,
                'max_delay': config.SCRAPING_RATE_LIMIT * 1.5,
            },
            'categories': {
                'laptops': '/702-ordinateur-portable',
                'gaming_laptops': '/681-pc-portable-gamer',
                'graphics_cards': '/410-carte-graphique-tunisie',
                'processors': '/421-processeur',
                'monitors': '/667-ecran-pc-tunisie'
            }
        }

        super().__init__(config_dict)
        self.robots_checker = robots_checker

        # Pagination config
        self.max_pages = config.SCRAPING_MAX_PAGES
        self.page_delay = config.SCRAPING_RATE_LIMIT

    def _extract_specs(self, title: str) -> Dict[str, Optional[str]]:
        """Extract technical specifications from product title"""
        specs = {
            'processor': None,
            'gpu': None,
            'ram_gb': None,
            'storage': None,
            'screen_size': None
        }

        # Processor patterns
        processor_patterns = [
            r'(Intel Core i[3579]-\d+\w*)',
            r'(AMD Ryzen [579] \d+\w*)',
            r'(Core i[3579] \d+\w*)',
            r'(i[3579]-\d+\w*)'
        ]
        for pattern in processor_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                specs['processor'] = match.group(1)
                break

        # GPU patterns
        gpu_patterns = [
            r'(RTX \d{4}\s*\w*)',
            r'(GTX \d{4}\s*\w*)',
            r'(Radeon \w+)',
            r'(GeForce \w+)',
            r'(NVIDIA \w+)'
        ]
        for pattern in gpu_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                specs['gpu'] = match.group(1)
                break

        # RAM (Go = GB in French)
        ram_match = re.search(r'(\d+)\s*Go(?:\s+RAM)?', title, re.IGNORECASE)
        if ram_match:
            specs['ram_gb'] = int(ram_match.group(1))

        # Storage
        storage_match = re.search(r'(\d+)\s*(Go|To)(?:\s+SSD)?', title, re.IGNORECASE)
        if storage_match:
            amount = storage_match.group(1)
            unit = storage_match.group(2)
            specs['storage'] = f"{amount} {unit} SSD"

        # Screen size
        screen_match = re.search(r'(\d{2}(?:\.\d)?)["\']', title)
        if screen_match:
            specs['screen_size'] = f"{screen_match.group(1)}\""

        return specs

    def _parse_product(self, item: BeautifulSoup) -> Optional[Dict]:
        """Parse a single product element"""
        try:
            # Title and URL
            title_elem = item.find('h2', class_='h3 product-title')
            if not title_elem or not title_elem.find('a'):
                return None

            title_link = title_elem.find('a')
            title = title_link.text.strip()
            url = title_link.get('href', '')

            # Price (format: "2,025.000 DT")
            price_elem = item.find('span', class_='price')
            price = None
            if price_elem:
                price_text = price_elem.text.strip()
                # Tunisianet format: "2,025.000 DT" = 2025 TND
                price_clean = re.sub(r'[^\d.]', '', price_text)
                try:
                    # Divide by 1000 to convert milimes to dinars
                    price = float(price_clean) / 1000
                except:
                    logger.warning(f"Could not parse price: {price_text}")

            # Regular price (if on sale)
            regular_price_elem = item.find('span', class_='regular-price')
            regular_price = None
            if regular_price_elem:
                reg_text = regular_price_elem.text.strip()
                reg_clean = re.sub(r'[^\d.]', '', reg_text)
                try:
                    regular_price = float(reg_clean) / 1000
                except:
                    pass

            # Calculate discount
            discount_percent = 0.0
            if regular_price and price and regular_price > price:
                discount_percent = round(
                    (regular_price - price) / regular_price * 100, 2
                )

            # Brand
            brand_elem = item.find('div', class_='brand')
            brand = None
            if brand_elem and brand_elem.find('img'):
                brand = brand_elem.find('img').get('alt', '').strip()

            # Stock status
            stock_elem = item.find('div', class_='product-availability')
            in_stock = False
            if stock_elem:
                stock_text = stock_elem.text.strip()
                in_stock = 'En stock' in stock_text or 'Disponible' in stock_text

            # Image
            img_elem = item.find('img', class_='img-fluid')
            image = None
            if img_elem:
                image = img_elem.get('data-src') or img_elem.get('src')

            # Extract specs from title
            specs = self._extract_specs(title)

            # Build product dict
            product = {
                'id': self.extract_product_id(url),
                'title': title,
                'price': price,
                'currency': 'TND',
                'regular_price': regular_price,
                'discount_percent': discount_percent,
                'brand': brand,
                'url': url,
                'image': image,
                'in_stock': in_stock,
                'source': 'tunisianet',
                'country': 'Tunisia',
                'scraped_at': datetime.now().isoformat(),
                **specs  # Add extracted specifications
            }

            return product

        except Exception as e:
            logger.error(f"Error parsing product: {e}")
            return None

    def scrape_page(self, url: str) -> List[Dict]:
        """
        Scrape a single page

        Args:
            url: Page URL to scrape

        Returns:
            List of product dictionaries
        """
        products = []

        # Check robots.txt
        if not self.robots_checker.can_fetch(url):
            logger.error(f"🚫 Blocked by robots.txt: {url}")
            return []

        # Fetch HTML
        html = self.fetch_page(url)
        if not html:
            return []

        # Parse with BeautifulSoup
        soup = self.parse_html(html)

        # Find all product items
        product_items = soup.find_all('article', class_='product-miniature')
        logger.info(f"Found {len(product_items)} product items on page")

        # Parse each product
        for idx, item in enumerate(product_items, 1):
            product = self._parse_product(item)
            if product:
                products.append(product)
                logger.debug(
                    f"✓ [{idx:02d}] {product['title'][:40]:40s} - "
                    f"{product['price']:7.2f} TND"
                )

        logger.success(f"✅ Parsed {len(products)}/{len(product_items)} products")
        return products

    def scrape_with_pagination(
            self,
            category_url: str,
            max_pages: int = None
    ) -> List[Dict]:
        """
        Scrape multiple pages with pagination

        Args:
            category_url: Category URL (e.g., /108-pc-portable)
            max_pages: Maximum pages to scrape (default from config)

        Returns:
            List of all products from all pages
        """
        max_pages = max_pages or self.max_pages
        all_products = []

        logger.info(f"🚀 Starting pagination scrape: {category_url}")
        logger.info(f"📄 Will scrape up to {max_pages} pages")

        for page_num in range(1, max_pages + 1):
            # Build page URL
            if page_num == 1:
                page_url = f"{self.base_url}{category_url}"
            else:
                # Tunisianet pagination: ?page=2
                separator = '&' if '?' in category_url else '?'
                page_url = f"{self.base_url}{category_url}{separator}page={page_num}"

            logger.info(f"\n{'=' * 60}")
            logger.info(f"📄 PAGE {page_num}/{max_pages}")
            logger.info(f"🔗 {page_url}")
            logger.info(f"{'=' * 60}")

            # Scrape this page
            products = self.scrape_page(page_url)

            if not products:
                logger.warning(f"No products found on page {page_num}, stopping pagination")
                break

            all_products.extend(products)

            logger.info(
                f"📊 Page {page_num} complete: {len(products)} products | "
                f"Total so far: {len(all_products)}"
            )

            # Wait before next page (except last)
            if page_num < max_pages and products:
                logger.info(f"⏸️  Waiting {self.page_delay}s before next page...")
                self.random_delay(self.page_delay, self.page_delay + 2)

        logger.success(
            f"\n🎉 PAGINATION COMPLETE! "
            f"Total products: {len(all_products)} from {page_num} pages"
        )

        return all_products

    def scrape_category(
            self,
            category_key: str,
            max_pages: int = None
    ) -> List[Dict]:
        """
        Scrape a category by key

        Args:
            category_key: Category key from config (e.g., 'laptops', 'gaming_laptops')
            max_pages: Override max pages from config

        Returns:
            List of products
        """
        if category_key not in self.config['categories']:
            logger.error(f"Unknown category: {category_key}")
            logger.info(f"Available: {list(self.config['categories'].keys())}")
            return []

        category_url = self.config['categories'][category_key]
        logger.info(f"🏷️  Scraping category: {category_key}")

        return self.scrape_with_pagination(category_url, max_pages)

    def save_to_json(self, products: List[Dict], filename: str) -> Path:
        """Save products to JSON file"""
        filepath = self.original_config.RAW_DATA_DIR / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        logger.success(f"💾 Saved {len(products)} products to: {filepath}")
        return filepath

    def save_to_csv(self, products: List[Dict], filename: str) -> Path:
        """Save products to CSV file"""
        import pandas as pd
        filepath = self.original_config.RAW_DATA_DIR / filename

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(products)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.success(f"💾 Saved {len(products)} products to: {filepath}")
        return filepath

    def print_summary(self, products: List[Dict]):
        """Print summary statistics"""
        if not products:
            logger.warning("No products to summarize")
            return

        import pandas as pd
        df = pd.DataFrame(products)

        print("\n" + "=" * 70)
        print(f"📊 SCRAPING SUMMARY - {len(products)} products")
        print("=" * 70)

        # Price stats
        if 'price' in df.columns and df['price'].notna().any():
            print("\n💰 Price Statistics (TND):")
            print(df['price'].describe())

        # Brand distribution
        if 'brand' in df.columns:
            print("\n🏢 Brand Distribution:")
            print(df['brand'].value_counts().head(10))

        # Stock status
        if 'in_stock' in df.columns:
            print("\n📦 Stock Status:")
            in_stock_count = df['in_stock'].sum()
            print(f"  In Stock: {in_stock_count}")
            print(f"  Out of Stock: {len(df) - in_stock_count}")

        # Discounts
        if 'discount_percent' in df.columns:
            discounted = df[df['discount_percent'] > 0]
            print(f"\n🏷️  Products on Sale: {len(discounted)}")
            if len(discounted) > 0:
                print(f"  Average discount: {discounted['discount_percent'].mean():.1f}%")

        # Sample products
        print("\n📦 Sample Products:")
        sample_cols = ['title', 'price', 'brand', 'in_stock']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(5).to_string(index=False))

        print("\n" + "=" * 70)
