"""
LDLC Hybrid Scraper - Playwright + BeautifulSoup
Necessary due to anti-bot protection
"""

from playwright.sync_api import sync_playwright, Page
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
import re
import json
import time
import random
from pathlib import Path
from utils.robots_checker import robots_checker


class LDLCHybridScraper:
    """
    LDLC scraper using Playwright (for anti-bot) + BeautifulSoup (for parsing)
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['base_url']

        # Rate limiting
        self.min_delay = config['rate_limit']['min_delay']
        self.max_delay = config['rate_limit']['max_delay']
        self.page_delay = config['rate_limit']['page_delay']
        self.last_request_time = 0

        # Pagination
        self.max_pages = config['pagination']['max_pages']

        # Robots checker
        self.robots_checker = robots_checker

    def random_delay(self, min_sec: float = None, max_sec: float = None):
        """Random delay"""
        min_sec = min_sec or self.min_delay
        max_sec = max_sec or self.max_delay
        delay = random.uniform(min_sec, max_sec)
        logger.debug(f"💤 Sleeping {delay:.1f}s...")
        time.sleep(delay)

    def respect_rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            wait_time = self.min_delay - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _extract_specs(self, title: str) -> Dict[str, Optional[str]]:
        """Extract specs from title"""
        specs = {
            'processor': None,
            'gpu': None,
            'ram_gb': None,
            'storage': None,
            'screen_size': None
        }

        # Processor
        processor_patterns = [
            r'(Intel Core i[3579]-\d+\w*)',
            r'(AMD Ryzen [579] \d+\w*)',
        ]
        for pattern in processor_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                specs['processor'] = match.group(1)
                break

        # GPU
        gpu_patterns = [
            r'(RTX \d{4}\s*\w*)',
            r'(GTX \d{4}\s*\w*)',
        ]
        for pattern in gpu_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                specs['gpu'] = match.group(1)
                break

        # RAM
        ram_match = re.search(r'(\d+)\s*(?:Go|GB)', title, re.IGNORECASE)
        if ram_match:
            specs['ram_gb'] = int(ram_match.group(1))

        # Storage
        storage_match = re.search(r'(\d+)\s*(?:Go|To|GB|TB)', title, re.IGNORECASE)
        if storage_match:
            specs['storage'] = storage_match.group(0)

        # Screen
        screen_match = re.search(r'(\d{2}(?:\.\d)?)["\']', title)
        if screen_match:
            specs['screen_size'] = f"{screen_match.group(1)}\""

        return specs

    def _parse_product(self, item) -> Optional[Dict]:
        """Parse product from BeautifulSoup element"""
        try:
            # Product ID
            product_id = item.get('data-product-id')

            # Title
            title_elem = (
                    item.select_one('.title-3') or
                    item.select_one('h3') or
                    item.select_one('a[title]')
            )

            if not title_elem:
                return None

            title = title_elem.get('title', '') or title_elem.text.strip()

            # URL
            link_elem = item.select_one('a[href*="/fiche/"]')
            url = ''
            if link_elem:
                url = link_elem.get('href', '')
            elif title_elem.name == 'a':
                url = title_elem.get('href', '')

            # Price
            price_elem = (
                    item.select_one('.price') or
                    item.select_one('[data-price]')
            )

            price = None
            if price_elem:
                price_data = price_elem.get('data-price')
                if price_data:
                    try:
                        price = float(price_data)
                    except:
                        pass

                if not price:
                    price_text = price_elem.text.strip()
                    # LDLC format: "699,95 €"
                    cleaned = re.sub(r'[^\d,.]', '', price_text)
                    cleaned = cleaned.replace(',', '.')
                    try:
                        price = float(cleaned)
                    except:
                        pass

            # Brand
            brand = None
            brand_patterns = [
                r'^(HP|ASUS|Lenovo|Dell|Acer|MSI|Apple|Samsung|Razer)\b',
            ]
            for pattern in brand_patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    brand = match.group(1).upper()
                    break

            # Stock
            stock_elem = item.select_one('.availability') or item.select_one('.stock')
            in_stock = True
            if stock_elem:
                stock_text = stock_elem.text.strip().lower()
                in_stock = 'disponible' in stock_text or 'en stock' in stock_text

            # Image
            img_elem = item.select_one('img')
            image = None
            if img_elem:
                image = img_elem.get('data-src') or img_elem.get('src')

            # Specs
            specs = self._extract_specs(title)

            product = {
                'id': product_id or str(hash(url))[-8:],
                'title': title,
                'price': price,
                'currency': 'EUR',
                'brand': brand,
                'url': url if url.startswith('http') else self.base_url + url,
                'image': image,
                'in_stock': in_stock,
                'source': 'ldlc',
                'country': 'France',
                'scraped_at': datetime.now().isoformat(),
                **specs
            }

            return product

        except Exception as e:
            logger.debug(f"Error parsing product: {e}")
            return None

    def scrape_page_with_playwright(self, url: str) -> List[Dict]:
        """
        Scrape page using Playwright
        """
        # Check robots.txt
        if not self.robots_checker.can_fetch(url):
            logger.error(f"🚫 Blocked by robots.txt: {url}")
            return []

        # Rate limiting
        self.respect_rate_limit()

        products = []

        try:
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(headless=True)

                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080},
                    locale='fr-FR'
                )

                page = context.new_page()

                logger.info(f"🌐 Loading: {url}")

                # Navigate
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                except Exception as e:
                    logger.error(f"Error loading page: {e}")
                    browser.close()
                    return []

                # Wait for products
                logger.info("⏳ Waiting for products...")

                try:
                    page.wait_for_selector('li.pdt-item', timeout=15000)
                    logger.info("✓ Products loaded")
                except:
                    logger.warning("Timeout waiting for products")

                # Wait a bit more
                time.sleep(3)

                # Scroll
                for i in range(3):
                    page.evaluate(f"window.scrollBy(0, {500 * (i + 1)})")
                    time.sleep(0.5)

                # Get HTML
                html = page.content()

                browser.close()

                logger.success(f"✅ Got HTML ({len(html)} bytes)")

                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')

                # Find products
                items = soup.select('li.pdt-item')

                logger.info(f"Found {len(items)} products")

                # Parse each
                for idx, item in enumerate(items, 1):
                    product = self._parse_product(item)
                    if product:
                        products.append(product)
                        logger.debug(f"✓ [{idx:02d}] {product['title'][:40]} - {product['price']} EUR")

                logger.success(f"✅ Parsed {len(products)}/{len(items)} products")

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return products

    def scrape_with_pagination(self, category_url: str, max_pages: int = None) -> List[Dict]:
        """Scrape multiple pages"""
        max_pages = max_pages or self.max_pages
        all_products = []

        logger.info(f"🚀 Starting LDLC scrape: {category_url}")
        logger.info(f"📄 Max pages: {max_pages}")
        logger.warning("⚠️  Using Playwright (anti-bot protection)")

        for page_num in range(1, max_pages + 1):
            # Build URL
            if page_num == 1:
                page_url = f"{self.base_url}{category_url}"
            else:
                base = category_url.rstrip('/')
                page_url = f"{self.base_url}{base}/page{page_num}/"

            logger.info(f"\n{'=' * 60}")
            logger.info(f"📄 PAGE {page_num}/{max_pages}")
            logger.info(f"{'=' * 60}")

            # Wait between pages
            if page_num > 1:
                logger.info(f"⏸️  Waiting {self.page_delay}s...")
                self.random_delay(self.page_delay, self.page_delay + 1)

            products = self.scrape_page_with_playwright(page_url)

            if not products:
                logger.warning(f"No products on page {page_num}")
                if page_num == 1:
                    logger.error("❌ No products on first page!")
                break

            all_products.extend(products)
            logger.info(f"📊 Page {page_num}: {len(products)} products | Total: {len(all_products)}")

        logger.success(f"\n🎉 COMPLETE! Total: {len(all_products)} products")
        return all_products

    def scrape_category(self, category_key: str, max_pages: int = None) -> List[Dict]:
        """Scrape category"""
        if category_key not in self.config['categories']:
            logger.error(f"Unknown category: {category_key}")
            return []

        category_url = self.config['categories'][category_key]
        return self.scrape_with_pagination(category_url, max_pages)

    def save_to_json(self, products: List[Dict], filename: str) -> Path:
        """Save to JSON"""
        from config import RAW_DATA_DIR
        filepath = RAW_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        logger.success(f"💾 Saved to: {filepath}")
        return filepath

    def save_to_csv(self, products: List[Dict], filename: str) -> Path:
        """Save to CSV"""
        import pandas as pd
        from config import RAW_DATA_DIR
        filepath = RAW_DATA_DIR / filename
        df = pd.DataFrame(products)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.success(f"💾 Saved to: {filepath}")
        return filepath