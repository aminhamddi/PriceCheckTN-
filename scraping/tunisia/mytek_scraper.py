"""
MyTek Hybrid Scraper - Playwright + BeautifulSoup
Necessary because products load via JavaScript
"""

from playwright.sync_api import sync_playwright
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


class MyTekHybridScraper:
    """
    MyTek scraper using Playwright (for JS) + BeautifulSoup (for parsing)
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_url = "https://www.mytek.tn/"  # Hardcoded for now

        # Rate limiting
        self.min_delay = config.SCRAPING_RATE_LIMIT
        self.max_delay = config.SCRAPING_RATE_LIMIT
        self.page_delay = config.SCRAPING_RATE_LIMIT
        self.last_request_time = 0

        # Pagination
        self.max_pages = config.SCRAPING_MAX_PAGES
        self.param_name = "p"

        # Robots checker
        self.robots_checker = robots_checker

    def random_delay(self, min_sec: float = None, max_sec: float = None):
        """Random delay"""
        min_sec = min_sec or self.min_delay
        max_sec = max_sec or self.max_delay
        delay = random.uniform(min_sec, max_sec)
        logger.debug(f" Sleeping {delay:.1f}s...")
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

    def _parse_mytek_price(self, price_text: str) -> Optional[float]:
        """Parse MyTek price: "429,000 DT" -> 429.0"""
        try:
            cleaned = price_text.replace('DT', '').strip()
            cleaned = cleaned.replace(',', '')
            price = float(cleaned) / 1000
            return price
        except:
            return None

    def _parse_product(self, item) -> Optional[Dict]:
        """Parse product from BeautifulSoup element"""
        try:
            # Product ID
            product_id = item.get('data-product-id')

            # Title
            title_elem = item.select_one('h1.product-item-name a')
            if not title_elem:
                return None

            title = title_elem.text.strip()
            url = title_elem.get('href', '')

            # SKU
            sku_elem = item.select_one('.sku')
            sku = sku_elem.text.strip() if sku_elem else None

            # Price
            price_box = item.select_one('.price-box')
            price = None
            regular_price = None

            if price_box:
                final_price_elem = price_box.select_one('.final-price')
                if final_price_elem:
                    price = self._parse_mytek_price(final_price_elem.text.strip())

                original_price_elem = price_box.select_one('.original-price')
                if original_price_elem:
                    regular_price = self._parse_mytek_price(original_price_elem.text.strip())

            # Discount
            discount_percent = 0.0
            if regular_price and price and regular_price > price:
                discount_percent = round((regular_price - price) / regular_price * 100, 2)

            # Brand
            brand_elem = item.select_one('.brand img')
            brand = brand_elem.get('alt', '').strip() if brand_elem else None

            # Stock
            stock_elem = item.select_one('.stock')
            in_stock = True
            if stock_elem:
                stock_text = stock_elem.text.strip().lower()
                in_stock = 'épuisé' not in stock_text

            # Image
            img_elem = item.select_one('.product-item-photo img')
            image = img_elem.get('src') if img_elem else None

            # Description
            desc_elem = item.select_one('.search-short-description')
            description = desc_elem.text.strip() if desc_elem else None

            # Specs
            specs = self._extract_specs(title)

            product = {
                'id': product_id or str(hash(url))[-8:],
                'title': title,
                'price': price,
                'currency': 'TND',
                'regular_price': regular_price,
                'discount_percent': discount_percent,
                'brand': brand,
                'sku': sku,
                'url': url if url.startswith('http') else self.base_url + url,
                'image': image,
                'description': description,
                'in_stock': in_stock,
                'source': 'mytek',
                'country': 'Tunisia',
                'scraped_at': datetime.now().isoformat(),
                **specs
            }

            return product

        except Exception as e:
            logger.debug(f"Error parsing product: {e}")
            return None

    def scrape_page_with_playwright(self, url: str) -> List[Dict]:
        """
        Scrape page using Playwright (handles JavaScript)
        Then parse with BeautifulSoup
        """
        # Check robots.txt
        if not self.robots_checker.can_fetch(url):
            logger.error(f" Blocked by robots.txt: {url}")
            return []

        # Rate limiting
        self.respect_rate_limit()

        products = []

        try:
            with sync_playwright() as p:
                # Launch browser (headless)
                browser = p.chromium.launch(headless=True)

                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    viewport={'width': 1920, 'height': 1080}
                )

                page = context.new_page()

                logger.info(f" Loading: {url}")

                # Navigate to page
                page.goto(url, wait_until='domcontentloaded', timeout=30000)

                # Wait for products to load (replace skeletons)
                logger.info(" Waiting for products to load...")

                try:
                    # Wait for actual products (not skeletons)
                    page.wait_for_selector('.product-container[data-product-id]', timeout=15000)
                    logger.info("✓ Products loaded")
                except:
                    logger.warning("Timeout waiting for products")

                # Additional wait for lazy loading
                time.sleep(3)

                # Scroll to load all images
                for i in range(3):
                    page.evaluate(f"window.scrollBy(0, {500 * (i+1)})")
                    time.sleep(0.5)

                # Get HTML after JavaScript execution
                html = page.content()

                browser.close()

                logger.success(f" Got HTML ({len(html)} bytes)")

                # Check if this is a 404 page
                if "404 Not Found" in html or "Whoops, our bad" in html:
                    logger.warning(" Got 404 error page - URL is invalid")
                    return []

                # Debug: Save HTML to check structure
                debug_html_path = Path("debug_mytek.html")
                with open(debug_html_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f" Saved debug HTML to {debug_html_path}")

                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')

                # Try different selectors
                # Original selector
                items_data_product = soup.find_all('div', class_='product-container', attrs={'data-product-id': True})
                logger.info(f"Found {len(items_data_product)} products with data-product-id")

                # Alternative selectors
                items_product_item = soup.find_all('div', class_='product-item')
                logger.info(f"Found {len(items_product_item)} products with product-item class")

                items_product_name = soup.find_all('h1', class_='product-item-name')
                logger.info(f"Found {len(items_product_name)} products with product-item-name")

                # Use the one with most results
                if len(items_data_product) > 0:
                    items = items_data_product
                    logger.info("Using data-product-id selector")
                elif len(items_product_item) > 0:
                    items = items_product_item
                    logger.info("Using product-item selector")
                else:
                    items = items_product_name
                    logger.info("Using product-item-name selector")

                # Parse each product
                for idx, item in enumerate(items, 1):
                    product = self._parse_product(item)
                    if product:
                        products.append(product)
                        logger.debug(f" [{idx:02d}] {product['title'][:40]} - {product['price']} TND")

                logger.success(f" Parsed {len(products)}/{len(items)} products")

        except Exception as e:
            logger.error(f" Error: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return products

    def scrape_with_pagination(self, category_url: str, max_pages: int = None) -> List[Dict]:
        """Scrape multiple pages"""
        max_pages = max_pages or self.max_pages
        all_products = []

        logger.info(f" Starting MyTek scrape: {category_url}")
        logger.info(f" Max pages: {max_pages}")
        logger.warning("  Using Playwright (slow but necessary for JS)")

        # Scrape first page
        first_url = f"{self.base_url}{category_url}"

        logger.info(f"\n{'='*60}")
        logger.info(f" PAGE 1/{max_pages}")
        logger.info(f"{'='*60}")

        products = self.scrape_page_with_playwright(first_url)

        if not products:
            logger.error(" No products on first page!")
            return []

        all_products.extend(products)
        logger.info(f" Page 1: {len(products)} products | Total: {len(all_products)}")

        # Extract pagination info
        pagination_id = self._extract_pagination_id(first_url)

        # Scrape remaining pages
        for page_num in range(2, max_pages + 1):
            # Build URL
            if pagination_id:
                page_url = f"{self.base_url}{category_url}?id={pagination_id}&{self.param_name}={page_num}"
            else:
                separator = '&' if '?' in category_url else '?'
                page_url = f"{self.base_url}{category_url}{separator}{self.param_name}={page_num}"

            logger.info(f"\n{'='*60}")
            logger.info(f" PAGE {page_num}/{max_pages}")
            logger.info(f"{'='*60}")

            # Wait between pages
            logger.info(f"  Waiting {self.page_delay}s...")
            self.random_delay(self.page_delay, self.page_delay + 2)

            products = self.scrape_page_with_playwright(page_url)

            if not products:
                logger.warning(f"No products on page {page_num}")
                break

            all_products.extend(products)
            logger.info(f" Page {page_num}: {len(products)} products | Total: {len(all_products)}")

        logger.success(f"\n COMPLETE! Total: {len(all_products)} products")
        return all_products

    def _extract_pagination_id(self, url: str) -> Optional[str]:
        """Extract pagination ID from first page"""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until='domcontentloaded')
                page.wait_for_selector('.pagination', timeout=5000)
                html = page.content()
                browser.close()

            soup = BeautifulSoup(html, 'html.parser')
            page_2_link = soup.select_one('.pagination a[href*="p=2"]')

            if page_2_link:
                href = page_2_link.get('href', '')
                id_match = re.search(r'[?&]id=(\d+)', href)
                if id_match:
                    return id_match.group(1)
        except:
            pass

        return None

    def scrape_category(self, category_key: str, max_pages: int = None) -> List[Dict]:
        """Scrape category"""
        # Updated category mapping based on current MyTek website structure
        categories = {
            'laptops': '/informatique/ordinateurs-portables.html',
            'gaming_laptops': '/informatique/ordinateurs-portables/pc-gamer.html',
            'desktops': '/informatique/ordinateur-de-bureau.html',
            'gaming_desktops': '/informatique/ordinateur-de-bureau/ordinateur-gamer.html',
            'components': '/informatique/composants-informatique.html',
            'gaming_components': '/gaming/composant-pc-gamer.html',
            'gaming_pc': '/gaming/gaming-pc.html',
            'graphics_cards': '/informatique/composants-informatique/carte-graphique.html',
            'processors': '/informatique/composants-informatique/processeur.html',
            'monitors': '/informatique/ordinateur-de-bureau/ecran.html'
        }

        if category_key not in categories:
            logger.error(f"Unknown category: {category_key}")
            return []

        category_url = categories[category_key]
        return self.scrape_with_pagination(category_url, max_pages)

    def save_to_json(self, products: List[Dict], filename: str) -> Path:
        """Save to JSON"""
        filepath = self.config.RAW_DATA_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        logger.success(f" Saved to: {filepath}")
        return filepath

    def save_to_csv(self, products: List[Dict], filename: str) -> Path:
        """Save to CSV"""
        import pandas as pd
        filepath = self.config.RAW_DATA_DIR / filename
        df = pd.DataFrame(products)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.success(f" Saved to: {filepath}")
        return filepath
