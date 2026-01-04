"""
Base scraper class using requests + BeautifulSoup
For sites that don't require JavaScript
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import time
import random
from loguru import logger
import hashlib


class SimpleBaseScraper(ABC):
    """Base class for simple scrapers (requests + BeautifulSoup)"""

    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['base_url']

        # Setup session with headers
        self.session = requests.Session()
        self.session.headers.update(config['headers'])

        # Rate limiting
        self.last_request_time = 0
        self.min_delay = config['rate_limit']['min_delay']
        self.max_delay = config['rate_limit']['max_delay']

    def respect_rate_limit(self):
        """Enforce minimum time between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            wait_time = self.min_delay - elapsed
            logger.debug(f"⏳ Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def random_delay(self, min_sec: float = None, max_sec: float = None):
        """Random delay between requests"""
        min_sec = min_sec or self.min_delay
        max_sec = max_sec or self.max_delay

        delay = random.uniform(min_sec, max_sec)
        logger.debug(f"💤 Sleeping {delay:.1f}s...")
        time.sleep(delay)

    def fetch_page(self, url: str, timeout: int = 15) -> Optional[str]:
        """
        Fetch page HTML using requests

        Args:
            url: Page URL
            timeout: Request timeout in seconds

        Returns:
            HTML content or None if failed
        """
        self.respect_rate_limit()

        try:
            logger.info(f"🌐 Fetching: {url}")

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            logger.success(f"✅ Status {response.status_code} ({len(response.text)} bytes)")
            return response.text

        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP Error {e.response.status_code}: {url}")
            if e.response.status_code in [403, 429]:
                logger.warning("🚫 Possibly blocked or rate limited!")
            return None

        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout after {timeout}s: {url}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request error: {e}")
            return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML with BeautifulSoup"""
        return BeautifulSoup(html, 'html.parser')

    def clean_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price from string"""
        import re
        if not price_str:
            return None

        # Remove non-numeric except comma and dot
        cleaned = re.sub(r'[^\d,.]', '', price_str)

        # Handle different formats
        if ',' in cleaned and '.' in cleaned:
            if cleaned.rindex(',') > cleaned.rindex('.'):
                # European: 1.299,99
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US: 1,299.99
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Assume European decimal
            cleaned = cleaned.replace(',', '.')

        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Could not parse price: {price_str}")
            return None

    def extract_product_id(self, url: str) -> str:
        """Extract product ID from URL"""
        import re
        patterns = [
            r'/(\d+)-',  # Tunisianet style: /12345-product
            r'/(\d+)\.html',  # Other style
            r'/p/(\d+)',  # Generic
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # Fallback: hash of URL
        return hashlib.md5(url.encode()).hexdigest()[:8]

    @abstractmethod
    def scrape_page(self, url: str) -> List[Dict]:
        """Scrape a single page"""
        pass

    @abstractmethod
    def scrape_with_pagination(self, category_url: str, max_pages: int) -> List[Dict]:
        """Scrape multiple pages with pagination"""
        pass