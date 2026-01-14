"""
Hybrid scraper: Playwright for navigation + BeautifulSoup for parsing
For sites with JavaScript/dynamic content but static product HTML
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from playwright.sync_api import sync_playwright, Page, Browser
from bs4 import BeautifulSoup
import time
import random
from loguru import logger
import hashlib
from utils.robots_checker import robots_checker


class HybridScraper(ABC):
    """
    Base class for hybrid scraping approach:
    - Playwright handles page loading, navigation, JavaScript
    - BeautifulSoup parses the static HTML (faster, simpler)
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['base_url']

        # Playwright settings
        self.headless = config['playwright']['headless']
        self.slow_mo = config['playwright']['slow_mo']
        self.timeout = config['playwright']['timeout']
        self.wait_after_load = config['playwright']['wait_after_load']

        # Rate limiting
        self.last_request_time = 0
        self.min_delay = config['rate_limit']['min_delay']
        self.max_delay = config['rate_limit']['max_delay']
        self.page_delay = config['rate_limit']['page_delay']

        # Pagination
        self.max_pages = config['pagination']['max_pages']

        # Anti-blocking
        self.robots_checker = robots_checker
        self.session_product_count = 0
        self.max_products_per_session = config['anti_block']['max_products_per_session']

        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]

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
        logger.debug(f" Sleeping {delay:.1f}s...")
        time.sleep(delay)

    def install_stealth_scripts(self, page: Page):
        """Install anti-detection JavaScript"""
        # Remove webdriver flag
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
        """)

        # Add chrome object
        page.add_init_script("""
            window.chrome = { runtime: {} };
        """)

        # Fake plugins
        page.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """)

        # Fake languages
        page.add_init_script("""
            Object.defineProperty(navigator, 'languages', {
                get: () => ['fr-FR', 'fr', 'ar-TN', 'ar', 'en-US', 'en']
            });
        """)

    def human_like_scroll(self, page: Page):
        """Simulate human scrolling"""
        try:
            # Scroll down in chunks
            for i in range(3):
                scroll_amount = random.randint(300, 600)
                page.evaluate(f"""
                    window.scrollBy({{
                        top: {scroll_amount},
                        behavior: 'smooth'
                    }});
                """)
                time.sleep(random.uniform(0.5, 1.2))

            # Scroll back up slightly (humans do this)
            page.evaluate("""
                window.scrollBy({
                    top: -150,
                    behavior: 'smooth'
                });
            """)
            time.sleep(random.uniform(0.3, 0.7))

        except Exception as e:
            logger.debug(f"Scroll simulation failed: {e}")

    def get_browser_context(self, playwright):
        """Create browser with stealth settings"""
        browser = playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        context = browser.new_context(
            user_agent=random.choice(self.user_agents),
            viewport={'width': 1920, 'height': 1080},
            locale='fr-FR',
            timezone_id='Africa/Tunis',
            geolocation={'latitude': 36.8065, 'longitude': 10.1815},  # Tunis
            permissions=['geolocation'],
            extra_http_headers={
                'Accept-Language': 'fr-FR,fr;q=0.9,ar-TN;q=0.8,en;q=0.6',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )

        return browser, context

    def fetch_page_html(self, url: str) -> Optional[str]:
        """
        Use Playwright to load page and return HTML

        Args:
            url: Page URL

        Returns:
            HTML content as string or None if failed
        """
        # Check robots.txt
        if not self.robots_checker.can_fetch(url):
            logger.error(f" Blocked by robots.txt: {url}")
            return None

        # Check session limits
        if self.session_product_count >= self.max_products_per_session:
            logger.warning(
                f"  Session limit reached ({self.max_products_per_session} products). "
                f"Stop to avoid blocking."
            )
            return None

        # Rate limiting
        self.respect_rate_limit()

        try:
            with sync_playwright() as p:
                browser, context = self.get_browser_context(p)
                page = context.new_page()

                # Install stealth
                self.install_stealth_scripts(page)

                logger.info(f" Navigating to: {url}")

                # Navigate
                response = page.goto(
                    url,
                    wait_until='domcontentloaded',
                    timeout=self.timeout
                )

                # Check response
                if response and response.status >= 400:
                    logger.error(f" HTTP {response.status}")
                    if response.status in [403, 429]:
                        logger.error(" BLOCKED! Stop immediately.")
                    browser.close()
                    return None

                # Wait for dynamic content
                self.wait_for_content(page)

                # Human behavior simulation
                self.human_like_scroll(page)

                # Additional wait
                page.wait_for_timeout(self.wait_after_load)

                # Get HTML
                html = page.content()

                # Check for block indicators
                if self._is_blocked(html):
                    logger.error(" Anti-bot detected in HTML!")
                    browser.close()
                    return None

                browser.close()

                logger.success(f" Got HTML ({len(html)} bytes)")
                return html

        except Exception as e:
            logger.error(f" Error fetching page: {e}")
            return None

    def _is_blocked(self, html: str) -> bool:
        """Check if response contains block indicators"""
        blocked_indicators = [
            'cloudflare',
            'captcha',
            'access denied',
            'blocked',
            'vous avez été bloqué',
            'unusual traffic',
            'bot detected'
        ]

        html_lower = html.lower()
        return any(indicator in html_lower for indicator in blocked_indicators)

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML with BeautifulSoup"""
        return BeautifulSoup(html, 'html.parser')

    def clean_price(self, price_str: str) -> Optional[float]:
        """Extract numeric price"""
        import re
        if not price_str:
            return None

        cleaned = re.sub(r'[^\d.,]', '', price_str)

        if ',' in cleaned and '.' in cleaned:
            if cleaned.rindex(',') > cleaned.rindex('.'):
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')

        try:
            return float(cleaned)
        except:
            return None

    def extract_product_id(self, url: str) -> str:
        """Extract product ID from URL"""
        import re
        patterns = [
            r'/(\d+)\.html',
            r'/(\d+)-',
            r'/p/(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return hashlib.md5(url.encode()).hexdigest()[:8]

    # Abstract methods to implement

    @abstractmethod
    def wait_for_content(self, page: Page):
        """Wait for page content to load"""
        pass

    @abstractmethod
    def parse_products_from_html(self, html: str) -> List[Dict]:
        """Parse products from HTML using BeautifulSoup"""
        pass

    @abstractmethod
    def build_page_url(self, category_url: str, page_num: int) -> str:
        """Build URL for specific page number"""
        pass