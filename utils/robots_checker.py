"""
Utility to check robots.txt compliance
"""

from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from loguru import logger
from functools import lru_cache


class RobotsChecker:
    """Check if URLs are allowed by robots.txt"""

    def __init__(self):
        self.parsers = {}

    @lru_cache(maxsize=100)
    def get_parser(self, base_url: str) -> RobotFileParser:
        """Get or create robots.txt parser for domain"""

        parsed = urlparse(base_url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self.parsers:
            robots_url = f"{domain}/robots.txt"

            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
                logger.info(f" Loaded robots.txt from {robots_url}")
            except Exception as e:
                logger.warning(f"Could not load robots.txt: {e}")
                # When robots.txt can't be loaded, create a new parser that allows everything
                parser = RobotFileParser()
                parser.set_url("")  # Empty URL means no robots.txt
                # RobotFileParser defaults to allowing all URLs when no rules are set

            self.parsers[domain] = parser

        return self.parsers[domain]

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be scraped"""
        try:
            parser = self.get_parser(url)
            allowed = parser.can_fetch(user_agent, url)

            if not allowed:
                logger.warning(f" robots.txt blocks: {url}")

            return allowed

        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            return True  # Default to allowing if check fails


# Global instance
robots_checker = RobotsChecker()
