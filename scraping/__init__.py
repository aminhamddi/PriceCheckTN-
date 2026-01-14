"""Scraping module for PriceCheck TN"""

from .base_scraper import HybridScraper

# Import hybrid scrapers (the ones that actually exist)
try:
    from .france.ldlc_hybrid_scraper import LDLCHybridScraper
except ImportError:
    LDLCHybridScraper = None

try:
    from .tunisia.mytek_scraper import MyTekHybridScraper
except ImportError:
    MyTekHybridScraper = None

try:
    from .tunisia.tunisianet_scraper import TunisianetScraper
except ImportError:
    TunisianetScraper = None

# Legacy imports (for backward compatibility)
try:
    from .france.ldlc_scraper import LdlcScraper
except ImportError:
    LdlcScraper = None

__all__ = [
    "HybridScraper",
    "LDLCHybridScraper",
    "MyTekHybridScraper",
    "TunisianetScraper",
    "LdlcScraper"
]