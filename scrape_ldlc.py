"""
LDLC scraper - Playwright version (anti-bot bypass)
"""

from scrapers.france.ldlc_hybrid_scraper import LDLCHybridScraper
from config import SCRAPING_CONFIG, LOGGING_CONFIG
from loguru import logger
from datetime import datetime
import sys


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(LOGGING_CONFIG['file'], format=LOGGING_CONFIG['format'], level="DEBUG", rotation="10 MB")


def main():
    setup_logging()

    print("\n" + "=" * 70)
    print("🇫🇷 LDLC SCRAPER - Playwright Version")
    print("=" * 70)
    print("  ℹ️  LDLC has anti-bot protection")
    print("  ℹ️  Using Playwright (headless)")
    print("  ⚠️  Slower but necessary")
    print("=" * 70)

    logger.info("=" * 70)
    logger.info("🚀 LDLC SCRAPER STARTING")
    logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    config = SCRAPING_CONFIG['ldlc']
    scraper = LDLCHybridScraper(config)

    categories_to_scrape = [
        ('laptops', 10),  # Start with 2 pages
        ('processors', 10),
        ('gaming_laptops', 10),
        ('graphics_cards', 10),

    ]

    all_products = []

    for category_key, max_pages in categories_to_scrape:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# CATEGORY: {category_key.upper()}")
        logger.info(f"{'#' * 70}\n")

        products = scraper.scrape_category(category_key, max_pages=max_pages)
        all_products.extend(products)

        logger.info(f"✅ Complete: {len(products)} products\n")

    if all_products:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraper.save_to_json(all_products, f"ldlc_products_{timestamp}.json")
        scraper.save_to_csv(all_products, f"ldlc_products_{timestamp}.csv")
        scraper.save_to_json(all_products, "ldlc_latest.json")
        scraper.save_to_csv(all_products, "ldlc_latest.csv")

        logger.success(f"\n✅ SUCCESS! {len(all_products)} products")
    else:
        logger.error("❌ No products scraped!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted")
    except Exception as e:
        logger.exception(f"❌ Error: {e}")