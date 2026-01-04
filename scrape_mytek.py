"""
MyTek scraper - Using Playwright (necessary for JS-loaded content)
"""

from scrapers.tunisia.mytek_hybrid_scraper import MyTekHybridScraper
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

    logger.info("="*70)
    logger.info("🚀 MYTEK SCRAPER STARTING")
    logger.info(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    config = SCRAPING_CONFIG['mytek']
    scraper = MyTekHybridScraper(config)

    # Start conservative
    categories_to_scrape = [
        ('laptops', 10),
        ('gaming_laptops', 10),
        ('components', 10),
        ('graphics_cards', 10),
        ('processors', 10),
        ('monitors', 10),

    ]

    all_products = []

    for category_key, max_pages in categories_to_scrape:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# CATEGORY: {category_key.upper()}")
        logger.info(f"{'#'*70}\n")

        products = scraper.scrape_category(category_key, max_pages=max_pages)
        all_products.extend(products)

        logger.info(f"✅ Complete: {len(products)} products\n")

    if all_products:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scraper.save_to_json(all_products, f"mytek_products_{timestamp}.json")
        scraper.save_to_csv(all_products, f"mytek_products_{timestamp}.csv")
        scraper.save_to_json(all_products, "mytek_latest.json")
        scraper.save_to_csv(all_products, "mytek_latest.csv")

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