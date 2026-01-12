"""\
LDLC scraper - Playwright version (anti-bot bypass)\
"""

from scrapers.france.ldlc_hybrid_scraper import LDLCHybridScraper
from config import config
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
    logger.add(config.LOGS_DIR / "ldlc_scraper.log",
              format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
              level="DEBUG", rotation="10 MB")

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

    # Initialize scraper
    scraper = LDLCHybridScraper(config)

    categories_to_scrape = [
        ('laptops', 2),  # Start with 2 pages
        ('processors', 2),
        ('gaming_laptops', 2),
        ('graphics_cards', 2),

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
    max_retries = 3
    retry_count = 0

    while retry_count <= max_retries:
        try:
            main()
            break  # Success, exit loop
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"❌ Attempt {retry_count}/{max_retries} failed: {e}")
                logger.info(f"🔄 Retrying in 5 seconds...")
                import time
                time.sleep(5)
            else:
                logger.error(f"❌ All {max_retries} attempts failed. Giving up.")
                logger.exception("Final error details:")
                sys.exit(1)
