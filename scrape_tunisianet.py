"""
Main script to scrape Tunisianet
Run this file to start scraping
"""

from scrapers.tunisia.tunisianet_scraper import TunisianetScraper
from config import SCRAPING_CONFIG, LOGGING_CONFIG
from loguru import logger
from datetime import datetime
import sys


def setup_logging():
    """Configure logging"""
    logger.remove()  # Remove default handler

    # Console output
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <level>{message}</level>",
        level=LOGGING_CONFIG['level']
    )

    # File output
    logger.add(
        LOGGING_CONFIG['file'],
        format=LOGGING_CONFIG['format'],
        level="DEBUG",
        rotation="10 MB"
    )


def main():
    """Main scraping function"""

    # Setup logging
    setup_logging()

    logger.info("=" * 70)
    print("🚀 TUNISIANET SCRAPER - Starting")
    logger.info("🚀 TUNISIANET SCRAPER - Starting")
    logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Initialize scraper
    config = SCRAPING_CONFIG['tunisianet']
    scraper = TunisianetScraper(config)

    # Choose what to scrape
    categories_to_scrape = [
        ('laptops', 10),  # Category key, max pages
        ('gaming_laptops', 10),
        ('graphics_cards', 10),
        ('processors', 10),
        ('monitors', 10)
    ]

    all_products = []

    # Scrape each category
    for category_key, max_pages in categories_to_scrape:
        logger.info(f"\n\n{'#' * 70}")
        logger.info(f"# CATEGORY: {category_key.upper()}")
        logger.info(f"{'#' * 70}\n")

        products = scraper.scrape_category(category_key, max_pages=max_pages)
        all_products.extend(products)

        logger.info(f"✅ Category '{category_key}' complete: {len(products)} products\n")

    # Save results
    if all_products:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_file = scraper.save_to_json(
            all_products,
            f"tunisianet_products_{timestamp}.json"
        )

        # Save CSV
        csv_file = scraper.save_to_csv(
            all_products,
            f"tunisianet_products_{timestamp}.csv"
        )
        print(csv_file)

        # Also save as "latest"
        scraper.save_to_json(all_products, "tunisianet_latest.json")
        scraper.save_to_csv(all_products, "tunisianet_latest.csv")

        # Print summary
        scraper.print_summary(all_products)

        logger.success(f"\n✅ SCRAPING COMPLETE!")
        logger.success(f"📊 Total products scraped: {len(all_products)}")
        logger.success(f"💾 Saved to: {json_file.parent}")
    else:
        logger.error("❌ No products scraped!")

    logger.info(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Scraping interrupted by user")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")