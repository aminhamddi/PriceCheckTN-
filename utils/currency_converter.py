"""
Currency Converter Utility - FIXED VERSION
"""

import requests
from typing import Optional
from datetime import datetime, timedelta
from loguru import logger


class CurrencyConverter:
    """Convert currencies using frankfurter.app API (reliable & free)"""

    def __init__(self):
        # Use frankfurter.app (more reliable than exchangerate.host)
        self.api_url = "https://api.frankfurter.app/latest"
        self.cache = {}
        self.cache_duration = timedelta(hours=24)

        # Manual fallback rates (as of January 2026)
        self.manual_rates = {
            'TND_EUR': 0.304,  # 1 TND ≈ 0.304 EUR
            'EUR_TND': 3.29,   # 1 EUR ≈ 3.29 TND
            'TND_USD': 0.32,
            'USD_TND': 3.12
        }

    def get_rate(self, from_currency: str, to_currency: str) -> float:
        """Get exchange rate"""

        # Special case: same currency
        if from_currency == to_currency:
            return 1.0

        cache_key = f"{from_currency}_{to_currency}"

        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                logger.debug(f"Using cached rate: {cached_data['rate']}")
                return cached_data['rate']

        # Try API first (only supports major currencies)
        if from_currency in ['EUR', 'USD', 'GBP'] and to_currency in ['EUR', 'USD', 'GBP', 'TND']:
            try:
                params = {
                    'from': from_currency,
                    'to': to_currency
                }

                response = requests.get(self.api_url, params=params, timeout=5)

                if response.status_code == 200:
                    data = response.json()

                    if 'rates' in data and to_currency in data['rates']:
                        rate = data['rates'][to_currency]

                        # Cache it
                        self.cache[cache_key] = {
                            'rate': rate,
                            'timestamp': datetime.now()
                        }

                        logger.info(f" API rate {from_currency}→{to_currency}: {rate:.4f}")
                        return rate

            except Exception as e:
                logger.warning(f" API failed: {e}")

        # Use manual fallback rates
        if cache_key in self.manual_rates:
            rate = self.manual_rates[cache_key]
            logger.info(f" Using manual rate {from_currency}→{to_currency}: {rate:.4f}")
            return rate

        # Last resort: calculate inverse
        inverse_key = f"{to_currency}_{from_currency}"
        if inverse_key in self.manual_rates:
            rate = 1.0 / self.manual_rates[inverse_key]
            logger.info(f" Using inverse rate {from_currency}→{to_currency}: {rate:.4f}")
            return rate

        # Absolute fallback
        logger.error(f" No rate found for {cache_key}, using 1.0")
        return 1.0

    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount"""
        if from_currency == to_currency:
            return amount

        rate = self.get_rate(from_currency, to_currency)
        result = amount * rate

        logger.debug(f"Convert: {amount} {from_currency} × {rate} = {result:.2f} {to_currency}")
        return result