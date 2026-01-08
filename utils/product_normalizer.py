"""
Product Normalizer Utility
---------------------------
Functions for normalizing, extracting, and enriching product data

Author: PriceCheck TN Team
Date: 2026-01-04
"""

import re
from typing import Dict, Optional, List
import unicodedata


class ProductNormalizer:
    """Normalize and extract information from product data"""

    def __init__(self):
        # Common brands in Tunisian/French market
        self.brands = {
            'hp', 'dell', 'lenovo', 'asus', 'acer', 'msi', 'apple', 'samsung',
            'lg', 'microsoft', 'razer', 'alienware', 'gigabyte', 'corsair',
            'logitech', 'sony', 'toshiba', 'huawei', 'xiaomi', 'oneplus',
            'intel', 'amd', 'nvidia', 'seagate', 'western digital', 'wd',
            'kingston', 'sandisk', 'crucial', 'hyperx', 'benq', 'viewsonic',
            'philips', 'aoc', 'adata', 'cooler master', 'thermaltake', 'nzxt',
            'evga', 'zotac', 'palit', 'gainward', 'pny', 'galax', 'asrock',
            'biostar', 'sapphire', 'xfx', 'powercolor', 'his', 'club3d',
            'creative', 'trust', 'genius', 'rapoo', 'redragon', 'steelseries',
            'roccat', 'cougar', 'coolermaster', 'deepcool', 'be quiet', 'fractal',
            'antec', 'seasonic', 'chieftec', 'zalman', 'enermax', 'silverstone'
        }

        # Category keywords (order matters - more specific first)
        self.category_keywords = {
            'Laptop': ['laptop', 'portable', 'notebook', 'ultrabook', 'chromebook', 'netbook', 'ordinateur portable'],
            'Desktop': ['desktop', 'pc fixe', 'tour', 'ordinateur de bureau', 'unite centrale', 'gaming pc', 'workstation'],
            'Monitor': ['monitor', 'ecran', 'display', 'screen', 'moniteur', 'lcd', 'led'],
            'Keyboard': ['keyboard', 'clavier', 'mecanique', 'mechanical'],
            'Mouse': ['mouse', 'souris', 'gaming mouse'],
            'Headset': ['headset', 'casque', 'headphone', 'ecouteur', 'earphone', 'earbuds'],
            'Webcam': ['webcam', 'camera', 'web cam'],
            'Processor': ['processor', 'processeur', 'cpu', 'intel core', 'ryzen'],
            'Graphics Card': ['gpu', 'graphics card', 'carte graphique', 'video card', 'geforce', 'radeon', 'gtx', 'rtx', 'rx'],
            'RAM': ['ram', 'memory', 'memoire', 'ddr4', 'ddr5', 'dimm'],
            'Storage': ['ssd', 'hdd', 'hard drive', 'disque dur', 'storage', 'nvme', 'm.2', 'sata'],
            'Motherboard': ['motherboard', 'carte mere', 'mainboard', 'mobo'],
            'Power Supply': ['power supply', 'alimentation', 'psu', 'bloc alimentation'],
            'Case': ['case', 'boitier', 'tower', 'chassis', 'boîtier'],
            'Router': ['router', 'routeur', 'modem', 'wifi', 'access point'],
            'Printer': ['printer', 'imprimante', 'multifunction'],
            'Scanner': ['scanner', 'scanneur'],
            'Speaker': ['speaker', 'enceinte', 'haut-parleur', 'subwoofer'],
            'Microphone': ['microphone', 'micro', 'studio mic'],
            'Cable': ['cable', 'câble', 'hdmi', 'usb cable', 'displayport'],
            'Adapter': ['adapter', 'adaptateur', 'dongle'],
            'Charger': ['charger', 'chargeur', 'ac adapter'],
            'Battery': ['battery', 'batterie', 'pile', 'power bank'],
            'Cooling': ['cooling', 'ventilateur', 'fan', 'refroidissement', 'watercooling', 'ventirad', 'radiator'],
            'UPS': ['ups', 'onduleur', 'backup power'],
            'Network Card': ['network card', 'carte reseau', 'ethernet', 'wifi card'],
            'Sound Card': ['sound card', 'carte son', 'audio card'],
            'Tablet': ['tablet', 'tablette', 'ipad'],
            'Other': []
        }

        # Processor patterns
        self.processor_patterns = [
            r'intel\s+core\s+i[3579](?:-\d{4,5}[A-Z]*)?',
            r'amd\s+ryzen\s+[3579](?:\s+\d{4}[A-Z]*)?',
            r'intel\s+celeron(?:\s+\w+)?',
            r'intel\s+pentium(?:\s+\w+)?',
            r'amd\s+athlon(?:\s+\w+)?',
            r'apple\s+m[123](?:\s+(?:pro|max|ultra))?',
            r'intel\s+xeon(?:\s+\w+)?',
            r'amd\s+threadripper(?:\s+\w+)?',
            r'intel\s+core\s+ultra\s+[579]',
        ]

        # RAM patterns
        self.ram_patterns = [
            r'\b(\d+)\s*(?:gb|go)\s*(?:ram|memory|memoire)',
            r'\bram\s*[:\-]?\s*(\d+)\s*(?:gb|go)',
            r'\b(\d+)gb\s+ddr[45]',
            r'\bddr[45]\s+(\d+)gb',
            r'\b(\d+)\s*(?:gb|go)\s+ddr[45]'
        ]

        # Storage patterns
        self.storage_patterns = [
            r'\b(\d+)\s*(?:gb|go|tb|to)\s*(?:ssd|hdd|nvme|m\.?2)',
            r'(?:ssd|hdd|nvme)\s*[:\-]?\s*(\d+)\s*(?:gb|go|tb|to)',
            r'\b(\d+)\s*(?:gb|tb)\s*(?:ssd|hdd)',
            r'\b(\d+)\s*to\s*(?:ssd|hdd)'
        ]

        # Screen size patterns
        self.screen_patterns = [
            r'(\d{2}(?:\.\d)?)\s*(?:inch|pouces|po|")',
            r'(\d{2}(?:\.\d)?)\s*(?:")',
            r'(\d{2}(?:\.\d)?)\'\''
        ]

    def normalize_product_name(self, name: str) -> str:
        """
        Normalize product name for consistent processing

        Steps:
        1. Remove accents
        2. Lowercase
        3. Remove special characters
        4. Remove extra spaces

        Args:
            name: Original product name

        Returns:
            Normalized product name
        """
        if not name or not isinstance(name, str):
            return ""

        # Remove accents (é -> e, à -> a, etc.)
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])

        # Lowercase
        name = name.lower()

        # Remove special characters but keep alphanumeric, spaces, and common separators
        name = re.sub(r'[^a-z0-9\s\-_./]', ' ', name)

        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()

        return name

    def extract_brand(self, name: str) -> Optional[str]:
        """
        Extract brand from product name

        Strategy:
        1. Check against known brands list
        2. If not found, use first word

        Args:
            name: Product name

        Returns:
            Brand name (uppercase) or "Unknown"
        """
        if not name:
            return "Unknown"

        name_normalized = self.normalize_product_name(name)

        # Check for each known brand (exact match)
        for brand in sorted(self.brands, key=len, reverse=True):  # Check longer brands first
            # Match whole word
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, name_normalized):
                return brand.upper()

        # Fallback: Try to extract first word as potential brand
        words = name_normalized.split()
        if words:
            first_word = words[0]
            # Only use if it's a reasonable brand name (2+ chars, alphabetic)
            if len(first_word) > 2 and first_word.isalpha():
                return first_word.upper()

        return "Unknown"

    def extract_model(self, name: str, brand: Optional[str] = None) -> Optional[str]:
        """
        Extract model from product name

        Strategy:
        1. Remove brand from name
        2. Extract first few significant words
        3. Stop at common keywords (specs)

        Args:
            name: Product name
            brand: Extracted brand (optional)

        Returns:
            Model name or None
        """
        if not name:
            return None

        name_normalized = self.normalize_product_name(name)

        # Remove brand from name if provided
        if brand and brand != "Unknown":
            name_normalized = name_normalized.replace(brand.lower(), '').strip()

        # Extract first few significant words (likely the model)
        words = name_normalized.split()
        model_words = []

        # Keywords that indicate start of specs (stop extracting model)
        spec_keywords = {
            'intel', 'amd', 'nvidia', 'with', 'avec', 'ram', 'ssd', 'hdd',
            'gb', 'go', 'tb', 'to', 'ddr4', 'ddr5', 'ghz', 'core', 'ryzen',
            'geforce', 'radeon', 'inch', 'pouces', 'lcd', 'led'
        }

        for word in words[:6]:  # Check first 6 words
            # Stop at spec keywords
            if word in spec_keywords:
                break

            # Keep alphanumeric words with length > 1
            if len(word) > 1 and any(c.isalnum() for c in word):
                model_words.append(word)

            # Stop after collecting 3 good words
            if len(model_words) >= 3:
                break

        if model_words:
            return ' '.join(model_words).upper()

        return None

    def categorize_product(self, name: str) -> str:
        """
        Categorize product based on name

        Strategy:
        1. Check against category keywords (ordered by specificity)
        2. Return first match
        3. Default to "Other"

        Args:
            name: Product name

        Returns:
            Category name
        """
        if not name:
            return "Other"

        name_normalized = self.normalize_product_name(name)

        # Check each category (order matters!)
        for category, keywords in self.category_keywords.items():
            if category == 'Other':
                continue

            for keyword in keywords:
                if keyword in name_normalized:
                    return category

        return "Other"

    def extract_specs(self, name: str, description: str = "") -> Dict[str, str]:
        """
        Extract technical specifications from product name and description

        Extracts:
        - Processor (Intel Core i7, AMD Ryzen 5, etc.)
        - RAM (8GB, 16GB, etc.)
        - Storage (512GB SSD, 1TB HDD, etc.)
        - Screen size (15.6 inch, 27", etc.)

        Args:
            name: Product name
            description: Product description (optional)

        Returns:
            Dictionary of specifications
        """
        specs = {}

        # Combine name and description for better extraction
        text = f"{name} {description}".lower()

        # Extract processor
        for pattern in self.processor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                specs['processor'] = match.group(0).strip()
                break

        # Extract RAM
        for pattern in self.ram_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ram_size = match.group(1)
                specs['ram'] = f"{ram_size}GB"
                break

        # Extract storage
        for pattern in self.storage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                storage_text = match.group(0).strip().upper()
                # Normalize format
                storage_text = re.sub(r'\s+', ' ', storage_text)
                specs['storage'] = storage_text
                break

        # Extract screen size
        for pattern in self.screen_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                screen_size = match.group(1)
                specs['screen'] = f"{screen_size} inch"
                break

        return specs

    def create_signature(self, brand: Optional[str], model: Optional[str],
                        specs: Dict[str, str]) -> str:
        """
        Create unique signature for product matching

        Signature format: brand_model_processor_ram_storage
        Example: hp_pavilion15_i7_16gb_512gb

        This signature is used for fuzzy matching products across sources.

        Args:
            brand: Extracted brand
            model: Extracted model
            specs: Extracted specifications

        Returns:
            Product signature string
        """
        parts = []

        # Add brand (normalized)
        if brand and brand != "Unknown":
            brand_clean = re.sub(r'[^a-z0-9]', '', brand.lower())
            if brand_clean:
                parts.append(brand_clean)

        # Add model (normalized, limited length)
        if model:
            model_clean = re.sub(r'[^a-z0-9]', '', model.lower())
            if model_clean:
                parts.append(model_clean[:15])  # Limit to 15 chars

        # Add processor (simplified)
        if 'processor' in specs:
            proc = specs['processor'].lower()

            # Extract key identifier
            proc_match = re.search(r'(i[3579]|ryzen\s*[3579]|m[123]|celeron|pentium|athlon)', proc)
            if proc_match:
                proc_key = proc_match.group(1).replace(' ', '')
                parts.append(proc_key)

        # Add RAM size
        if 'ram' in specs:
            ram = re.search(r'(\d+)', specs['ram'])
            if ram:
                parts.append(f"{ram.group(1)}gb")

        # Add storage size and type
        if 'storage' in specs:
            storage = specs['storage'].lower()

            # Extract size
            size_match = re.search(r'(\d+)\s*(?:gb|tb)', storage)
            if size_match:
                size = size_match.group(1)
                unit = 'tb' if 'tb' in storage else 'gb'

                # Add type if present (ssd/hdd)
                storage_type = ''
                if 'ssd' in storage:
                    storage_type = 'ssd'
                elif 'hdd' in storage:
                    storage_type = 'hdd'

                storage_part = f"{size}{unit}"
                if storage_type:
                    storage_part += storage_type

                parts.append(storage_part)

        # Add screen size (if present)
        if 'screen' in specs:
            screen = re.search(r'(\d+(?:\.\d)?)', specs['screen'])
            if screen:
                parts.append(f"{screen.group(1)}in")

        # Join parts with underscore
        if parts:
            signature = '_'.join(parts)
            # Ensure signature is not too long
            return signature[:100]  # Max 100 chars

        return "unknown"

    def extract_price_from_text(self, text: str) -> Optional[float]:
        """
        Extract price value from text string

        Handles formats like:
        - "2,499.00 TND"
        - "549,99 €"
        - "1 234.56"

        Args:
            text: Text containing price

        Returns:
            Price as float or None
        """
        if not text:
            return None

        # Remove currency symbols and text
        price_str = re.sub(r'[^\d.,\s-]', '', str(text))

        # Remove spaces
        price_str = price_str.replace(' ', '')

        # Handle different decimal separators
        if ',' in price_str and '.' not in price_str:
            # European format: 1.234,56
            price_str = price_str.replace(',', '.')
        elif ',' in price_str and '.' in price_str:
            # Mixed format
            if price_str.index(',') > price_str.index('.'):
                # European: 1.234,56
                price_str = price_str.replace('.', '').replace(',', '.')
            else:
                # US: 1,234.56
                price_str = price_str.replace(',', '')

        try:
            return float(price_str)
        except:
            return None

    def similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity score between two texts

        Uses token-based comparison

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0

        # Normalize both texts
        norm1 = self.normalize_product_name(text1)
        norm2 = self.normalize_product_name(text2)

        # Tokenize
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        if not union:
            return 0.0

        return len(intersection) / len(union)