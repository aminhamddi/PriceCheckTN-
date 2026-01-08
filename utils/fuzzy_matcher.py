"""
Fuzzy Matcher Utility
---------------------
Multi-level product matching algorithm
"""

from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
import pandas as pd
from loguru import logger


class ProductMatcher:
    """Match products using multi-level fuzzy matching"""

    def __init__(self):
        self.match_threshold = 70  # Minimum score for match
        self.high_confidence_threshold = 85
        self.medium_confidence_threshold = 70

    def find_best_match(self, tn_product: pd.Series,
                       fr_products: pd.DataFrame) -> Optional[Dict]:
        """
        Find best matching French product for a Tunisia product
        """
        best_match = None
        best_score = 0
        best_method = None

        # Level 1: Exact signature match
        tn_signature = tn_product.get('product_signature', '')

        if tn_signature and tn_signature != 'unknown':
            exact_matches = fr_products[
                fr_products['product_signature'] == tn_signature
            ]

            if not exact_matches.empty:
                fr_product = exact_matches.iloc[0]
                best_match = fr_product
                best_score = 100.0
                best_method = 'exact_signature'

        # Level 2: Fuzzy string matching
        if best_match is None:
            tn_name = tn_product.get('product_name_normalized', '')

            if tn_name:
                for _, fr_product in fr_products.iterrows():
                    fr_name = fr_product.get('product_name_normalized', '')

                    if not fr_name:
                        continue

                    # Token sort ratio (ignore word order)
                    score = fuzz.token_sort_ratio(tn_name, fr_name)

                    if score > best_score:
                        best_score = score
                        best_match = fr_product
                        best_method = 'fuzzy_string'

        # Level 3: Specs-based matching
        if best_match is None or best_score < self.high_confidence_threshold:
            specs_match = self._match_by_specs(tn_product, fr_products)

            if specs_match and specs_match['score'] > best_score:
                best_score = specs_match['score']
                best_match = specs_match['product']
                best_method = 'specs_based'

        # Only return if above threshold
        if best_score < self.match_threshold:
            return None

        # Check if we actually found a match
        if best_match is None:
            return None

        # Determine confidence
        if best_score >= self.high_confidence_threshold:
            confidence = 'high'
        elif best_score >= self.medium_confidence_threshold:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Build match result
        match = {
            'match_id': f"match_{tn_product['product_id']}_{best_match['product_id']}",
            'tn_product': tn_product.to_dict(),
            'fr_product': best_match.to_dict(),
            'match_quality': {
                'match_score': round(best_score / 100, 3),
                'confidence': confidence,
                'match_method': best_method
            }
        }

        return match

    def _match_by_specs(self, tn_product: pd.Series,
                       fr_products: pd.DataFrame) -> Optional[Dict]:
        """Match based on specifications"""
        tn_brand = tn_product.get('brand', '')
        tn_specs = tn_product.get('specifications', '{}')

        if not tn_brand or tn_brand == 'Unknown':
            return None

        # Filter by same brand
        same_brand = fr_products[fr_products['brand'] == tn_brand]

        if same_brand.empty:
            return None

        # Try to parse specs
        try:
            import json
            tn_specs_dict = json.loads(tn_specs) if isinstance(tn_specs, str) else tn_specs
        except:
            tn_specs_dict = {}

        best_match = None
        best_score = 0

        for _, fr_product in same_brand.iterrows():
            fr_specs = fr_product.get('specifications', '{}')

            try:
                fr_specs_dict = json.loads(fr_specs) if isinstance(fr_specs, str) else fr_specs
            except:
                fr_specs_dict = {}

            # Calculate specs similarity
            score = self._calculate_specs_similarity(tn_specs_dict, fr_specs_dict)

            if score > best_score:
                best_score = score
                best_match = fr_product

        if best_score > self.match_threshold:
            return {
                'product': best_match,
                'score': best_score
            }

        return None

    def _calculate_specs_similarity(self, specs1: Dict, specs2: Dict) -> float:
        """Calculate similarity between two spec dictionaries"""
        if not specs1 or not specs2:
            return 0.0

        common_keys = set(specs1.keys()) & set(specs2.keys())

        if not common_keys:
            return 0.0

        matches = 0
        for key in common_keys:
            val1 = str(specs1[key]).lower()
            val2 = str(specs2[key]).lower()

            if val1 == val2:
                matches += 1

        similarity = (matches / len(common_keys)) * 100
        return similarity