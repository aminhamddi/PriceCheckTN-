"""
Configuration for scraping
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Scraping settings
SCRAPING_CONFIG = {
    'tunisianet': {
        'base_url': 'https://www.tunisianet.com.tn',
        'categories': {
            'laptops': '/108-pc-portable',
            'gaming_laptops': '/681-pc-portable-gamer',
            'components': '/110-composant-informatique',
            'graphics_cards': '/111-carte-graphique',
            'processors': '/112-processeur',
        },
        'rate_limit': {
            'min_delay': 2.0,
            'max_delay': 4.0,
            'page_delay': 3.0,
        },
        'pagination': {
            'max_pages': 5,
            'products_per_page': 20,
        },
        'headers': {
            'User-Agent': 'PriceCheckTN-Bot/1.0 (Student Research Project)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
        }
    },
'ldlc': {
    'base_url': 'https://www.ldlc.com',
    'categories': {
        'laptops': '/informatique/ordinateur-portable/pc-portable/c4265/',
        'gaming_laptops': '/informatique/ordinateur-portable/pc-portable-gamer/c4303/',
        'components': '/informatique/pieces-informatique/c4266/',
        'graphics_cards': '/informatique/pieces-informatique/carte-graphique-interne/c4684/',
        'processors': 'informatique/pieces-informatique/processeur/c4300/',
        'monitors': '/informatique/peripherique-pc/moniteur-pc/c4623/',
    },
    'rate_limit': {
        'min_delay': 3.0,
        'max_delay': 5.0,
        'page_delay': 4.0,
    },
    'pagination': {
        'max_pages': 10,
        'products_per_page': 48,
        'param_name': 'page',  # LDLC uses ?page=2
    },
    'headers': {
        'User-Agent': 'PriceCheckTN-Bot/1.0 (Student Research Project)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
    }
},

    # ========================================================================
    # MYTEK - Same simple structure as Tunisianet (Requests + BS4)
    # ========================================================================
    'mytek': {
        'base_url': 'https://www.mytek.tn',
        'categories': {
            'laptops': '/informatique/ordinateurs-portables.html',
            'gaming_laptops': '/gaming/gaming-pc.html',
            'components': '/informatique/composants-informatique.html',
            'graphics_cards': '/composants-informatique/carte-graphique.html',
            'processors': '/informatique/composants-informatique/processeur.html',
            'monitors': '/informatique/ordinateur-de-bureau/ecran.html',
        },
        'rate_limit': {
            'min_delay': 5.0,  # SLOWER than Tunisianet (was blocked)
            'max_delay': 8.0,
            'page_delay': 8.0,  # Long wait between pages
        },
        'pagination': {
            'max_pages': 3,  # Conservative: max 3 pages
            'products_per_page': 24,
            'param_name': 'p',  # MyTek uses ?p=2 for page 2
        },
        'headers': {
            'User-Agent': 'PriceCheckTN-Bot/1.0 (Academic Research; MyTek scraper)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,ar-TN;q=0.8,en;q=0.7',
        }
    },
}


# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'scraping.log'
}