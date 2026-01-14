# PriceCheck TN - MLOps Pipeline

Application intelligente de comparaison de prix informatiques Tunisie vs France avec détection de faux avis par NLP.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Editer .env avec vos configurations
```

### 3. Démarrer les services
```bash
# MLflow
mlflow server --host 0.0.0.0 --port 5000

# Prefect
prefect server start

# API
python scripts/deploy.py
```

## 📁 Structure Recommandée

```
PriceCheckTN/
├── scraping/          # Scrapers (Playwright + BeautifulSoup)
├── nlp/               # Modèles NLP (BERT + XGBoost)
├── mlops/             # Pipeline MLOps (DVC, Prefect, MLflow)
├── api/               # API FastAPI
├── utils/             # Utilitaires
├── scripts/           # Scripts d'orchestration
├── tests/             # Tests
├── docs/              # Documentation
├── notebooks/         # Notebooks exploration/training
├── data/              # Données (versionnées DVC)
└── models/            # Modèles entraînés
```

## 🎯 Commandes Principales

```bash
# Lancer le scraping
python scripts/run_scraping.py

# Exécuter le pipeline complet
python mlops/run_pipeline.py

# Entraîner les modèles
python mlops/training/bert_training.py
python mlops/training/mlflow_training.py

# Lancer l'API
python scripts/deploy.py
```

## 🔧 Configuration

Variables d'environnement (`.env`) :
- `MLFLOW_TRACKING_URI`: http://localhost:5000
- `PREFECT_API_URL`: http://127.0.0.1:4200/api
- `MONGO_URI`: mongodb://localhost:27017

## 📊 Monitoring

- **MLflow**: http://localhost:5000
- **Prefect**: http://localhost:4200
- **API Docs**: http://localhost:8000/docs

## 🧪 Tests

```bash
python -m pytest tests/
```

## 📚 Documentation

- [CI/CD](docs/ci-cd.md)
- [Architecture](docs/architecture.md)
- [API](docs/api.md)