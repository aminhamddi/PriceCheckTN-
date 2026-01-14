# 🚀 QUICK START - PriceCheckTN

## Installation

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 3. Vérifier l'installation
python -c "from nlp.prediction import FakeReviewPredictor; print('✅ NLP OK')"
python -c "from api.main import app; print('✅ API OK')"
```

## Utilisation

### 1. Scraping des données
```bash
python scripts/run_scraping.py --site all --output data/raw
```

### 2. Entraînement des modèles
```bash
# XGBoost
python mlops/training/mlflow_training.py

# BERT
python mlops/training/bert_training.py
```

### 3. Lancement de l'API
```bash
python scripts/deploy.py
# Accéder à http://localhost:8000/docs
```

### 4. Exécution du pipeline complet
```bash
python mlops/run_pipeline.py
```

## Structure du projet

```
PriceCheckTN/
├── scraping/          # Scrapers (France & Tunisie)
├── nlp/               # Modèles NLP & prédiction
├── mlops/             # MLOps (tracking, registry, orchestration)
├── api/               # API FastAPI
├── utils/             # Utilitaires (devise, fuzzy matching)
├── scripts/           # Scripts d'orchestration
├── tests/             # Tests
├── notebooks/         # Exploration & training
└── models/            # Modèles entraînés
```

## Commandes utiles

```bash
# Vérifier les modèles disponibles
python mlops/model_registry/cli.py list

# Lancer les tests
python -m pytest tests/

# Voir les logs MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Voir les tâches Prefect
prefect server start
```

## Prochaines étapes

1. ✅ **Fait** : Structure réorganisée
2. 🔄 **À faire** : Tester le pipeline complet
3. 🔄 **À faire** : Vérifier l'API
4. 🔄 **À faire** : Exécuter les tests

## Dépannage

### Modèle BERT introuvable
```bash
python mlops/training/bert_training.py
```

### API ne démarre pas
```bash
# Vérifier les dépendances
pip install -r requirements-api.txt
```

### DVC pipeline erreur
```bash
dvc repro
```

---

**Le projet est prêt à l'emploi !** 🎉