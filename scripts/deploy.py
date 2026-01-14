#!/usr/bin/env python3
"""
Script de déploiement simplifié pour PriceCheck TN
"""

import subprocess
import sys

def run_command(cmd, description):
    print(f"\n {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f" {description} réussi")
    except subprocess.CalledProcessError as e:
        print(f" {description} échoué: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print(" DEPLOIEMENT PRICECHECK TN")
    print("=" * 60)
    
    # Vérifier dépendances
    run_command("python -c \"import mlflow, prefect, fastapi\" ", "Vérification dépendances")
    
    print("\n  Assurez-vous que MLflow et Prefect sont démarrés:")
    print("   mlflow server --host 0.0.0.0 --port 5000")
    print("   prefect server start")
    
    print("\n Lancement API FastAPI...")
    run_command("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload", "Démarrage API")
    
    print("\n Déploiement terminé!")
    print(" API disponible sur: http://localhost:8000")
    print(" Docs API: http://localhost:8000/docs")

if __name__ == "__main__":
    main()