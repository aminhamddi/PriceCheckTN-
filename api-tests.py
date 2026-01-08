import requests

BASE_URL = "https://reviewscheck-webapp-demo.azurewebsites.net"

# --- 1. Test root ---
resp = requests.get(f"{BASE_URL}/")
print("Root endpoint:", resp.status_code, resp.text)

# --- 2. Test health ---
resp = requests.get(f"{BASE_URL}/health")
print("Health endpoint:", resp.status_code, resp.json())

# --- 3. Test single prediction ---
review = {
    "text": "Excellent produit, je recommande !",
    "method": "bert_only"
}

resp = requests.post(f"{BASE_URL}/predict", json=review)
print("Predict endpoint (single review):", resp.status_code, resp.json())

# --- 4. Test batch prediction ---
batch_reviews = {
    "reviews": [
        {"text": "Super produit, livraison rapide", "method": "bert_only"},
        {"text": "Je n'ai jamais reçu ma commande", "method": "ensemble"},
        {"text": "Très mauvais service, arnaque", "method": "sklearn_only"}
    ]
}

resp = requests.post(f"{BASE_URL}/predict/batch", json=batch_reviews)
print("Predict endpoint (batch):", resp.status_code, resp.json())

# --- 5. Test model info ---
resp = requests.get(f"{BASE_URL}/model/info")
print("Model info endpoint:", resp.status_code, resp.json())
