import gdown
import os
import torch

model_weights = torch.load("models/osnet_x0_25_market1501.pt")

# Crée le dossier models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# URL du modèle OSNet (convertie pour gdown)
url = "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"
output = "models/osnet_x0_25_market1501.pt"

print("📥 Téléchargement du modèle OSNet...")
gdown.download(url, output, quiet=False)
print("✅ Modèle téléchargé dans:", output)
