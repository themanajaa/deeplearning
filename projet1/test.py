import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms

# Charger le modèle ONNX
session = ort.InferenceSession("mnist_model.onnx")

# Préparation de la transformation identique à celle du training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Ajuste selon ton training !
])

# Charger jeu de test MNIST (download si besoin)
test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)

# Prendre une image test avec son label (exemple: index 1)
img, label = test_dataset[1]

# Préparer l'entrée pour le modèle ONNX
input_name = session.get_inputs()[0].name

# MNIST : batch_size=1, canaux=1, hauteur=28, largeur=28
input_data = img.numpy().reshape(1, 1, 28, 28).astype(np.float32)

# Inférence
outputs = session.run(None, {input_name: input_data})

# Résultat brut (logits)
logits = outputs[0]

# Affichage des logits et prédiction finale
print("Logits:", logits)
predicted_label = np.argmax(logits)
print(f"Prédiction du modèle : {predicted_label}")
print(f"Label réel : {label}")
