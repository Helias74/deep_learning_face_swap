import torch
from torchvision import datasets, transforms
from model import SimpleCNN, HumanCNN 
import os
import shutil
from PIL import Image

# =========================
# Paramètres
# =========================
BATCH_SIZE = 32
DEVICE = "cpu"

# =========================
# Transformations
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# Dataset & DataLoader
# =========================
val_dataset = datasets.ImageFolder("data/val", transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

class_names = val_dataset.classes  # ['cats', 'dogs']

# =========================
# Nettoyage du dossier erreurs
# =========================
ERROR_DIR = "errors"


# créer le dossier errors s'il n'existe pas
os.makedirs(ERROR_DIR, exist_ok=True)

for cls in class_names:
    path = os.path.join(ERROR_DIR, cls)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# =========================
# Chargement du modèle
# =========================
model = HumanCNN()
#model.load_state_dict(torch.load("human_cnn_v1.pth", map_location=DEVICE))
torch.save(model.state_dict(), "human_cnn_v1.pth")
model.eval()

# =========================
# Évaluation
# =========================
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for i in range(len(labels)):
            total += 1
            true_label = labels[i].item()
            pred_label = predictions[i].item()

            if true_label == pred_label:
                correct += 1
            else:
                # chemin original de l'image
                img_path, _ = val_dataset.samples[total - 1]
                img = Image.open(img_path).convert("RGB")

                # sauvegarde dans errors/<classe_reelle>/
                class_name = class_names[true_label]
                filename = os.path.basename(img_path)
                save_path = os.path.join(ERROR_DIR, class_name, filename)

                img.save(save_path)

# =========================
# Résultat
# =========================
accuracy = 100 * correct / total
print(f"Accuracy sur le jeu de validation : {accuracy:.2f}%")
print(f"Images mal classées sauvegardées dans le dossier 'errors/'")