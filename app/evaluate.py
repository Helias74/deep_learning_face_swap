import torch
from torchvision import datasets, transforms
from model import ShapeCNN
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
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_dataset.classes
num_classes = len(class_names)

# =========================
# Nettoyage du dossier erreurs
# =========================
ERROR_DIR = "errors"
os.makedirs(ERROR_DIR, exist_ok=True)

for cls in class_names:
    path = os.path.join(ERROR_DIR, cls)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

# =========================
# Chargement du modèle
# =========================
model = ShapeCNN(num_classes=num_classes)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# Évaluation
# =========================
correct = 0
total = 0
sample_idx = 0  # index réel dans le dataset

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predictions[i].item()

            if true_label == pred_label:
                correct += 1
            else:
                img_path, _ = val_dataset.samples[sample_idx]
                img = Image.open(img_path).convert("RGB")

                class_name = class_names[true_label]
                filename = os.path.basename(img_path)
                save_path = os.path.join(ERROR_DIR, class_name, filename)

                img.save(save_path)

            total += 1
            sample_idx += 1

# =========================
# Résultat
# =========================
accuracy = 100 * correct / total
print(f"Accuracy sur le jeu de validation : {accuracy:.2f}%")
print("Images mal classées sauvegardées dans le dossier 'errors/'")