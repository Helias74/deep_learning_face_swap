import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from model import Regressor

IMAGES_DIR = "dataset_faces/Custom_Data/images/train"
LABELS_DIR = "dataset_faces/Custom_Data/labels/train"
MODEL_PATH = "model.pth"

class FaceRegDataset(Dataset):
    def __init__(self):
        self.samples = []
        for fname in os.listdir(IMAGES_DIR):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue
            
            img_path = os.path.join(IMAGES_DIR, fname)
            label_path = os.path.join(LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
            
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if len(lines) > 0:
                    parts = lines[0].split()
                    cx, cy = float(parts[1]), float(parts[2])
                    size = max(float(parts[3]), float(parts[4]))
                    self.samples.append((img_path, [cx, cy, size]))

        print(f"✅ Images d'entraînement trouvées : {len(self.samples)}")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(target, dtype=torch.float32)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(FaceRegDataset(), batch_size=16, shuffle=True)

    model = Regressor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss() 

    epochs = 15 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x)

            pred_center = pred[:, :2]
            target_center = y[:, :2]
            
            pred_size_sqrt = torch.sqrt(pred[:, 2] + 1e-6)
            target_size_sqrt = torch.sqrt(y[:, 2] + 1e-6)
            
            loss_center = criterion(pred_center, target_center)
            loss_size = criterion(pred_size_sqrt, target_size_sqrt)
            
            loss = (3.0 * loss_center) + loss_size

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - Erreur: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train_model()
