import sys
import os
import torch
from torchvision import transforms
from PIL import Image

from model import Regressor

MODEL_PATH = "model.pth"

def crop_instant(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regressor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)[0]
    
    cx_pct, cy_pct, size_pct = pred[0].item(), pred[1].item(), pred[2].item()
    
    width = size_pct * W
    height = size_pct * H * 1.2 
    
    center_x = cx_pct * W
    center_y = (cy_pct - 0.1) * H 
    
    x1 = max(0, int(center_x - width / 2))
    y1 = max(0, int(center_y - height / 2))
    x2 = min(W, int(center_x + width / 2))
    y2 = min(H, int(center_y + height / 2))

    output_name = image_path.replace(".jpg", "_crop.jpg")
    img.crop((x1, y1, x2, y2)).save(output_name)
    print(f"✅ Recadrage instantané réussi : {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilisation : python crop.py mon_image.jpg")
        sys.exit(1)
    crop_instant(sys.argv[1])
