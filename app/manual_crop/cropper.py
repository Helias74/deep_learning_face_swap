"""
cropper.py — Crop manuel avec le modèle PyTorch custom
Utilise la même logique que crop.py
"""
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


def crop_face_manual(image_path: str, output_path: str = None) -> str:
    """
    Crop avec le modèle custom CNN.
    Utilise la formule testée et validée de crop.py
    """
    BASE_DIR = Path(__file__).resolve().parents[1]
    model_path = BASE_DIR / "models" / "crop" / "model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    
    # Charger le modèle
    from model import Regressor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regressor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Charger l'image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    
    # Transformation (SANS normalisation, comme dans crop.py)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    x = transform(img).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        pred = model(x)[0]
    
    # Extraire les 3 valeurs
    cx_pct = pred[0].item()
    cy_pct = pred[1].item()
    size_pct = pred[2].item()
    
    print(f"🔍 Prédiction : cx={cx_pct:.3f}, cy={cy_pct:.3f}, size={size_pct:.3f}")
    
    # ════════════════════════════════════════════════════════════
    # FORMULE EXACTE DE crop.py (testée et validée)
    # ════════════════════════════════════════════════════════════
    width = size_pct * W
    height = size_pct * H * 1.2  # 20% plus haut pour inclure tout le visage
    
    center_x = cx_pct * W
    center_y = (cy_pct - 0.1) * H  # Décalage de 10% vers le haut
    
    x1 = max(0, int(center_x - width / 2))
    y1 = max(0, int(center_y - height / 2))
    x2 = min(W, int(center_x + width / 2))
    y2 = min(H, int(center_y + height / 2))
    
    print(f"🔍 BBox : ({x1}, {y1}, {x2}, {y2})")
    print(f"🔍 Dimensions : {x2-x1}x{y2-y1}")
    
    # Crop
    cropped_img = img.crop((x1, y1, x2, y2))
    
    # Déterminer le chemin de sortie
    if output_path is None:
        base = Path(image_path)
        output_path = str(base.parent / f"{base.stem}_crop{base.suffix}")
    
    cropped_img.save(output_path)
    print(f"✅ Crop sauvegardé : {output_path}")
    
    return output_path


def crop_two_faces_manual(image_a_path: str, image_b_path: str) -> tuple[str, str]:
    """Crop deux images avec le modèle manuel."""
    print("🔧 Crop manuel avec model.pth")
    crop_a = crop_face_manual(image_a_path)
    crop_b = crop_face_manual(image_b_path)
    return crop_a, crop_b