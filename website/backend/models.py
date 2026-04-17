import os
from pathlib import Path
import sys
import torch
from torchvision import transforms
from PIL import Image
from fastapi import HTTPException #Permet de spécifier les erreurs avec raise

from queries import (
    sql_insert_user, sql_get_user_by_email, sql_get_user_by_id,sql_get_password_by_email,
    sql_get_models,
)


# ── USERS ──

def create_user(email: str, password: str, name: str):
    
    if sql_get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    
    user_id = sql_insert_user(email, password, name)
    return {"id": user_id, "email": email, "name": name}

def connection (email:str, password:str):
    user = sql_get_user_by_email(email)
    if sql_get_user_by_email(email) and user["password"]==password:
        return {"id": user["id"], "email": user["email"], "name": user["name"]}
    else :
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
        
# ── MODELS ──

#Scan et mise à jours par rapport au dossier de l'application
def scan_model():
    BASE_DIR = Path(__file__).resolve().parents[2]
    models_directory = BASE_DIR / "app/models"
    
    if not models_directory.exists():
        return []

    models_data = []

    for file in sorted(models_directory.glob("*.pth")):
        models_data.append({
            "name": file.name,
            "file_path": str(file)
        })

    return models_data

#Récupération des modèles actuellement en BD
def get_models():
    rows = sql_get_models()
    return [{"name": row["name"], "file_path": row["file_path"]} for row in rows]
    


# ── CROP ──

# Ajouter le chemin vers app/manual_crop
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "app" / "manual_crop"))
sys.path.insert(0, str(BASE_DIR / "app" / "face_swap"))

# Importer le modèle Regressor
from model import Regressor
from swapper import face_swap_from_paths 

# Chemin vers le modèle de crop
CROP_MODEL_PATH = BASE_DIR / "app" / "models" / "model.pth"

def crop_face(image_path: str, output_path: str = None) -> str:
    
    if not os.path.exists(CROP_MODEL_PATH):
        raise FileNotFoundError(f"Modèle de crop introuvable : {CROP_MODEL_PATH}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Regressor()
    model.load_state_dict(torch.load(CROP_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    x = transform(img).unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        pred = model(x)[0]
    
    # Extraire les coordonnées
    cx_pct, cy_pct, size_pct = pred[0].item(), pred[1].item(), pred[2].item()
    
    # Calculer les coordonnées de crop
    width = size_pct * W
    height = size_pct * H * 1.2 
    
    center_x = cx_pct * W
    center_y = (cy_pct - 0.1) * H 
    
    x1 = max(0, int(center_x - width / 2))
    y1 = max(0, int(center_y - height / 2))
    x2 = min(W, int(center_x + width / 2))
    y2 = min(H, int(center_y + height / 2))
    
    # Déterminer le chemin de sortie
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_crop.jpg"
    
    # Cropper et sauvegarder
    img.crop((x1, y1, x2, y2)).save(output_path)
    print(f"✅ Recadrage réussi : {output_path}")
    
    return output_path



def crop_two_images(source_path: str, target_path: str) -> tuple[str, str]:
    
    print(f"🔄 Crop de l'image source : {source_path}")
    source_crop = crop_face(source_path)
    
    print(f"🔄 Crop de l'image target : {target_path}")
    target_crop = crop_face(target_path)
    
    print(f"✅ Deux images croppées avec succès")
    
    return source_crop, target_crop



def perform_face_swap(source_crop_path: str, target_crop_path: str, output_path: str = None) -> str:

    if not os.path.exists(source_crop_path):
        raise FileNotFoundError(f"Image source croppée introuvable : {source_crop_path}")
    
    if not os.path.exists(target_crop_path):
        raise FileNotFoundError(f"Image cible croppée introuvable : {target_crop_path}")
    
    # Déterminer le chemin de sortie
    if output_path is None:
        base_dir = Path(source_crop_path).parent
        output_path = str(base_dir / "result.jpg")
    
    # Effectuer le face swap
    print(f" Face swap : {source_crop_path} → {target_crop_path}")
    result_path = face_swap_from_paths(
        target_path=target_crop_path,
        source_path=source_crop_path,
        output_path=output_path,
        use_gpu=False  # CPU sur Render
    )
    
    return result_path