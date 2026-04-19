import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(BASE_DIR / "app" / "yolo_crop"))
sys.path.insert(0, str(BASE_DIR / "app" / "face_swap"))
sys.path.insert(0, str(BASE_DIR / "app" / "manual_crop"))

from detector import crop_two_faces_yolo
from swapper import face_swap_from_paths
from cropper import crop_two_faces_manual


def crop_two_images(image_a_path: str, image_b_path: str, crop_model: str = 'yolov8n') -> tuple[str, str]:
    """
    Crop deux images avec le modèle spécifié.
    """
    print(f"🔧 Modèle de crop sélectionné : {crop_model}")
    
    if crop_model == 'model':
        return crop_two_faces_manual(image_a_path, image_b_path)
    else:
        return crop_two_faces_yolo(image_a_path, image_b_path, model_name=crop_model)


def perform_face_swap(source_crop_path: str, target_crop_path: str, swap_model: str = 'buffalo_l', output_path: str = None) -> str:
    """
    Effectue le face swap avec le modèle spécifié.
    """
    if not os.path.exists(source_crop_path):
        raise FileNotFoundError(f"Image source introuvable : {source_crop_path}")
    
    if not os.path.exists(target_crop_path):
        raise FileNotFoundError(f"Image cible introuvable : {target_crop_path}")
    
    if output_path is None:
        base_dir = Path(source_crop_path).parent
        output_path = str(base_dir / "result.jpg")
    
    print(f"🔧 Modèle de swap sélectionné : {swap_model}")
    
    result_path = face_swap_from_paths(
        target_path=target_crop_path,
        source_path=source_crop_path,
        output_path=output_path,
        use_gpu=False,
        model_name=swap_model
    )
    
    return result_path


def create_user(email: str, password: str, name: str = None):
    from queries import sql_insert_user
    return sql_insert_user(email, password, name)


def connection(email: str, password: str):
    from queries import sql_get_password_by_email
    user = sql_get_password_by_email(email)
    if user and user["password"] == password:
        return {"id": user["id"], "email": user["email"], "name": user["name"]}
    return None


def get_models():
    from queries import sql_get_all_models
    return sql_get_all_models()


def scan_model():
    """Scan les modèles dans app/models/ et les catégorise."""
    models = []
    
    # Scan crop models
    crop_dir = BASE_DIR / "app" / "models" / "crop"
    if crop_dir.exists():
        for file in crop_dir.iterdir():
            if file.suffix in ['.pt', '.pth']:
                models.append({
                    "name": file.stem,
                    "file_path": str(file),
                    "model_type": "crop"
                })
    
    # Scan face_swap models
    swap_dir = BASE_DIR / "app" / "models" / "face_swap"
    if swap_dir.exists():
        for model_folder in swap_dir.iterdir():
            if model_folder.is_dir() and model_folder.name.startswith("buffalo_"):
                models.append({
                    "name": model_folder.name,
                    "file_path": str(model_folder),
                    "model_type": "face_swap"
                })
    
    return models