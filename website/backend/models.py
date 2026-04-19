import sys
import os
from pathlib import Path

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parents[2]

# Ajouter les chemins
sys.path.insert(0, str(BASE_DIR / "app" / "yolo_crop"))
sys.path.insert(0, str(BASE_DIR / "app" / "face_swap"))

from detector import crop_two_faces_yolo
from swapper import face_swap_from_paths


def crop_two_images(image_a_path: str, image_b_path: str) -> tuple[str, str]:
    """
    Crop deux images avec YOLOv8.
    """
    return crop_two_faces_yolo(image_a_path, image_b_path)


def perform_face_swap(source_crop_path: str, target_crop_path: str, output_path: str = None) -> str:
    """
    Effectue le face swap entre deux images déjà croppées.
    """
    if not os.path.exists(source_crop_path):
        raise FileNotFoundError(f"Image source introuvable : {source_crop_path}")
    
    if not os.path.exists(target_crop_path):
        raise FileNotFoundError(f"Image cible introuvable : {target_crop_path}")
    
    if output_path is None:
        base_dir = Path(source_crop_path).parent
        output_path = str(base_dir / "result.jpg")
    
    result_path = face_swap_from_paths(
        target_path=target_crop_path,
        source_path=source_crop_path,
        output_path=output_path,
        use_gpu=False
    )
    
    return result_path


def create_user(email: str, password: str, name: str = None):
    from queries import insert_user
    return insert_user("faceswap.db", email, password, name)


def connection(email: str, password: str):
    from queries import verify_user
    return verify_user("faceswap.db", email, password)


def get_models():
    from queries import get_all_models
    return get_all_models("faceswap.db")


def scan_model():
    models = []
    models_dir = BASE_DIR / "app" / "models"
    
    if models_dir.exists():
        for file in models_dir.iterdir():
            if file.suffix in ['.pt', '.pth', '.onnx']:
                models.append({
                    "name": file.stem,
                    "path": str(file),
                    "type": file.suffix[1:]
                })
    
    return models