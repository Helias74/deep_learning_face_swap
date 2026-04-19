"""
detector.py — Crop de visages avec YOLOv8
"""
import cv2
from pathlib import Path


def crop_face_yolo(image_path: str, output_path: str = None, model_name: str = 'yolov8n') -> str:
    """
    Détecte et crop le visage principal avec YOLOv8.
    
    Args:
        image_path: Chemin de l'image d'entrée
        output_path: Chemin de sortie (optionnel)
        model_name: Nom du modèle ('yolov8n', 'model', etc.)
    
    Returns:
        str: Chemin de l'image croppée
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")
    
    # ════════════════════════════════════════════════════════════
    # Charger le bon modèle selon le choix
    # ════════════════════════════════════════════════════════════
    BASE_DIR = Path(__file__).resolve().parents[1]
    
    # Chercher le modèle dans app/models/crop/
    model_path = BASE_DIR / "models" / "crop" / f"{model_name}.pt"
    
    if model_path.exists():
        print(f"✅ Utilisation du modèle : {model_path}")
        model = YOLO(str(model_path))
    else:
        # Fallback : téléchargement automatique si c'est un modèle standard
        print(f"⚠️ Modèle local {model_path} absent, utilisation standard")
        model = YOLO(f'{model_name}.pt')
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable : {image_path}")
    
    H, W = image.shape[:2]
    
    # Détection (classe 0 = personne)
    results = model(image, verbose=False, classes=[0])
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        raise Exception("Aucune personne détectée")
    
    # Prendre la détection avec le meilleur score
    best_box = max(boxes, key=lambda b: b.conf[0])
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
    
    # Crop la partie supérieure (visage)
    height = y2 - y1
    y2_face = y1 + int(height * 0.4)
    
    # Ajouter une marge
    margin = 0.1
    width = x2 - x1
    
    x1 = max(0, int(x1 - width * margin))
    y1 = max(0, int(y1 - height * margin * 0.5))
    x2 = min(W, int(x2 + width * margin))
    y2_face = min(H, int(y2_face + height * margin * 0.5))
    
    # Crop
    cropped = image[y1:y2_face, x1:x2]
    
    # Sauvegarder
    if output_path is None:
        base = Path(image_path)
        output_path = str(base.parent / f"{base.stem}_crop{base.suffix}")
    
    cv2.imwrite(output_path, cropped)
    
    return output_path


def crop_two_faces_yolo(image_a_path: str, image_b_path: str, model_name: str = 'yolov8n') -> tuple[str, str]:
    """
    Crop deux images avec YOLO.
    
    Args:
        image_a_path: Chemin image A
        image_b_path: Chemin image B
        model_name: Nom du modèle à utiliser
    
    Returns:
        tuple: (chemin_a_crop, chemin_b_crop)
    """
    crop_a = crop_face_yolo(image_a_path, model_name=model_name)
    crop_b = crop_face_yolo(image_b_path, model_name=model_name)
    
    return crop_a, crop_b