"""
detector.py — Crop de visages avec YOLOv8
"""
import cv2
from pathlib import Path


def ensure_yolo_model():
    """Télécharge YOLOv8 dans app/models si absent."""
    BASE_DIR = Path(__file__).resolve().parents[1]
    model_path = BASE_DIR / "models" / "yolov8n.pt"
    
    if model_path.exists():
        return str(model_path)
    
    print(f"📥 Téléchargement de yolov8n.pt dans {model_path.parent}...")
    
    try:
        from ultralytics import YOLO
        
        # Télécharger avec Ultralytics (dans le cache)
        temp_model = YOLO('yolov8n.pt')
        
        # Copier depuis le cache vers app/models
        import shutil
        cache_path = temp_model.ckpt_path
        shutil.copy(cache_path, model_path)
        
        print(f"✅ Modèle copié dans {model_path}")
        
    except Exception as e:
        print(f"❌ Erreur téléchargement : {e}")
        # Fallback : téléchargement direct
        import urllib.request
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ Téléchargement direct réussi")
    
    return str(model_path)


def crop_face_yolo(image_path: str, output_path: str = None) -> str:
    """
    Détecte et crop le visage principal avec YOLOv8.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("pip install ultralytics")
    
    # Charger le modèle depuis app/models
    model_path = ensure_yolo_model()
    model = YOLO(model_path)
    
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


def crop_two_faces_yolo(image_a_path: str, image_b_path: str) -> tuple[str, str]:
    """
    Crop deux images avec YOLO.
    """
    crop_a = crop_face_yolo(image_a_path)
    crop_b = crop_face_yolo(image_b_path)
    
    return crop_a, crop_b