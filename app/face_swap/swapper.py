"""
swapper.py — Face swap avec InsightFace
"""
import cv2
import numpy as np
from pathlib import Path

_models_cache = {}  # Cache par nom de modèle


def ensure_model_downloaded():
    """Télécharge inswapper_128.onnx si absent."""
    import urllib.request
    
    model_dir = Path.home() / ".insightface" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "inswapper_128.onnx"
    
    if model_path.exists():
        return str(model_path)
    
    print(f"📥 Téléchargement de inswapper_128.onnx...")
    
    urls = [
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    ]
    
    for i, url in enumerate(urls):
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Téléchargement réussi")
            return str(model_path)
        except Exception as e:
            if i < len(urls) - 1:
                print(f"❌ Échec, tentative suivante...")
            else:
                raise RuntimeError(f"Impossible de télécharger le modèle")


def load_models(model_name='buffalo_l', use_gpu=False):
    """
    Charge les modèles InsightFace avec cache par modèle.
    
    Args:
        model_name: 'buffalo_l', 'buffalo_sc', etc.
    """
    global _models_cache
    
    # Vérifier si ce modèle est déjà en cache
    if model_name in _models_cache:
        print(f"✅ Modèle {model_name} déjà en cache")
        return _models_cache[model_name]
    
    print(f"🔄 Chargement de {model_name}...")
    
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError("pip install insightface onnxruntime")
    
    ctx_id = -1  # CPU uniquement
    
    # ════════════════════════════════════════════════════════════
    # Chercher le modèle dans app/models/face_swap
    # ════════════════════════════════════════════════════════════
    BASE_DIR = Path(__file__).resolve().parents[1]
    local_model_path = BASE_DIR / "models" / "face_swap" / model_name
    
    if local_model_path.exists():
        print(f"✅ Utilisation du modèle local : {local_model_path}")
        app = FaceAnalysis(name=model_name, root=str(BASE_DIR / "models" / "face_swap"))
    else:
        print(f"⚠️ Modèle local absent, téléchargement dans ~/.insightface/")
        app = FaceAnalysis(name=model_name)
    
    # Adapter det_size selon le modèle
    det_sizes = {
        'buffalo_s': (256, 256),
        'buffalo_sc': (320, 320),
        'buffalo_l': (640, 640)
    }
    det_size = det_sizes.get(model_name, (320, 320))
    
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    
    print("🔄 Préparation inswapper...")
    model_path = ensure_model_downloaded()
    swapper = get_model(model_path, providers=['CPUExecutionProvider'])
    
    # Mettre en cache
    _models_cache[model_name] = (app, swapper)
    
    print(f"✅ {model_name} chargé et mis en cache")
    return app, swapper


def get_aligned_face(app, image):
    """Détecte le visage principal."""
    faces = app.get(image)
    if len(faces) == 0:
        raise Exception("Aucun visage détecté")
    
    face = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]
    return face


def swap_faces(app, swapper, target_img, source_img):
    """Effectue le face swap."""
    target_face = get_aligned_face(app, target_img)
    source_face = get_aligned_face(app, source_img)
    
    result = swapper.get(
        target_img,
        target_face,
        source_face,
        paste_back=True
    )
    
    return result


def face_swap_from_paths(target_path: str, source_path: str, output_path: str, use_gpu: bool = False, model_name: str = 'buffalo_l'):
    """
    Face swap complet depuis les chemins de fichiers.
    
    Args:
        model_name: Nom du modèle InsightFace à utiliser
    """
    print(f"🔄 Chargement du modèle {model_name}...")
    app, swapper = load_models(model_name=model_name, use_gpu=use_gpu)
    
    print(f"🔄 Chargement des images...")
    target_img = cv2.imread(str(target_path))
    source_img = cv2.imread(str(source_path))
    
    if target_img is None:
        raise FileNotFoundError(f"Image cible introuvable : {target_path}")
    
    if source_img is None:
        raise FileNotFoundError(f"Image source introuvable : {source_path}")
    
    print("🔄 Face swap en cours...")
    result = swap_faces(app, swapper, target_img, source_img)
    
    cv2.imwrite(str(output_path), result)
    print(f"✅ Face swap terminé : {output_path}")
    
    return output_path