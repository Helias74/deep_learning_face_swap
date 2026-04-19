"""
swapper.py — Face swap avec InsightFace
"""
import cv2
import numpy as np
from pathlib import Path

# Cache global pour les modèles (lazy loading)
_models_cache = None


def ensure_model_downloaded():
    """Télécharge inswapper_128.onnx si absent."""
    import urllib.request
    
    model_dir = Path.home() / ".insightface" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "inswapper_128.onnx"
    
    if model_path.exists():
        print(f"✅ Modèle inswapper déjà présent")
        return str(model_path)
    
    print(f"📥 Téléchargement de inswapper_128.onnx (~538 MB)...")
    
    urls = [
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    ]
    
    for i, url in enumerate(urls):
        try:
            print(f"   Tentative {i+1}/{len(urls)}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Téléchargement réussi")
            return str(model_path)
        except Exception as e:
            print(f"❌ Échec tentative {i+1}: {e}")
            if i < len(urls) - 1:
                print(f"🔄 Tentative avec URL suivante...")
            else:
                raise RuntimeError(f"Impossible de télécharger le modèle")


def load_models(use_gpu=False):
    """
    Charge les modèles InsightFace avec cache global.
    """
    global _models_cache
    
    if _models_cache is not None:
        print("✅ Modèles déjà en cache")
        return _models_cache
    
    print("🔄 Premier chargement des modèles InsightFace...")
    
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError("pip install insightface onnxruntime")
    
    ctx_id = -1  # CPU uniquement
    
    # Utiliser buffalo_l (meilleure qualité que buffalo_sc)
    print("🔄 Chargement FaceAnalysis (buffalo_l)...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    print("🔄 Préparation inswapper...")
    model_path = ensure_model_downloaded()
    swapper = get_model(model_path, providers=['CPUExecutionProvider'])
    
    _models_cache = (app, swapper)
    
    print("✅ Modèles chargés et mis en cache")
    return app, swapper


def get_aligned_face(app, image):
    """Détecte le visage principal."""
    faces = app.get(image)
    if len(faces) == 0:
        raise Exception("Aucun visage détecté dans l'image")
    
    # Prendre le visage le plus grand
    face = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]
    return face


def swap_faces(app, swapper, target_img, source_img):
    """
    Effectue le face swap.
    
    Args:
        app: Modèle FaceAnalysis
        swapper: Modèle inswapper
        target_img: Image cible (numpy BGR)
        source_img: Image source (numpy BGR)
    
    Returns:
        Image résultat (numpy BGR)
    """
    target_face = get_aligned_face(app, target_img)
    source_face = get_aligned_face(app, source_img)
    
    result = swapper.get(
        target_img,
        target_face,
        source_face,
        paste_back=True
    )
    
    return result


def face_swap_from_paths(target_path: str, source_path: str, output_path: str, use_gpu: bool = False):
    """
    Face swap complet depuis les chemins de fichiers.
    
    Args:
        target_path: Chemin image cible
        source_path: Chemin image source
        output_path: Chemin de sortie
        use_gpu: Utiliser GPU (False sur Render)
    
    Returns:
        str: Chemin vers l'image résultat
    """
    print("🔄 Chargement des modèles InsightFace...")
    app, swapper = load_models(use_gpu=use_gpu)
    
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