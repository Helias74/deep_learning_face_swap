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
        print(f"✅ Modèle déjà présent")
        return str(model_path)
    
    print(f"📥 Téléchargement de inswapper_128.onnx (~538 MB, 2-5 min)...")
    
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
    Charge les modèles InsightFace (avec cache).
    Les modèles sont chargés UNE SEULE FOIS au premier appel.
    """
    global _models_cache
    
    # Si déjà chargés, retourner le cache
    if _models_cache is not None:
        print("✅ Modèles déjà en cache")
        return _models_cache
    
    print("🔄 Premier chargement des modèles InsightFace...")
    
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
    except ImportError:
        raise ImportError("pip install insightface onnxruntime")
    
    ctx_id = -1  # Toujours CPU sur Render
    
    print("🔄 Chargement FaceAnalysis...")
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    print("🔄 Préparation inswapper...")
    model_path = ensure_model_downloaded()
    swapper = get_model(model_path, providers=['CPUExecutionProvider'])
    
    # Mettre en cache
    _models_cache = (app, swapper)
    
    print("✅ Modèles chargés et mis en cache")
    return app, swapper


def get_aligned_face(app, image):
    """Détecte le visage principal."""
    faces = app.get(image)
    if len(faces) == 0:
        raise Exception("Aucun visage détecté")
    return sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)[0]


def swap_faces(app, swapper, target_img, source_img):
    """Effectue le face swap."""
    target_face = get_aligned_face(app, target_img)
    source_face = get_aligned_face(app, source_img)
    result = swapper.get(target_img, target_face, source_face, paste_back=True)
    return result


def face_swap_from_paths(target_path: str, source_path: str, output_path: str, use_gpu: bool = False):
    """Face swap complet depuis les chemins."""
    print("🔄 Chargement des modèles InsightFace...")
    app, swapper = load_models(use_gpu=use_gpu)
    
    print(f"🔄 Chargement des images...")
    target_img = cv2.imread(str(target_path))
    source_img = cv2.imread(str(source_path))
    
    if target_img is None or source_img is None:
        raise FileNotFoundError("Erreur chargement images")
    
    print("🔄 Face swap en cours...")
    result = swap_faces(app, swapper, target_img, source_img)
    
    cv2.imwrite(str(output_path), result)
    print(f"✅ Face swap terminé : {output_path}")
    
    return output_path