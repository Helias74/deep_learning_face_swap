def load_models(use_gpu=False):
    """Charge les modèles InsightFace avec config optimisée."""
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
    
    ctx_id = -1
    
    # Utiliser un modèle plus léger (buffalo_sc au lieu de buffalo_l)
    print("🔄 Chargement FaceAnalysis (version légère)...")
    app = FaceAnalysis(name='buffalo_sc')  # ← buffalo_sc = plus léger
    app.prepare(ctx_id=ctx_id, det_size=(320, 320))  # ← Réduire det_size
    
    print("🔄 Préparation inswapper...")
    model_path = ensure_model_downloaded()
    swapper = get_model(model_path, providers=['CPUExecutionProvider'])
    
    _models_cache = (app, swapper)
    
    print("✅ Modèles chargés et mis en cache")
    return app, swapper