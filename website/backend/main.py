from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db
from models import scan_model
from queries import insert_model

app = FastAPI(title="FaceSwap Vanilla API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Création des tables
init_db()


@app.on_event("startup")
async def startup_event():
    """Démarrage du serveur : pré-télécharger les modèles"""
    print("🚀 Démarrage du serveur...")
    
    # 1. Scan des modèles de crop
    print("📊 Scan des modèles de crop...")
    models = scan_model()
    insert_model("faceswap.db", models)
    print("✅ Modèles de crop chargés")
    
    # 2. Pré-télécharger les modèles InsightFace
    print("📥 Pré-téléchargement des modèles InsightFace...")
    try:
        # Importer et charger les modèles au démarrage
        import sys
        from pathlib import Path
        
        # Ajouter le chemin
        BASE_DIR = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(BASE_DIR / "app" / "face_swap"))
        
        from swapper import load_models
        
        # Charger les modèles (téléchargement si nécessaire)
        app, swapper = load_models(use_gpu=False)
        
        print("✅ Modèles InsightFace prêts en cache")
    except Exception as e:
        print(f"⚠️ Erreur pré-téléchargement InsightFace : {e}")
        print("→ Les modèles seront téléchargés au premier appel")
    
    print("✅ Serveur prêt")


@app.get("/health")
def health():
    """Endpoint de santé"""
    return {"status": "ok", "message": "FaceSwap API is running"}


@app.get("/ping")
def ping():
    """Endpoint pour garder le serveur éveillé"""
    from datetime import datetime
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "Backend is awake"
    }


@app.get("/")
def read_root():
    """Page d'accueil de l'API"""
    return {
        "message": "FaceSwap Vanilla API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "/health",
            "ping": "/ping",
            "models": "/swap/models",
            "crop": "/swap/crop",
            "face_swap": "/swap/process"
        }
    }