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
def load_models():
    """Charge les modèles ML au démarrage du serveur"""
    print("🚀 Démarrage du serveur...")
    print("📊 Scan des modèles de crop...")
    
    # Scan des modèles disponibles
    models = scan_model()
    
    # Insertion en BDD
    insert_model("faceswap.db", models)
    
    print("✅ Serveur prêt")
    # Note: Les modèles InsightFace seront chargés au premier appel /swap/process


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