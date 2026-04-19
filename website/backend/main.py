from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db
from models import scan_model
from queries import insert_model

app = FastAPI(title="FaceSwap Vanilla API")

# ════════════════════════════════════════════════════════════
# CORS : Autoriser toutes les origines (frontend Render)
# ════════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Permet toutes les origines
    allow_credentials=True,        # Permet les cookies (ajouté)
    allow_methods=["*"],           # Permet toutes les méthodes
    allow_headers=["*"],           # Permet tous les headers
)

app.include_router(router)

# Création des tables à l'importation
init_db()


# ════════════════════════════════════════════════════════════
# Événement au démarrage : Charger les modèles ML
# ════════════════════════════════════════════════════════════
@app.on_event("startup")
def load_models():
    """Charge les modèles ML au démarrage du serveur"""
    print("🚀 Démarrage du serveur...")
    print("📊 Chargement des modèles ML...")
    
    # Scan des modèles disponibles
    models = scan_model()
    
    # Insertion en BDD
    insert_model("faceswap.db", models)
    
    print("✅ Modèles chargés avec succès")


# ════════════════════════════════════════════════════════════
# Health check
# ════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return {"status": "ok", "message": "FaceSwap API is running"}


# ════════════════════════════════════════════════════════════
# Endpoint de ping (pour garder le serveur éveillé)
# ════════════════════════════════════════════════════════════
@app.get("/ping")
def ping():
    """
    Endpoint pour garder le serveur éveillé sur Render.
    Le plan gratuit arrête le serveur après 15 min d'inactivité.
    """
    from datetime import datetime
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "message": "Backend is awake"
    }


# ════════════════════════════════════════════════════════════
# Route racine
# ════════════════════════════════════════════════════════════
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