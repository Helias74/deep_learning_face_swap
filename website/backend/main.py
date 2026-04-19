from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db
from models import scan_model
from queries import insert_model

app = FastAPI(title="FaceSwap API")

# ════════════════════════════════════════════════════════════
# CORS - DOIT ÊTRE AVANT include_router() !
# ════════════════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════════════════
app.include_router(router)

# ════════════════════════════════════════════════════════════
# Initialisation
# ════════════════════════════════════════════════════════════
init_db()


@app.on_event("startup")
def startup_event():
    """Scan et insertion des modèles au démarrage"""
    print("🚀 Démarrage du serveur...")
    print("📊 Scan des modèles...")
    
    models = scan_model()
    insert_model("faceswap.db", models)
    
    print(f"✅ {len(models)} modèles chargés")


@app.get("/")
def read_root():
    return {"message": "FaceSwap API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}