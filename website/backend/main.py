from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db

app = FastAPI(title="FaceSwap API")

# ════════════════════════════════════════════════════════════
# CORS
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
# Initialisation base de données
# ════════════════════════════════════════════════════════════
init_db()


@app.get("/")
def read_root():
    """Page d'accueil"""
    return {"message": "FaceSwap API", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok"}