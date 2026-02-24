from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db

app = FastAPI(title="FaceSwap Vanilla API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Crée les tables au démarrage
init_db()

@app.get("/health")
def health():
    return {"status": "ok"}
