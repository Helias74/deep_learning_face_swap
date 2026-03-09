from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from views import router
from queries import init_db
from models import scan_model
from queries import insert_model

app = FastAPI(title="FaceSwap Vanilla API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# création tables
init_db()


@app.on_event("startup")
def load_models():
    init_db() 

    print("startup models")

    models = scan_model()
    insert_model("faceswap.db", models)


@app.get("/health")
def health():
    return {"status": "ok"}