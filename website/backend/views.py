from fastapi import APIRouter,FastAPI
from pydantic import BaseModel
from models import create_user,connection,scan_model,get_models
from queries import insert_model
from pathlib import Path


router = APIRouter()
app = FastAPI()



# ── Schémas (ce que le frontend envoie) ──

class RegisterBody(BaseModel):
    email: str
    password: str
    name: str | None = None

class LoginBody(BaseModel):
    email: str
    password: str
# ── Routes users ──

@router.post("/users/register")
def register(body: RegisterBody):
    user = create_user(body.email, body.password, body.name)
    return {"user": user}

@router.post("/users/login")
def login(body: LoginBody):
    user = connection(body.email, body.password)
    return {"user": user}

@app.on_event("startup")
def load_models():
    print ("etré dfans la route")
    models = scan_model() 
    BASE_DIR = Path(__file__).resolve().parents[2]
    models_directory = BASE_DIR / "app/models"
    
@router.get("/swap/models")
def list_models():
    return get_models()
