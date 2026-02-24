from fastapi import APIRouter
from pydantic import BaseModel
from models import create_user,connection

router = APIRouter()


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

