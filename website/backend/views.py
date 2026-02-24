from fastapi import APIRouter
from pydantic import BaseModel
from models import create_user

router = APIRouter()


# ── Schémas (ce que le frontend envoie) ──

class RegisterBody(BaseModel):
    email: str
    password: str
    name: str | None = None


# ── Routes users ──

@router.post("/users/register")
def register(body: RegisterBody):
    user = create_user(body.email, body.password, body.name)
    return {"user": user}
