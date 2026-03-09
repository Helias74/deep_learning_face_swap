import os
from pathlib import Path


from queries import (
    sql_insert_user, sql_get_user_by_email, sql_get_user_by_id,sql_get_password_by_email,
)
from fastapi import HTTPException #Permet de spécifier les erreurs avec raise


# ── USERS ──

def create_user(email: str, password: str, name: str):
    
    if sql_get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    
    user_id = sql_insert_user(email, password, name)
    return {"id": user_id, "email": email, "name": name}

def connection (email:str, password:str):
    user = sql_get_user_by_email(email)
    if sql_get_user_by_email(email) and user["password"]==password:
        return {"id": user["id"], "email": user["email"], "name": user["name"]}
    else :
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
        
# ── MODELS ──

def scan_model():
    BASE_DIR = Path(__file__).resolve().parents[2]
    models_directory = BASE_DIR / "app/models"
    
    if not models_directory.exists():
        return {"error": "Dossier models introuvable"}

    models_data = []

    for file in sorted(models_directory.glob("*.pth")):
        models_data.append({
            "name": file.name,
            "file_path": str(file)
        })

    return models_data

    

    