from queries import (
    sql_insert_user, sql_get_user_by_email, sql_get_user_by_id,
)
from fastapi import HTTPException #Permet de spécifier les erreurs avec raise


# ── USERS ──

def create_user(email: str, password: str, name: str):
    
    if sql_get_user_by_email(email):
        raise HTTPException(status_code=400, detail="Email déjà utilisé")
    
    user_id = sql_insert_user(email, password, name)
    return {"id": user_id, "email": email, "name": name}
