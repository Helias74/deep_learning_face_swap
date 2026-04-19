from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid

from models import create_user, connection, get_models, crop_two_images, perform_face_swap

router = APIRouter()

# ════════════════════════════════════════════════════════════
# Configuration des chemins
# ════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Uploads directory: {UPLOAD_DIR}")


# ════════════════════════════════════════════════════════════
# Schémas Pydantic
# ════════════════════════════════════════════════════════════
class RegisterBody(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginBody(BaseModel):
    email: str
    password: str


# ════════════════════════════════════════════════════════════
# Routes utilisateurs
# ════════════════════════════════════════════════════════════
@router.post("/users/register")
def register(body: RegisterBody):
    """Créer un nouveau compte utilisateur"""
    user = create_user(body.email, body.password, body.name)
    return {"user": user}


@router.post("/users/login")
def login(body: LoginBody):
    """Connexion utilisateur"""
    user = connection(body.email, body.password)
    return {"user": user}


# ════════════════════════════════════════════════════════════
# Routes face swap
# ════════════════════════════════════════════════════════════
@router.get("/swap/models")
def list_models():
    """Liste les modèles disponibles"""
    return get_models()


@router.post("/swap/crop")
async def crop_images(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    """Crop les visages de deux images et retourne la source croppée"""
    
    # Créer un dossier de session unique
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Sauvegarder les images
        photo_a_path = session_dir / "photo_a.jpg"
        photo_b_path = session_dir / "photo_b.jpg"
        
        with open(photo_a_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        
        with open(photo_b_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        
        # Crop les visages
        source_crop_path, target_crop_path = crop_two_images(
            str(photo_a_path), 
            str(photo_b_path)
        )
        
        # Retourner l'image source croppée
        return FileResponse(
            path=source_crop_path,
            media_type="image/jpeg",
            filename="photo_a_crop.jpg"
        )
    
    except Exception as e:
        print(f"❌ Erreur crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/swap/process")
async def process_face_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    """Effectue le face swap complet : crop + swap"""
    
    # Créer un dossier de session unique
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Sauvegarder les images
        photo_a_path = session_dir / "photo_a.jpg"
        photo_b_path = session_dir / "photo_b.jpg"
        
        with open(photo_a_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        
        with open(photo_b_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        
        print(f"✅ Images sauvegardées")
        
        # Crop les visages
        print(f"🔄 Crop des visages...")
        source_crop_path, target_crop_path = crop_two_images(
            str(photo_a_path), 
            str(photo_b_path)
        )
        print(f"✅ Visages croppés")
        
        # Face swap
        print(f"🎭 Face swap en cours...")
        result_path = perform_face_swap(
            source_crop_path=source_crop_path,
            target_crop_path=target_crop_path
        )
        print(f"✅ Face swap terminé")
        
        # Retourner le résultat
        return FileResponse(
            path=result_path,
            media_type="image/jpeg",
            filename="faceswap_result.jpg"
        )
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))