from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid

from models import create_user, connection, get_models, crop_two_images, perform_face_swap
from queries import sql_get_models_by_type

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RegisterBody(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginBody(BaseModel):
    email: str
    password: str


@router.post("/users/register")
def register(body: RegisterBody):
    user = create_user(body.email, body.password, body.name)
    return {"user": user}


@router.post("/users/login")
def login(body: LoginBody):
    user = connection(body.email, body.password)
    return {"user": user}


@router.get("/swap/models")
def list_all_models():
    """Liste tous les modèles."""
    return get_models()


@router.get("/swap/models/crop")
def list_crop_models():
    """Liste uniquement les modèles de crop."""
    try:
        models = sql_get_models_by_type("crop")
        return models
    except Exception as e:
        print(f"❌ Erreur liste crop models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/swap/models/face_swap")
def list_face_swap_models():
    """Liste uniquement les modèles de face swap."""
    try:
        models = sql_get_models_by_type("face_swap")
        return models
    except Exception as e:
        print(f"❌ Erreur liste swap models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/swap/process")
async def process_face_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    crop_model: str = Form("yolov8n"),
    swap_model: str = Form("buffalo_l")
):
    """Face swap complet avec choix des modèles."""
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        photo_a_path = session_dir / "photo_a.jpg"
        photo_b_path = session_dir / "photo_b.jpg"
        
        with open(photo_a_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        
        with open(photo_b_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        
        print(f"✅ Images sauvegardées")
        print(f"🔧 Modèle crop: {crop_model}")
        print(f"🔧 Modèle swap: {swap_model}")
        
        # Crop
        source_crop_path, target_crop_path = crop_two_images(
            str(photo_a_path), 
            str(photo_b_path),
            crop_model=crop_model
        )
        print(f"✅ Visages croppés")
        
        # Face swap
        result_path = perform_face_swap(
            source_crop_path=source_crop_path,
            target_crop_path=target_crop_path,
            swap_model=swap_model
        )
        print(f"✅ Face swap terminé")
        
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