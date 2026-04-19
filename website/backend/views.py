from fastapi import APIRouter, FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import uuid

# Importer depuis models.py
from models import create_user, connection, scan_model, get_models, crop_two_images,perform_face_swap 
from queries import insert_model

router = APIRouter()
app = FastAPI()



BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True) 


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




@router.post("/swap/crop")
async def crop_images(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):  
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Sauvegarder les images uploadées
        photo_a_path = session_dir / "photo_a.jpg"
        photo_b_path = session_dir / "photo_b.jpg"
        
        with open(photo_a_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        
        with open(photo_b_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        
        print(f"✅ Images sauvegardées dans {session_dir}")
        
        # 2. Appeler la fonction de models.py pour cropper
        source_crop_path, target_crop_path = crop_two_images(
            str(photo_a_path), 
            str(photo_b_path)
        )
        
        print(f"✅ Crops terminés : {source_crop_path}, {target_crop_path}")
        
        # 3. Retourner l'image source croppée
        return FileResponse(
            path=source_crop_path,
            media_type="image/jpeg",
            filename="photo_a_crop.jpg"
        )
    
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé : {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/swap/process")
async def process_face_swap(
    source: UploadFile = File(...),
    target: UploadFile = File(...)
):
    
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Sauvegarder les images uploadées
        photo_a_path = session_dir / "photo_a.jpg"
        photo_b_path = session_dir / "photo_b.jpg"
        
        with open(photo_a_path, "wb") as f:
            shutil.copyfileobj(source.file, f)
        
        with open(photo_b_path, "wb") as f:
            shutil.copyfileobj(target.file, f)
        
        print(f"✅ Images sauvegardées dans {session_dir}")
        
        # 2. Crop les visages
        print(f"🔄 Crop des visages...")
        source_crop_path, target_crop_path = crop_two_images(
            str(photo_a_path), 
            str(photo_b_path)
        )
        
        print(f"✅ Visages croppés")
        
        # 3. Face swap
        print(f"🎭 Face swap en cours...")
        result_path = perform_face_swap(
            source_crop_path=source_crop_path,
            target_crop_path=target_crop_path
        )
        
        print(f"✅ Face swap terminé : {result_path}")
        
        # 4. Retourner le résultat
        return FileResponse(
            path=result_path,
            media_type="image/jpeg",
            filename="faceswap_result.jpg"
        )
    
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé : {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))