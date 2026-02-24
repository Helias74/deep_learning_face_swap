# FaceSwap Vanilla

## Stack
- Frontend : HTML + CSS + JS pur
- Backend : Python + FastAPI
- BDD : SQLite (fichier local)

## Lancement

```bash
# Backend
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
# Ouvrir frontend/pages/index.html dans le navigateur
# OU lancer un serveur simple :
cd frontend
python3 -m http.server 3000
```

## Architecture
```
backend/
├── queries.py      # tout le SQL
├── models.py       # logique autour des requêtes SQL
├── views.py        # routes FastAPI
└── main.py         # point d'entrée

frontend/
├── pages/          # fichiers HTML
├── styles/         # fichiers CSS
└── scripts/        # fichiers JS
```
