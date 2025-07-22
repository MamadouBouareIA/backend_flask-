# coding utf-8
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialisation FastAPI
app = FastAPI(
    title="API Prédiction Véhicule",
    description="API pour la classification d'images de véhicules avec MobileNetV2",
    version="1.0.0"
)

# Autoriser les requêtes CORS (Flutter, web, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle et les labels au démarrage
MODEL_PATH = "model_voiture_professionnel.keras"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des labels : {e}")

@app.post("/predict", summary="Prédire la classe d'une image")
async def predict(file: UploadFile = File(...)):
    # Vérification du type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = model.predict(img_array)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        label = class_names[idx]

        return JSONResponse({
            "label": label,
            "confidence": confidence,
            "class_index": idx
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

@app.get("/", summary="Test API")
def root():
    return {"message": "API de prédiction de véhicules opérationnelle."}

if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000, reload=True)