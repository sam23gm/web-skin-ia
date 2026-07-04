import os
import time

import joblib
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModel

# ==================================================
# RUTAS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "modelos")

# Repositorio del modelo en Hugging Face
HF_MODEL_ID = "sam23gm/web-skin-model"

XGB_MODEL_PATH = os.path.join(
    MODEL_PATH,
    "modelo_xgb_ft_Balanced_v2.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_PATH,
    "config_mbert_balanced_v2.pkl"
)

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Web Skin IA",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# VARIABLES GLOBALES
# ==================================================

tokenizer = None
bert_model = None
xgb_model = None
config = None

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# REQUEST
# ==================================================

class ReviewRequest(BaseModel):
    text: str

# ==================================================
# CARGAR MODELOS
# ==================================================

@app.on_event("startup")
def load_models():

    global tokenizer
    global bert_model
    global xgb_model
    global config

    print("=" * 60)
    print("Cargando modelos...")
    print("=" * 60)

    config = joblib.load(CONFIG_PATH)

    print("Descargando tokenizer desde Hugging Face...")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID
    )

    print("✅ Tokenizer cargado")

    print("Descargando modelo mBERT desde Hugging Face...")

    bert_model = AutoModel.from_pretrained(
        HF_MODEL_ID
    )

    bert_model.to(device)
    bert_model.eval()

    print("✅ Modelo mBERT cargado")

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    print("✅ Modelo XGBoost cargado")

    print("=" * 60)
    print("Backend listo")
    print("=" * 60)

# ==================================================
# EMBEDDINGS
# ==================================================

def get_embedding(text):

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt"
    )

    encoded = {
        k: v.to(device)
        for k, v in encoded.items()
    }

    with torch.no_grad():

        outputs = bert_model(**encoded)

        embedding = (
            outputs
            .last_hidden_state
            .mean(dim=1)
            .cpu()
            .numpy()
        )

    return embedding

# ==================================================
# FRONTEND
# ==================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            TEMPLATES_DIR,
            "index.html"
        )
    )

# ==================================================
# TEST
# ==================================================

@app.get("/test")
def test():

    emb = get_embedding(
        "This moisturizer is amazing."
    )

    return {
        "success": True,
        "shape": list(emb.shape)
    }

# ==================================================
# PREDICCIÓN
# ==================================================

@app.post("/predict")
def predict(data: ReviewRequest):

    texto = data.text.strip()

    if texto == "":
        return {
            "success": False,
            "message": "Ingrese una reseña."
        }

    inicio = time.perf_counter()

    embedding = get_embedding(texto)

    prediction = int(
        xgb_model.predict(embedding)[0]
    )

    probabilities = xgb_model.predict_proba(
        embedding
    )[0]

    probability = float(
        probabilities[prediction]
    )

    label = config["labels"][prediction]

    fin = time.perf_counter()

    tiempo_ms = round(
        (fin - inicio) * 1000,
        2
    )

    return {
        "success": True,
        "prediction": prediction,
        "label": label,
        "probability": round(
            probability * 100,
            2
        ),
        "inference_time_ms": tiempo_ms
    }

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )