"""
Pneumonia Detection API - FastAPI backend
Uses a ViT model fine-tuned on chest X-rays from huggingface hub.
Model: nickmuchi/vit-finetuned-chest-xray-pneumonia
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import time
import logging

from transformers import pipeline


# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Pneumonia Detection API",
    description="Deep learning-based chest X-ray analysis for pneumonia detection",
    version="2.0.0"
)

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL CONFIG ----------------
MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
classifier = None


# ---------------- LOAD MODEL ----------------
@app.on_event("startup")
async def load_model():
    global classifier
    logger.info(f"Loading model: {MODEL_ID}")

    try:
        classifier = pipeline(
            "image-classification",
            model=MODEL_ID
        )
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")


# ---------------- SCHEMAS ----------------
class PredictionResult(BaseModel):
    label: str
    confidence: float
    all_scores: list[dict]
    inference_time_ms: float
    model_id: str
    verdict: str
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: str


# ---------------- HELPER FUNCTION ----------------
def map_risk(label: str, confidence: float):

    is_pneumonia = "PNEUMONIA" in label.upper()

    if is_pneumonia:

        risk = "HIGH" if confidence >= 0.85 else "MEDIUM"

        verdict = (
            "Pneumonia detected - Please consult a physician immediately"
            if confidence >= 0.85
            else "Possible Pneumonia - Further evaluation is needed"
        )

    else:

        risk = "LOW"
        verdict = "Normal - No signs of pneumonia detected"

    return verdict, risk


# ---------------- HEALTH CHECK ----------------
@app.get("/health", response_model=HealthResponse)
async def health_check():

    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        model_id=MODEL_ID
    )


# ---------------- PREDICTION API ----------------
@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again shortly"
        )

    # Validate image type
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG/PNG images are supported"
        )

    # Read image
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not process uploaded image"
        )

    # Run AI model
    t0 = time.perf_counter()

    try:
        results = classifier(image, top_k=None)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {e}"
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Sort results
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    top = results[0]

    verdict, risk_level = map_risk(
        top["label"],
        top["score"]
    )

    return PredictionResult(
        label=top["label"],
        confidence=top["score"],
        all_scores=[
            {"label": r["label"], "score": r["score"]}
            for r in results
        ],
        inference_time_ms=round(elapsed_ms, 2),
        model_id=MODEL_ID,
        verdict=verdict,
        risk_level=risk_level
    )


# ---------------- MODEL INFO ----------------
@app.get("/model-info")
async def model_info():

    return {
        "model_id": MODEL_ID,
        "architecture": "Vision Transformer (ViT)",
        "task": "Image Classification - Chest X-Ray",
        "classes": [
            "NORMAL",
            "PNEUMONIA"
        ],
        "source": "Hugging Face"
    }