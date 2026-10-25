#!/usr/bin/env python3
"""
FastAPI backend for zero-shot image classification with CLIP models.
Serves model inference endpoints; the frontend communicates via REST API.
"""

import io
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="Zero-Shot CLIP Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_MODELS = [
    {"id": "openai/clip-vit-base-patch32",  "name": "CLIP ViT-B/32",  "description": "Faster, ~151M params"},
    {"id": "openai/clip-vit-base-patch16",  "name": "CLIP ViT-B/16",  "description": "Balanced, ~150M params"},
    {"id": "openai/clip-vit-large-patch14", "name": "CLIP ViT-L/14",  "description": "Higher accuracy, ~304M params"},
]


@lru_cache(maxsize=3)
def load_model(model_id: str):
    """Load and cache a CLIP model + processor."""
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return model, processor


class ClassifyRequest(BaseModel):
    labels: list[str]


@app.get("/api/models")
def get_models():
    """Return list of available CLIP models."""
    return AVAILABLE_MODELS


@app.post("/api/classify/{model_id}")
async def classify_image(model_id: str, file: UploadFile = File(...), labels: str = Form(...)):
    """
    Run zero-shot classification on an uploaded image.

    - model_id: Hugging Face model ID (path-encoded with slashes replaced by --)
    - file: uploaded image file
    - labels: comma-separated candidate labels
    """
    # Decode model_id (slashes replaced with -- for URL safety)
    real_model_id = model_id.replace("--", "/")
    if not any(m["id"] == real_model_id for m in AVAILABLE_MODELS):
        return {"error": f"Unknown model: {real_model_id}"}

    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if len(label_list) < 2:
        return {"error": "At least 2 labels are required"}

    # Read and process image
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image: {str(e)}"}

    # Load model
    try:
        model, processor = load_model(real_model_id)
    except Exception as e:
        return {"error": f"Failed to load model {real_model_id}: {str(e)}"}

    # Run inference
    inputs = processor(
        text=label_list,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()

    # Build sorted results
    results = []
    order = np.argsort(probs)[::-1]
    for rank, idx in enumerate(order, start=1):
        results.append({
            "rank": rank,
            "label": label_list[idx],
            "probability": float(probs[idx]),
            "probability_pct": f"{probs[idx]*100:.2f}%",
        })

    return {
        "model_id": real_model_id,
        "model_name": next(m["name"] for m in AVAILABLE_MODELS if m["id"] == real_model_id),
        "image_size": {"width": image.width, "height": image.height},
        "labels": label_list,
        "probabilities": [float(p) for p in probs],
        "results": results,
        "top_prediction": label_list[int(np.argmax(probs))],
        "top_confidence": f"{float(np.max(probs))*100:.2f}%",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
