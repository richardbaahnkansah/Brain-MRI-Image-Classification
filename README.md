# Zero-Shot Image Classification Demo with CLIP

An interactive web application that performs **zero-shot image classification** using Hugging Face CLIP-based models. Upload an image, define custom target categories, and compare predictions across multiple CLIP model variants side by side.

## Architecture

```
clip-classification-demo/
├── index.html       # Frontend: vanilla JS + Chart.js, served from S3
├── api_server.py    # Backend: FastAPI server running CLIP inference
├── app.py           # Alternative: standalone Streamlit version
├── requirements.txt # Python dependencies
└── README.md
```

The app uses a **decoupled architecture**:
- **Frontend** (`index.html`): Vanilla JavaScript with Chart.js for visualization, served as static files
- **Backend** (`api_server.py`): FastAPI server that loads CLIP models and runs inference, connected via port-forwarding proxy

## Features

- **Image Upload** — Drag-and-drop or click to upload (PNG, JPEG, WebP, BMP)
- **Custom Labels** — Define your own candidate categories (comma-separated)
- **Multi-Model Comparison** — Select up to 3 CLIP models for side-by-side evaluation:
  - `openai/clip-vit-base-patch32` (fast, 151M params)
  - `openai/clip-vit-base-patch16` (balanced, 150M params)
  - `openai/clip-vit-large-patch14` (higher accuracy, 304M params)
- **Probability Bar Charts** — Horizontal Chart.js bar charts showing the full probability distribution per label
- **Top-3 Result Tables** — Structured tables displaying rank, label, and probability percentage
- **Cross-Model Summary** — Comparison table showing top prediction and confidence for each model, plus agreement/disagreement indicator
- **Example Label Sets** — Quick-fill presets for common use cases (animals, medical imaging, everyday objects, food)
- **Model Caching** — Models loaded once and cached via `@lru_cache` for fast re-runs

## Quick Start

### Prerequisites

- Python 3.10+
- ~4 GB free disk space for model weights

### Installation

```bash
cd clip-classification-demo
pip install -r requirements.txt
```

### Run the Backend

```bash
python api_server.py
# or: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Run the Frontend

Open `index.html` in your browser, or serve it with any static file server:

```bash
python -m http.server 3000
# Open http://localhost:3000
```

The frontend automatically detects whether it's running locally (`http://localhost:8000`) or through the deployment proxy.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/models` | List available CLIP models |
| `POST` | `/api/classify/{model_id}` | Classify an uploaded image against custom labels |

### Example API Call

```bash
curl -X POST "http://localhost:8000/api/classify/openai--clip-vit-base-patch32" \
  -F "file=@photo.jpg" \
  -F "labels=a photo of a cat, a photo of a dog, a photo of a bird"
```

**Note:** Model IDs use `--` instead of `/` in the URL path (e.g., `openai--clip-vit-base-patch32`).

## How It Works

1. **Image & Labels** — You upload an image and provide candidate text labels (e.g., "a photo of a cat, a photo of a dog").
2. **CLIP Encoding** — Each selected CLIP model encodes the image and all text labels into a shared embedding space.
3. **Similarity & Softmax** — Cosine similarity between image and text embeddings is computed, then softmax produces a probability distribution over labels.
4. **Visualization** — Results are displayed as horizontal bar charts and structured tables for each model, with a cross-model summary when multiple models are selected.

## Example Use Cases

- **Medical Imaging** — Classify modality (chest X-ray, brain MRI, skin lesion, retinal scan)
- **Animal Recognition** — Categorize animal photos
- **Object Detection** — Identify everyday objects in scenes
- **Food Classification** — Distinguish food types

## Alternative: Streamlit Version

A standalone Streamlit version is included in `app.py`. To run it:

```bash
streamlit run app.py
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| [FastAPI](https://fastapi.tiangolo.com) | Backend API server |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | CLIP model loading & inference |
| [PyTorch](https://pytorch.org) | Deep learning backend |
| [Chart.js](https://www.chartjs.org) | Interactive probability bar charts |
| [Pillow](https://python-pillow.org) | Image processing |

## Notes

- Models download from Hugging Face on first use (~600 MB–1.2 GB each).
- Inference runs on CPU by default. If CUDA is available, PyTorch will use it automatically.
- Adding context to labels (e.g., "a photo of a cat" instead of "cat") typically improves CLIP accuracy.
