"""
Zero-Shot Image Classification Demo with CLIP Models
-----------------------------------------------------
Interactive Streamlit web app for zero-shot image classification using
Hugging Face CLIP-based models. Supports side-by-side comparison of
multiple models with probability bar charts and structured result tables.
"""

import io
import re
from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = OrderedDict([
    ("openai/clip-vit-base-patch32",  "CLIP ViT-B/32  (faster, ~151M params)"),
    ("openai/clip-vit-base-patch16",  "CLIP ViT-B/16  (balanced, ~150M params)"),
    ("openai/clip-vit-large-patch14", "CLIP ViT-L/14  (higher accuracy, ~304M params)"),
])

MAX_MODELS = 3  # safety cap to limit memory / load time

EXAMPLE_LABEL_SETS = {
    "Animal categories": "a photo of a cat, a photo of a dog, a photo of a bird, a photo of a fish",
    "Medical imaging": "chest X-ray, brain MRI, skin lesion, retinal scan, abdominal CT",
    "Everyday objects": "a cup, a laptop, a car, a tree, a building, a person",
    "Food types": "pizza, sushi, burger, salad, pasta, steak",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Zero-Shot CLIP Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    """Load a CLIP model and processor, cached per model_id."""
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return model, processor


@st.cache_data(show_spinner=False)
def run_classification(model_id: str, image_bytes: bytes, labels: tuple[str, ...]):
    """Run zero-shot classification and return sorted probabilities.

    Cached on (model_id, image_bytes, labels) so re-running with the same
    inputs across sidebar changes doesn't re-invoke the model.
    """
    model, processor = load_model(model_id)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Build text inputs — prepend "a photo of " to bare labels for better
    # CLIP performance if the user hasn't already added context.
    text_inputs = []
    for label in labels:
        text_inputs.append(label)

    inputs = processor(
        text=text_inputs,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image shape: (1, num_labels)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0).cpu().numpy()
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_labels(raw: str) -> list[str]:
    """Parse comma- or newline-separated labels, strip whitespace, dedupe."""
    parts = re.split(r"[,\n]", raw)
    seen: set[str] = set()
    labels: list[str] = []
    for p in parts:
        label = p.strip()
        if label and label.lower() not in seen:
            seen.add(label.lower())
            labels.append(label)
    return labels


def make_bar_chart(labels: list[str], probs: np.ndarray, model_display: str) -> go.Figure:
    """Horizontal bar chart of probability distribution."""
    # Sort by probability descending for readability
    order = np.argsort(probs)  # ascending → reversed for top-to-bottom
    sorted_labels = [labels[i] for i in order]
    sorted_probs = probs[order]

    colors = []
    max_idx = len(sorted_probs) - 1  # highest prob after sort
    for i in range(len(sorted_probs)):
        if i == max_idx:
            colors.append("#2563eb")  # blue for top
        else:
            colors.append("#93c5fd")  # light blue

    fig = go.Figure(
        go.Bar(
            x=sorted_probs * 100,
            y=sorted_labels,
            orientation="h",
            text=[f"{p*100:.2f}%" for p in sorted_probs],
            textposition="outside",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"Probability Distribution — {model_display}", font=dict(size=14)),
        xaxis=dict(title="Probability (%)", range=[0, 105], tickformat=".0f"),
        yaxis=dict(autorange="reversed"),
        height=max(350, len(labels) * 35),
        margin=dict(l=10, r=60, t=50, b=40),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_top3_table(labels: list[str], probs: np.ndarray) -> list[dict]:
    """Return list of dicts for the top-3 predictions."""
    order = np.argsort(probs)[::-1]  # descending
    top3 = []
    for rank, idx in enumerate(order[:3], start=1):
        top3.append({
            "Rank": rank,
            "Label": labels[idx],
            "Probability": f"{probs[idx]*100:.2f}%",
            "Raw Score": f"{probs[idx]:.6f}",
        })
    return top3


def build_summary_table(selected_models: list[str], all_results: dict) -> list[dict]:
    """Cross-model summary: top prediction and confidence per model."""
    rows = []
    for model_id in selected_models:
        if model_id not in all_results:
            continue
        labels, probs = all_results[model_id]
        best_idx = int(np.argmax(probs))
        display = AVAILABLE_MODELS.get(model_id, model_id)
        rows.append({
            "Model": display,
            "Top Prediction": labels[best_idx],
            "Confidence": f"{probs[best_idx]*100:.2f}%",
            "Labels Evaluated": len(labels),
        })
    return rows


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### Model Selection")
    st.caption("Choose 1–3 CLIP models for side-by-side comparison.")

    selected_models = st.multiselect(
        "Models",
        options=list(AVAILABLE_MODELS.keys()),
        default=["openai/clip-vit-base-patch32"],
        format_func=lambda x: AVAILABLE_MODELS[x],
        max_selections=MAX_MODELS,
    )

    st.markdown("---")
    st.markdown("### Custom Labels")
    st.caption("Enter candidate categories, separated by commas or new lines.")

    example_choice = st.selectbox(
        "Quick-fill example labels",
        options=["— Custom —"] + list(EXAMPLE_LABEL_SETS.keys()),
        index=0,
    )

    default_text = EXAMPLE_LABEL_SETS.get(example_choice, "")
    if example_choice == "— Custom —":
        default_text = "a photo of a cat, a photo of a dog, a photo of a bird"

    labels_input = st.text_area(
        "Target categories",
        value=default_text,
        height=120,
        help="Each label is sent to CLIP as a text prompt. Adding context "
             "(e.g. 'a photo of a cat') typically improves accuracy.",
    )

    st.markdown("---")

    # Info box
    with st.expander("ℹ️ How it works"):
        st.write(
            "This app uses **CLIP** (Contrastive Language–Image Pre-training) "
            "models from Hugging Face to perform **zero-shot** image "
            "classification — no training required. You provide an image and "
            "a set of candidate text labels; CLIP computes the similarity "
            "between the image and each label, then outputs a probability "
            "distribution via softmax."
        )
        st.write(
            "**Models** are loaded once and cached. Larger models (ViT-L/14) "
            "may take longer to download on first use."
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🖼️ Zero-Shot Image Classification with CLIP")
st.markdown(
    "Upload an image, define your own labels, and compare CLIP model "
    "predictions side by side. Probabilities are displayed as horizontal bar "
    "charts with structured top-3 result tables."
)
st.markdown("---")

# --- Image upload ---
col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Supported formats: PNG, JPEG, WebP, BMP",
    )

with col_info:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.metric("Image size", f"{image.width} × {image.height} px")
        st.metric("Format", image.format or "Unknown")

# --- Validation ---
if not selected_models:
    st.warning("Please select at least one model in the sidebar.")
    st.stop()

labels = parse_labels(labels_input)

if len(labels) < 2:
    st.warning("Please provide at least 2 unique labels for classification.")
    st.stop()

if uploaded_file is None:
    st.info("👆 Upload an image to begin classification.")
    st.stop()

# --- Display uploaded image ---
st.markdown("### Uploaded Image")
img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
with img_col2:
    st.image(image, caption="Input image", use_container_width=True)

st.markdown("---")

# --- Run classification ---
image_bytes = uploaded_file.getvalue()

all_results: dict[str, tuple[list[str], np.ndarray]] = {}

progress_bar = st.progress(0, text="Loading models and running inference…")
status_text = st.empty()

for i, model_id in enumerate(selected_models):
    display = AVAILABLE_MODELS.get(model_id, model_id)
    status_text.text(f"Running {display}…")
    probs = run_classification(model_id, image_bytes, tuple(labels))
    all_results[model_id] = (labels, probs)
    progress_bar.progress((i + 1) / len(selected_models), text=f"Completed {display}")

progress_bar.empty()
status_text.empty()

# --- Side-by-side results ---
num_models = len(selected_models)
st.markdown(f"### Results — {num_models} Model{'s' if num_models > 1 else ''} Compared")

cols = st.columns(num_models)

for col, model_id in zip(cols, selected_models):
    labels_list, probs = all_results[model_id]
    display = AVAILABLE_MODELS.get(model_id, model_id)

    with col:
        st.markdown(f"#### {display}")

        # Probability bar chart
        fig = make_bar_chart(labels_list, probs, display)
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{model_id}")

        st.markdown("##### Top 3 Predictions")
        top3_data = build_top3_table(labels_list, probs)
        st.table(top3_data)

        # Full probability table (collapsible)
        with st.expander("View all label probabilities"):
            all_data = []
            order = np.argsort(probs)[::-1]
            for rank, idx in enumerate(order, start=1):
                all_data.append({
                    "Rank": rank,
                    "Label": labels_list[idx],
                    "Probability": f"{probs[idx]*100:.2f}%",
                })
            st.table(all_data)

st.markdown("---")

# --- Summary comparison table ---
if num_models > 1:
    st.markdown("### 📊 Cross-Model Summary")
    summary_rows = build_summary_table(selected_models, all_results)
    st.table(summary_rows)

    # Agreement indicator
    top_preds = []
    for model_id in selected_models:
        labels_list, probs = all_results[model_id]
        top_preds.append(labels_list[int(np.argmax(probs))])

    if len(set(top_preds)) == 1:
        st.success(f"✅ All {num_models} models agree: **{top_preds[0]}**")
    else:
        st.warning(
            "⚠️ Models disagree on the top prediction: "
            + ", ".join(f"**{p}**" for p in top_preds)
        )

# --- Footer ---
st.markdown("---")
st.caption(
    "Built with [Streamlit](https://streamlit.io), "
    "[Hugging Face Transformers](https://huggingface.co/docs/transformers), "
    "and [Plotly](https://plotly.com/python/). "
    "Models: [CLIP](https://github.com/openai/CLIP) by OpenAI."
)
