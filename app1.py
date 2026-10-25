# app.py
import io
import csv
import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Medical Tumor Classification Demo", layout="wide")

st.title("Medical Imaging Tumor Classification Demo")
st.caption("Research / educational use only — not for clinical diagnosis.")

TASK_PRESETS = {
    "Brain MRI tumor": ["No Tumor", "Glioma", "Meningioma", "Pituitary"],
    "Chest X-ray abnormality": ["Normal", "Abnormal"],
    "Custom": ["No Tumor", "Tumor"],
}

MODEL_REGISTRY = {
    "CLIP Base": {
        "type": "hf_zero_shot",
        "model_id": "openai/clip-vit-base-patch32",
    },
    "CLIP Large": {
        "type": "hf_zero_shot",
        "model_id": "openai/clip-vit-large-patch14",
    },
    "CLIP H-14": {
        "type": "hf_zero_shot",
        "model_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    },
    
}

@st.cache_resource
def load_zero_shot(model_id: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="zero-shot-image-classification",
        model=model_id,
        device=device,
    )

def predict(model_id: str, image, labels):
    clf = load_zero_shot(model_id)
    preds = clf(image, candidate_labels=labels)
    df = pd.DataFrame(preds)
    df = df.rename(columns={"label": "Label", "score": "Probability"})
    df["Probability"] = df["Probability"].astype(float)
    return df.sort_values("Probability", ascending=False).reset_index(drop=True)

def bar_chart(df, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    d = df.sort_values("Probability", ascending=True)
    ax.barh(d["Label"], d["Probability"], color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, clear_figure=True)

def top3(df):
    out = df.head(3).copy()
    out["Rank"] = np.arange(1, len(out) + 1)
    return out[["Rank", "Label", "Probability"]]

def to_csv_bytes(summary_df, detail_map):
    buf = io.StringIO()
    buf.write("Summary\n")
    summary_df.to_csv(buf, index=False)
    buf.write("\n")
    for name, df in detail_map.items():
        buf.write(f"{name}\n")
        df.to_csv(buf, index=False)
        buf.write("\n")
    return buf.getvalue().encode("utf-8")

with st.sidebar:
    st.header("Configuration")
    preset = st.selectbox("Task preset", list(TASK_PRESETS.keys()), index=0)
    labels_text = st.text_area(
        "Target labels",
        value="\n".join(TASK_PRESETS[preset]),
        height=140,
        help="One label per line.",
    )
    model_names = st.multiselect(
        "Models to compare",
        list(MODEL_REGISTRY.keys()),
        default=["CLIP Base", "CLIP H-14"],
    )
    show_detail_tables = st.toggle("Show full per-label tables", value=True)

labels = [x.strip() for x in labels_text.splitlines() if x.strip()]

uploaded = st.file_uploader(
    "Upload a medical image",
    type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
)

if uploaded is None:
    st.info("Upload an image to run classification.")
    st.stop()

if not labels:
    st.error("Please provide at least one label.")
    st.stop()

if not model_names:
    st.warning("Select at least one model.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

left, right = st.columns([1, 2], gap="large")
with left:
    st.image(image, caption="Uploaded image", use_container_width=True)
    st.write("**Labels:**")
    st.write(", ".join(labels))

results = {}
summary_rows = []

with st.spinner("Running inference..."):
    for model_name in model_names:
        model_id = MODEL_REGISTRY[model_name]["model_id"]
        try:
            df = predict(model_id, image, labels)
            results[model_name] = df
            best = df.iloc[0]
            summary_rows.append(
                {
                    "Model": model_name,
                    "Model ID": model_id,
                    "Top Label": best["Label"],
                    "Top Probability": float(best["Probability"]),
                }
            )
        except Exception as e:
            err_df = pd.DataFrame(
                {"Label": labels, "Probability": [np.nan] * len(labels)}
            )
            results[model_name] = err_df
            summary_rows.append(
                {
                    "Model": model_name,
                    "Model ID": model_id,
                    "Top Label": "Error",
                    "Top Probability": np.nan,
                }
            )
            st.error(f"{model_name} failed: {e}")

summary_df = pd.DataFrame(summary_rows)

with right:
    tabs = st.tabs(model_names)
    for tab, model_name in zip(tabs, model_names):
        with tab:
            df = results[model_name]
            st.subheader(model_name)
            bar_chart(df, f"{model_name} probability distribution")
            st.markdown("### Top 3 results")
            st.dataframe(
                top3(df).style.format({"Probability": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
            if show_detail_tables:
                st.markdown("### Full probabilities")
                st.dataframe(
                    df[["Label", "Probability"]].style.format({"Probability": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

st.markdown("## Model comparison")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

csv_data = to_csv_bytes(summary_df, results)
st.download_button(
    "Download results as CSV",
    data=csv_data,
    file_name="tumor_classification_results.csv",
    mime="text/csv",
)

st.markdown(
    """
### Notes
- This is a research demo, not a diagnostic tool.
- For medical imaging, a model fine-tuned on the exact modality and anatomy is strongly preferred.
- CLIP-style zero-shot outputs are best treated as a baseline comparison.
"""
)