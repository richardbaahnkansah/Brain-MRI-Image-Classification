import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
import plotly.graph_objects as go

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Brain MRI Image Classification",
    layout="wide",
)

st.title("🧠 Brain MRI Image Classification")

st.write(
    "Upload a Brain MRI image and select a Hugging Face model."
)

# =====================================================
# AVAILABLE MODELS
# Add new models here only
# =====================================================

MODELS = {
    "SwinV2 Tiny Brain Tumor": {
        "repo": "amjadfqs/swinv2-tiny-patch4-window8-256-finetuned-brain-tumor",
    },
     "Brain Tumor MRI": {
         "repo": "Devarshi/Brain_Tumor_Classification",
    },
    "Swin Transformer Model": {
            "repo": "Devarshi/Brain_Tumor_Class_swin",
    },
    
}

# =====================================================
# MODEL SELECTION
# =====================================================

selected_models = st.multiselect(
    "Select one or more models",
    list(MODELS.keys()),
    default=[list(MODELS.keys())[0]]
)

if len(selected_models) == 0:
    st.warning("Please select at least one model.")
    st.stop()
# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model(model_name):

    processor = AutoImageProcessor.from_pretrained(
        model_name
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name
    )

    model.eval()

    return processor, model

# =====================================================
# PREDICTION
# =====================================================

def predict(image, model_name):

    processor, model = load_model(model_name)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    labels = model.config.id2label

    if isinstance(next(iter(labels.keys())), str):

        class_names = [
            labels[str(i)]
            for i in range(len(probabilities))
        ]

    else:

        class_names = [
            labels[i]
            for i in range(len(probabilities))
        ]

    df = pd.DataFrame({

        "Label": class_names,
        "Probability": probabilities.numpy()

    })

    df = df.sort_values(
        "Probability",
        ascending=False
    ).reset_index(drop=True)

    return df

# =====================================================
# BAR CHART
# =====================================================

# def plot_probabilities(df, title):

#     fig, ax = plt.subplots(figsize=(8, 4))

#     chart = df.sort_values(
#         "Probability",
#         ascending=True
#     )

#     ax.barh(
#         chart["Label"],
#         chart["Probability"],
#         color="#4C78A8"
#     )

#     ax.set_xlim(0, 1)

#     ax.set_xlabel(
#         "Probability"
#     )

#     ax.set_title(
#         title
#     )

#     ax.grid(
#         axis="x",
#         alpha=0.25
#     )

#     st.pyplot(
#         fig,
#         clear_figure=True
#     )

def plot_probabilities(df, title):

    chart = df.sort_values(
        "Probability",
        ascending=True
    )

    labels = chart["Label"].tolist()
    probabilities = chart["Probability"].values


    colors = []

    for i in range(len(probabilities)):

        if i == len(probabilities)-1:
            colors.append("#2563eb")  # top prediction
        else:
            colors.append("#93c5fd")  # other predictions


    fig = go.Figure(
        go.Bar(
            x=probabilities * 100,
            y=labels,
            orientation="h",
            text=[
                f"{p*100:.2f}%"
                for p in probabilities
            ],
            textposition="outside",
            marker_color=colors,

            hovertemplate=
            "<b>%{y}</b><br>"
            "Probability: %{x:.2f}%"
            "<extra></extra>"
        )
    )


    fig.update_layout(

        title=dict(
            text=title,
            font=dict(size=14)
        ),

        xaxis=dict(
            title="Probability (%)",
            range=[0,105],
            tickformat=".0f"
        ),

        yaxis=dict(
            autorange="reversed"
        ),

        height=max(
            350,
            len(labels)*35
        ),

        margin=dict(
            l=10,
            r=60,
            t=50,
            b=40
        ),

        showlegend=False,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded = st.file_uploader(
    "Upload Brain MRI",
    type=["png", "jpg", "jpeg","webp"]
)

# =====================================================
# RUN INFERENCE
# =====================================================

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    comparison = []
    results = {}

    progress = st.progress(0)

    total = len(selected_models)


    # ==============================
    # RUN ALL MODELS FIRST
    # ==============================

    for i, model_name in enumerate(selected_models):

        repo = MODELS[model_name]["repo"]

        with st.spinner(f"Running {model_name}..."):

            df = predict(image, repo)

        results[model_name] = df

        top = df.iloc[0]

        comparison.append({

            "Model": model_name,
            "Prediction": top["Label"],
            "Confidence": top["Probability"]

        })

        progress.progress((i + 1) / total)


    progress.empty()


    # ==============================
    # IMAGE + MODEL GRAPHS
    # ==============================

    # ==============================
    # IMAGE + MODEL RESULTS
    # ==============================

    st.divider()

    st.subheader("MRI Image and Model Predictions")


    image_col, model_col = st.columns(
        [1, 3],
        gap="large"
    )


    # ==============================
    # IMAGE COLUMN
    # ==============================

    with image_col:

        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Input MRI Scan",
            use_container_width=True
        )


    # ==============================
    # MODEL TABS
    # ==============================

    with model_col:

        tabs = st.tabs(selected_models)

        for tab, model_name in zip(tabs, selected_models):

            with tab:

                df = results[model_name]

                top = df.iloc[0]


                st.subheader(
                    model_name
                )


                st.success(
                    f"Prediction: {top['Label']}"
                )


                st.metric(
                    "Confidence",
                    f"{top['Probability']:.2%}"
                )


                # plot_probabilities(
                #     df,
                #     f"{model_name} Probability Distribution"
                # )
                plot_probabilities(
                    results[model_name],
                    f"{model_name} Probability Distribution"
                )


                st.markdown(
                    "### Full Prediction Scores"
                )


                st.dataframe(
                    df.style.format(
                        {
                            "Probability": "{:.4f}"
                        }
                    ),
                    hide_index=True,
                    use_container_width=True
                )

    # ==============================
    # MODEL RESULTS SIDE BY SIDE
    # ==============================

    st.divider()

    st.subheader("Model Results")


    result_columns = st.columns(
        len(selected_models)
    )


    for index, model_name in enumerate(selected_models):

        df = results[model_name]

        top = df.iloc[0]


        with result_columns[index]:

            st.write(
                f"### {model_name}"
            )


            st.success(
                top["Label"]
            )


            st.metric(
                "Confidence",
                f"{top['Probability']:.2%}"
            )


            st.dataframe(
                df.style.format({
                    "Probability":"{:.4f}"
                }),
                hide_index=True,
                use_container_width=True
            )



    # ==============================
    # COMPARISON TABLE
    # ==============================

    st.divider()

    st.subheader(
        "Model Comparison"
    )


    comparison_df = pd.DataFrame(
        comparison
    )


    comparison_df["Confidence"] = comparison_df[
        "Confidence"
    ].apply(
        lambda x: f"{x:.2%}"
    )


    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True
    )


    # ==============================
    # DOWNLOAD
    # ==============================

    csv = pd.DataFrame(
        comparison
    ).to_csv(
        index=False
    ).encode()


    st.download_button(
        "Download Comparison CSV",
        csv,
        "model_comparison.csv",
        "text/csv"
    )
st.markdown(
    """
### Notes
- This is a research demo, not a diagnostic tool.
- For medical imaging, a model fine-tuned on the exact modality and anatomy is strongly preferred.
- CLIP-style zero-shot outputs are best treated as a baseline comparison.
"""
)