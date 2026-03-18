import io
import json
from typing import Any, Dict, List, Optional

from PIL import Image
import streamlit as st


# -----------------------------
# Replace this with your actual local VLM call
# -----------------------------
def classify_with_vlm(image: Image.Image, labels: List[str]) -> Dict[str, Any]:
    """
    Dummy classifier so the app works before the real local VLM is connected.

    Logic:
    - Picks a deterministic label based on image size
    - Generates fake normalized scores for all labels
    - Returns a mock raw response
    """
    if not labels:
        return {
            "label": "unknown",
            "confidence": None,
            "raw_response": "No labels provided.",
            "scores": None,
        }

    width, height = image.size
    index = (width * height) % len(labels)
    chosen = labels[index]

    base_scores = {}
    total = 0.0
    for i, label in enumerate(labels):
        score = 1.0 / (1 + abs(i - index))
        base_scores[label] = score
        total += score

    scores = {label: value / total for label, value in base_scores.items()}

    return {
        "label": chosen,
        "confidence": scores[chosen],
        "raw_response": f"Dummy VLM response: predicted '{chosen}' from {len(labels)} candidate labels.",
        "scores": scores,
    }


# -----------------------------
# Utilities
# -----------------------------
def parse_labels(label_text: str) -> List[str]:
    return [x.strip() for x in label_text.split(",") if x.strip()]


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Local VLM Image Classifier", page_icon="🖼️", layout="centered")
st.title("🖼️ Local VLM Image Classification")
st.caption("Upload an image, choose candidate labels, and classify with your local Visual Language Model.")

with st.sidebar:
    st.header("Settings")
    labels_text = st.text_area(
        "Candidate labels",
        value="cat, dog, car, bicycle, person",
        help="Comma-separated labels used by your VLM for classification.",
        height=120,
    )
    show_raw = st.checkbox("Show raw model response", value=True)
    show_scores = st.checkbox("Show per-label scores", value=True)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    labels = parse_labels(labels_text)

    if not labels:
        st.warning("Please provide at least one candidate label.")
    else:
        if st.button("Classify", type="primary"):
            with st.spinner("Running local VLM..."):
                result = classify_with_vlm(image, labels)

            predicted_label = result.get("label", "unknown")
            confidence = result.get("confidence")
            raw_response = result.get("raw_response")
            scores = result.get("scores")

            st.subheader("Prediction")
            st.success(f"Predicted label: **{predicted_label}**")

            if confidence is not None:
                st.write(f"Confidence: **{confidence:.2%}**")

            if show_scores and isinstance(scores, dict) and scores:
                st.subheader("Scores")
                sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(sorted_scores)
                st.json(sorted_scores)

            if show_raw and raw_response:
                st.subheader("Raw model response")
                st.code(str(raw_response), language="text")

st.divider()
st.markdown(
    """
### WE JUST NEED TO CONNECT THE REAL LOCAL VLM HERE!
)

"""
)
