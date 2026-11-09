import json

import streamlit as st
from PIL import Image

from src.inference import load_model, predict_image
from src.train import ROOT

st.set_page_config(page_title="Dog vs Cat Vision", page_icon="🐾", layout="wide")
st.title("Dog vs Cat Vision")
st.caption("CNN image classification with confidence scores and model evaluation.")

MODEL_PATH = ROOT / "artifacts" / "dog_cat_classifier.keras"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"


@st.cache_resource
def get_model():
    return load_model()


prediction_tab, report_tab = st.tabs(["Image prediction", "Model report"])
with prediction_tab:
    uploaded = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with right:
            try:
                result = predict_image(get_model(), image)
                is_dog = result["label"] == "dogs"
                st.metric("Prediction", "DOG 🐶" if is_dog else "CAT 🐱", f'{result["confidence"]:.1%} confidence')
                st.progress(result["confidence"])
                st.caption(f'Dog probability: {result["dog_probability"]:.1%}')
                st.caption(f'Cat probability: {result["cat_probability"]:.1%}')
            except FileNotFoundError as exc:
                st.error(str(exc))

with report_tab:
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        metrics = metadata["test_metrics"]
        st.subheader("Held-out test performance")
        cols = st.columns(4)
        for col, key in zip(cols, ["accuracy", "auc", "precision", "recall"]):
            col.metric(key.upper(), f"{metrics[key]:.3f}")
        st.write("CNN architecture comparison on validation data")
        st.dataframe(metadata["validation_comparison"], use_container_width=True)
        st.caption("The test set is evaluated after architecture selection using validation AUC.")
    else:
        st.info("Train a model first. See the README for dataset and training instructions.")
