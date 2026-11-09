from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "dog_cat_classifier.keras"
IMAGE_SIZE = (160, 160)
CLASS_NAMES = ["cats", "dogs"]


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Train the CNN before inference.")
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)


def predict_image(model, image: Image.Image) -> dict:
    dog_probability = float(model.predict(preprocess_image(image), verbose=0)[0][0])
    index = 1 if dog_probability >= 0.5 else 0
    confidence = dog_probability if index else 1.0 - dog_probability
    return {
        "label": CLASS_NAMES[index],
        "confidence": confidence,
        "dog_probability": dog_probability,
        "cat_probability": 1.0 - dog_probability,
    }
