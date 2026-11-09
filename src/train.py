from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "dog_cat_classifier.keras"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42


def load_datasets(data_dir: Path):
    train_dir, test_dir = data_dir / "training_set", data_dir / "test_set"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError("Expected training_set/ and test_set/ folders.")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
        validation_split=0.2, subset="training", seed=SEED, shuffle=True
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary",
        validation_split=0.2, subset="validation", seed=SEED, shuffle=False
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=False
    )
    if train_ds.class_names != ["cats", "dogs"]:
        raise ValueError(f"Expected cats/ and dogs/ folders, got {train_ds.class_names}.")
    autotune = tf.data.AUTOTUNE
    return tuple(ds.cache().prefetch(autotune) for ds in (train_ds, val_ds, test_ds))


def build_model(variant: str) -> keras.Model:
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.12)(x)
    filters = [32, 64, 128] if variant == "cnn_v1" else [32, 64, 128, 192]
    for index, count in enumerate(filters):
        x = layers.Conv2D(count, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        if index:
            x = layers.Dropout(0.15)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="dog_probability")(x)
    model = keras.Model(inputs, outputs, name=variant)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class NullRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def train(data_dir: Path, epochs: int = 12, use_mlflow: bool = True) -> dict:
    train_ds, val_ds, test_ds = load_datasets(data_dir)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if use_mlflow:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("dog-vs-cat-classification")

    comparisons, best_auc, best_model, best_variant = [], -1.0, None, None
    for variant in ("cnn_v1", "cnn_v2"):
        model = build_model(variant)
        context = mlflow.start_run(run_name=variant) if use_mlflow else NullRun()
        with context:
            model.fit(
                train_ds, validation_data=val_ds, epochs=epochs, verbose=1,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor="val_auc", mode="max", patience=3, restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
                    ),
                ],
            )
            validation = {k: float(v) for k, v in model.evaluate(val_ds, return_dict=True, verbose=0).items()}
            comparisons.append({"variant": variant, **validation})
            if use_mlflow:
                mlflow.log_params({
                    "architecture": variant, "image_size": "160x160",
                    "batch_size": BATCH_SIZE, "epochs_requested": epochs, "seed": SEED
                })
                mlflow.log_metrics({f"validation_{k}": v for k, v in validation.items()})
                mlflow.tensorflow.log_model(model, artifact_path="model")
            if validation["auc"] > best_auc:
                best_auc, best_model, best_variant = validation["auc"], model, variant

    test_metrics = {k: float(v) for k, v in best_model.evaluate(test_ds, return_dict=True, verbose=0).items()}
    best_model.save(MODEL_PATH)
    metadata = {
        "model_name": best_variant,
        "class_names": ["cats", "dogs"],
        "image_size": list(IMAGE_SIZE),
        "validation_comparison": comparisons,
        "test_metrics": test_metrics,
        "selection_metric": "validation_auc",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dog-vs-cat CNN.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.data_dir, args.epochs, not args.no_mlflow), indent=2))
