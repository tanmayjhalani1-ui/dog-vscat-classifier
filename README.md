# Dog vs Cat Image Classification

An end-to-end CNN image-classification project extending the original notebook into a reproducible training and inference application. It includes CNN architecture comparison, MLflow experiment tracking, Streamlit, FastAPI, and Docker.

## Preserved original files

- DOGvsCAT.ipynb — original notebook
- dogvscat.py — original Streamlit application
- dataset — original KaggleHub download note

## Dataset

The notebook uses Kaggle dataset chetankv/dogs-cats-images. Download it with KaggleHub:

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('chetankv/dogs-cats-images'))"
```

Pass the downloaded directory containing training_set/ and test_set/ to training. Each split must contain cats/ and dogs/ folders:

```text
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

The dataset is not duplicated in this repository; the original notebook and download instructions are preserved.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --data-dir /path/to/dataset --epochs 12
```

Training creates a validation split from training_set, leaving test_set untouched until model selection is complete. Two CNN configurations are compared using validation AUC. The selected model is evaluated on the held-out test set and saved to artifacts/dog_cat_classifier.keras.

Training parameters, validation metrics, and model artifacts are logged to MLflow in mlruns/.

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Run Streamlit

```bash
streamlit run app.py
```

## Run the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- GET /health — service health
- POST /predict — upload a JPEG, PNG, or WEBP image as multipart form field file
- Interactive documentation: /docs

## Docker

```bash
docker compose up --build
```

Streamlit runs on port 8501 and FastAPI on port 8000. Model artifacts are mounted from the local artifacts/ directory. Train the model before using either inference service.

## Model design and evaluation

- Images are resized to 160 × 160 and normalized inside the model.
- Random horizontal flips, rotations, and zoom augment training images.
- CNN candidates use convolution, batch normalization, max pooling, global average pooling, and dropout.
- Early stopping monitors validation AUC; learning-rate reduction responds to validation loss.
- Final test metrics are computed only after architecture selection.

## Limitations

The model learns from the Kaggle dataset's image distribution. Confidence is the model's predicted class probability, not a guarantee of correctness. Performance can change with image quality, unusual breeds, backgrounds, lighting, and distribution shift. The model is a binary cat/dog classifier and should not be used for other animal classes.

## Original work

The original notebook, Streamlit application, and dataset-download note remain in the repository alongside the reproducible application and API implementation.
