import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dog vs Cat", page_icon="🐱", layout="centered")

st.title("🐱🐶 Dog vs Cat Predictor")
st.markdown("---")

DATASET_PATH = "C:/Users/user/Downloads/dataset"

if not os.path.exists(DATASET_PATH):
    st.error(f"Dataset not found at: {DATASET_PATH}")
    st.stop()

train_path = os.path.join(DATASET_PATH, "training_set")
test_path = os.path.join(DATASET_PATH, "test_set")

@st.cache_resource
def build_and_train_model():
    with st.spinner("🔄 Training model... (fast version)"):

        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            train_path,
            target_size=(64, 64),   # reduced from 128
            batch_size=32,
            class_mode='binary'
        )

        test_set = test_datagen.flow_from_directory(
            test_path,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )

        model = Sequential()

        model.add(Conv2D(16, 3, activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D(2))

        model.add(Conv2D(32, 3, activation='relu'))
        model.add(MaxPooling2D(2))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # 🔥 LIMIT DATA FOR SPEED
        model.fit(
            training_set,
            epochs=2,                     # reduced from 5
            steps_per_epoch=50,           # only small subset
            validation_data=test_set,
            validation_steps=20,
            verbose=0
        )

        return model

try:
    model = build_and_train_model()
    st.success("✅ Model ready!")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.markdown("---")

st.subheader("📸 Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "gif", "bmp"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Your Image**")
        image_display = Image.open(uploaded_file)
        st.image(image_display, use_container_width=True)

    with col2:
        st.write("**Prediction Result**")

        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)
        confidence = prediction[0][0]

        if confidence > 0.5:
            result = "🐶 DOG"
            conf_value = confidence * 100
        else:
            result = "🐱 CAT"
            conf_value = (1 - confidence) * 100

        st.metric("Result", result, f"{conf_value:.2f}%")