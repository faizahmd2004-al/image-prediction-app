import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = load_model("model/pet_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


class_names = ['cat', 'dog']

st.title("🩺 Chidinma Pet Class Prediction App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def predict_image(img):
    
    if not isinstance(img, Image.Image):
        img = Image.open(img)

    img = img.resize((128, 128))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions, axis=1)[0]]
    return predicted_class

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    result = predict_image(uploaded_file)
    st.write(f"### ✅ Prediction: {result}")
