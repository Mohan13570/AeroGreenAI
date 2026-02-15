import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="centered")
st.title("🌿 Tomato Disease Detection System")

# Load model
try:
    disease_model = joblib.load("disease_model.pkl")
except:
    st.error("disease_model.pkl not found. Train model first.")
    st.stop()

disease_classes = ["Healthy", "Leaf Spot", "Blight", "Mosaic Virus"]

st.subheader("Upload Tomato Leaf Image")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = image.resize((128,128))
    image_array = np.array(image)

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    features = hsv.flatten().reshape(1,-1)

    prediction = disease_model.predict(features)[0]
    confidence = disease_model.predict_proba(features).max()

    st.image(image, width=300)

    st.success(f"Detected: {disease_classes[prediction]}")
    st.info(f"Confidence: {round(confidence*100,2)}%")

    st.subheader("Recommended Action")

    if prediction == 0:
        st.success("Plant is healthy.")

    elif prediction == 1:
        st.warning("Leaf Spot detected. Use copper fungicide.")

    elif prediction == 2:
        st.error("Blight detected. Remove infected leaves immediately.")

    elif prediction == 3:
        st.error("Mosaic Virus detected. Remove infected plant to prevent spread.")
