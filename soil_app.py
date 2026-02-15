import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="centered")
st.title("🌍 Soil Moisture Detection System")

# Load model
try:
    soil_model = joblib.load("soil_model.pkl")
except:
    st.error("soil_model.pkl not found. Train model first.")
    st.stop()

soil_classes = ["Dry Soil", "Moist Soil", "Waterlogged Soil"]

st.subheader("Upload Soil Image")

uploaded_file = st.file_uploader(
    "Upload soil surface image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = image.resize((128,128))
    image_array = np.array(image)

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    features = hsv.flatten().reshape(1,-1)

    prediction = soil_model.predict(features)[0]
    confidence = soil_model.predict_proba(features).max()

    st.image(image, width=300)

    st.success(f"Detected: {soil_classes[prediction]}")
    st.info(f"Confidence: {round(confidence*100,2)}%")

    st.subheader("Recommended Action")

    if prediction == 0:
        st.warning("Soil is dry. Irrigation required immediately.")

    elif prediction == 1:
        st.success("Soil moisture is optimal.")

    elif prediction == 2:
        st.error("Soil is waterlogged. Stop irrigation and improve drainage.")
