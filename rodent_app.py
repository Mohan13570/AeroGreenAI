import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="centered")
st.title("🐀 Rodent Detection System")

try:
    rodent_model = joblib.load("rodent_model.pkl")
except:
    st.error("rodent_model.pkl not found. Train model first.")
    st.stop()

classes = ["No Rodent Detected", "Rodent Detected"]

st.subheader("Upload Field Image")

uploaded_file = st.file_uploader(
    "Upload farm field image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = image.resize((128,128))
    image_array = np.array(image)

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    features = hsv.flatten().reshape(1,-1)

    prediction = rodent_model.predict(features)[0]
    confidence = rodent_model.predict_proba(features).max()

    st.image(image, width=300)

    st.success(f"Prediction: {classes[prediction]}")
    st.info(f"Confidence: {round(confidence*100,2)}%")

    if prediction == 1:
        st.error("🚁 Alert: Deploy drone immediately to investigate rodent activity.")
    else:
        st.success("Field secure. No rodent activity detected.")
