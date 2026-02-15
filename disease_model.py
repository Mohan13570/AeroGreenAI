from PIL import Image
import numpy as np
import joblib
import cv2
import streamlit as st

# Load trained ML model
disease_model = joblib.load("disease_model.pkl")

disease_classes = ["Healthy", "Leaf Spot", "Blight", "Mosaic Virus"]

st.subheader("🌿 Disease Detection (ML Model)")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img = img.resize((128,128))
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = cv2.resize(gray, (64,64)).flatten().reshape(1,-1)

    prediction = disease_model.predict(features)[0]

    st.image(img, caption="Uploaded Image", width=200)
    st.success(f"Detected Disease: {disease_classes[prediction]}")
