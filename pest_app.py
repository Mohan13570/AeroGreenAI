import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="centered")
st.title("🪲 Pest Detection System")

# Load trained model
try:
    pest_model = joblib.load("pest_model.pkl")
except:
    st.error("pest_model.pkl not found. Train the model first.")
    st.stop()

classes = ["No Pest Detected", "Pest Detected"]

st.subheader("Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Upload plant/leaf image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    image = image.resize((128,128))
    image_array = np.array(image)

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    features = hsv.flatten().reshape(1,-1)

    prediction = pest_model.predict(features)[0]
    confidence = pest_model.predict_proba(features).max()

    st.image(image, width=300)

    st.success(f"Prediction: {classes[prediction]}")
    st.info(f"Confidence: {round(confidence*100,2)}%")

    st.subheader("Recommended Action")

    if prediction == 0:
        st.success("Crop is healthy. No pest control required.")
    else:
        st.error("Pest infestation detected!")
        st.write("• Inspect plant immediately.")
        st.write("• Apply organic pesticide.")
        st.write("• Monitor surrounding plants.")
