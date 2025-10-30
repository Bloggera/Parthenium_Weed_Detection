# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🌿 Parthenium Weed Detection")

model = load_model("parthenium_detector.h5")

uploaded = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.error(f"⚠️ Parthenium Weed Detected! Confidence: {pred*100:.2f}%")
    else:
        st.success(f"✅ No Weed Detected. Confidence: {(1-pred)*100:.2f}%")
