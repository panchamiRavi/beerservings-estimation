import pickle
import streamlit as st
import numpy as np
from os import path
import os
# Title of the app
st.title("🍺 Beer Servings Estimation App")

# Correct model file path
# file_name = "LR_model__pipeline.pkl"  # Your model file name


current_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
model_path = os.path.join(current_dir, "model", "LR_model_pipeline.pkl")
# Correct: relative path inside 'model' folder
print(model_path)

try:
    with open(model_path, "rb") as f:
        LR_model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at: {model_path}")
    st.stop()
except PermissionError:
    st.error(f"Permission denied when accessing: {model_path}")
    st.stop()

# User inputs
country = st.selectbox(
    "Select a country",
    ["Germany", "USA", "India", "Brazil", "Czech Republic", "Ireland", "Japan"],
)

continent = st.selectbox(
    "Select a continent",
    ["Europe", "North America", "Asia", "South America", "Africa", "Oceania"],
)

beer = st.number_input("Beer servings", min_value=0)
spirit = st.number_input("Spirit servings", min_value=0)
wine = st.number_input("Wine servings", min_value=0)

# Prediction
if st.button("Predict"):
    features = np.array([[beer, spirit, wine]])
    
    try:
        prediction = LR_model.predict(features)
        st.markdown(f"### 🧪 Estimated total liters of pure alcohol: **{round(prediction[0], 2)}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
