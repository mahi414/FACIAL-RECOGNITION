import streamlit as st
from tensorflow.keras.models import load_model

st.title("Emotion Model Test")
st.write("Loading emotion detection model...")

@st.cache_resource
def get_model():
    return load_model("models/emotion_model.keras")

model = get_model()

st.success("Model loaded successfully!")
st.write("Input shape:", str(model.input_shape))
st.write("Output shape:", str(model.output_shape))
