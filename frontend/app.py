import streamlit as st
import requests

st.title("AI Gender Detection Dashboard")

uploaded_file = st.file_uploader("Upload Face Image")

if uploaded_file:

    st.image(uploaded_file)

    files = {"file": uploaded_file.getvalue()}

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files=files
    )

    result = response.json()

    st.success("Prediction: " + result["prediction"])