import streamlit as st
import requests

st.title("Resume Role Screening")

uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    st.info("Sending file to prediction API...")

    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
    response = requests.post("http://localhost:10000/predict_role/", files=files)

    if response.status_code == 200:
        role = response.json().get("predicted_role")
        st.success(f"Predicted Role: {role}")
    else:
        st.error(f"Error: {response.json().get('error')}")
