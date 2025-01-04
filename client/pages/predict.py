import streamlit as st
import pandas as pd
import requests
import logging
import os

import model

logger = logging.getLogger()

txt = st.text_area(
    "Input the data for prediction",
    "All work and no play makes Jack a dull boy\nAll work and no play makes Jack a dull boy"
)
server_url = os.getenv("SERVER_URL")


def get_loaded_models():
    try:
        url = f'{server_url}/api/v1/models/models'
        response = requests.get(url)
        response.raise_for_status()

        response_json = response.json()
        return [model["id"] for model in response_json.get("models", [])]
    except requests.exceptions.RequestException as e:
        st.error("Error communicating with server, try repeating request in some time")
        logger.error(f"Error communicating with server {e}")
        return None
    except Exception as e:
        st.error("An unexpected error occurred, try repeating request in some time")
        logger.exception(f"An unexpected error occurred: {e}")
        return None


def get_model_result(text):
    try:
        url = f'{server_url}/api/v1/models/predict'
        headers = {"Content-Type": "application/json"}
        payload = {
            "X": text
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        st.error("Error communicating with server, try repeating request in some time")
        logger.error(f"Error communicating with server {e}")
        return None
    except Exception as e:
        st.error("An unexpected error occurred, try repeating request in some time")
        logger.exception(f"An unexpected error occurred: {e}")
        return None


st.sidebar.markdown("# Get predictions for your input using uploaded models")

loaded_models = get_loaded_models()
models = loaded_models if loaded_models else model.model_types
current_model = st.selectbox(label='Pick a model', options=models)
if current_model:
    st.write(f"You have chosen {current_model}")

if st.button("Get Prediction"):
    with st.spinner("Asking model..."):
        summary = get_model_result(txt.split('\n'))
    if summary:
        data = {
            "text": summary["texts"],
            "prediction": summary["predictions"],
            "probabilities": [
                f"Class 0: {prob[0]:.4f}, Class 1: {prob[1]:.4f}"
                for prob in summary["probabilities"]
            ],
        }
        df = pd.DataFrame(data)

        st.subheader("The predictions are:")
        st.table(df)
    else:
        st.error("Could not retrieve text. Check the error messages above.")
