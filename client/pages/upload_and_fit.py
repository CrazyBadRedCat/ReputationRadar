import os
import random

import streamlit as st
import logging
import requests
import pandas as pd

from configure_model import create_model_configurator
from model import model_types, Model
from wordcloud import WordCloud

logger = logging.getLogger('Streamlit_logger')
server_url = os.getenv("SERVER_URL")

st.set_page_config(page_title="Upload data and select curr_model", page_icon="fire")

st.markdown("# Pick a model to fit")
selected_model_type = st.selectbox(
    "What model would you like to train?",
    model_types.keys(),
)
logger.info(f"A chosen model is {selected_model_type}")

st.markdown("# Upload train data")


def upload_csv_to_api(file_path, op_id):
    url = f"{server_url}/api/v1/models/fit/upload_csv"
    headers = {}
    params = {'operation_id': op_id}

    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, headers=headers, params=params, files=files)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during the API call: {e}")
        return None


def send_fit_api(model_type: str, curr_model: Model, op_id: int):
    url = f"{server_url}/api/v1/models/fit"
    json = {
        "id": model_type.lower(),
        "operation_id": str(op_id),
        "hyperparameters": curr_model.get_hyperparameters()
    }

    try:
        response = requests.post(url, json=json)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred during the API call: {e}")
        st.exception("An exception occurred, try sending this model again")
        return None


def build_word_cloud(df: pd.DataFrame):
    all_text = ' '.join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    st.image(wordcloud.to_array())


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
operation_id = 0
if uploaded_file:
    temp_file_path = uploaded_file.name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    operation_id = random.randint(1, 1000)
    response = upload_csv_to_api(temp_file_path, operation_id)

    if response:
        st.success(f"CSV uploaded successfully. Operation ID: {operation_id}")
        st.write(response.json())
    else:
        st.error("CSV was not uploaded. Try to do it again")
    os.remove(temp_file_path)

    analytics_on = st.toggle("Show analytics")

    if analytics_on:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())
        st.markdown("Cloud of words")
        build_word_cloud(df)

else:
    st.write("Waiting on file upload...")

st.sidebar.markdown("# Uppload data and fit a model with it here")
st.sidebar.markdown("You need to upload a csv file with the train data.")
st.sidebar.markdown("You may also choose hyperparameters for the model.")

if uploaded_file:
    st.title("Run training")
    st.markdown("Select hyperparameters and press \"Fit Model\" to run the training process.")
    st.markdown("Note that it may take a few minutes to complete.")

    model = create_model_configurator(selected_model_type)
    logger.info(f'Created model: {model}')
    if model:
        st.success("## Result hyperparameters:")
        st.write(model.get_hyperparameters())

    if st.button("Fit model"):
        send_fit_api(selected_model_type, model, operation_id)
