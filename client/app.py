import streamlit as st
import logging.handlers
import os

LOG_DIR = "/logs"
LOG_FILE = os.path.join(LOG_DIR, "streamlit_app.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('Streamlit_logger')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

rotating_file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
)
rotating_file_handler.setFormatter(formatter)
logger.addHandler(rotating_file_handler)

logger.info("Started an app")

st.markdown("# ReputationRadar")
st.markdown("Go to \"upload and fit\" page to train a model")
st.markdown("Then go to \"predict\" page to get predictions")
st.markdown("[GitHub](https://github.com/CrazyBadRedCat/ReputationRadar)")
