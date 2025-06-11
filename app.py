# app.py
import streamlit as st
import os
import logging
from dotenv import load_dotenv

from config import APP_TITLE
from pipeline import run_fact_checking_pipeline

# --- Logging Configuration ---
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# --- Page Configuration ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
logging.info("Streamlit app started.")

# --- Authentication (Load API keys from .env for local dev) ---
load_dotenv()
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface"]["api_token"]
        logging.info("Loaded Hugging Face API token from Streamlit secrets.")
    except:
        logging.error("Hugging Face API token not found in .env or Streamlit secrets.")
        st.error("Hugging Face API token not found! Please set it in your Streamlit secrets or a .env file.", icon="🚨")
        st.stop()

# --- UI ---
st.title(APP_TITLE)
st.markdown("This system analyzes a statement, extracts the key claim, and verifies it against a trusted fact base using a Retrieval-Augmented Generation (RAG) pipeline.")

input_text = st.text_area(
    "Enter a statement to verify:", 
    "The Indian government has announced free electricity to all farmers starting July 2025.",
    height=100
)

if st.button("🔎 Verify Statement"):
    if not input_text.strip():
        st.warning("Please enter a statement.")
        logging.warning("User submitted an empty input.")
    else:
        logging.info(f"User submitted input: {input_text}")
        with st.spinner("Analyzing... This may take a moment."):
            try:
                result = run_fact_checking_pipeline(input_text)
                logging.info(f"Pipeline result: Verdict={result['verdict']}, Confidence={result['confidence']}, Claim={result['extracted_claim']}")

                st.subheader(f"Verdict: {result['verdict']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", value=result['confidence'])
                with col2:
                    st.info(f"**Extracted Claim:** {result['extracted_claim']}")
                
                st.markdown("**Reasoning:**")
                st.info(result['reasoning'])
                
                if result['evidence']:
                    with st.expander("Show Retrieved Evidence"):
                        for ev in result['evidence']:
                            st.markdown(f"- *{ev}*")

            except FileNotFoundError as e:
                logging.exception("Initialization Error - Vector DB missing.")
                st.error(f"Initialization Error: {e}. Please ensure the vector database has been built by running `build_database.py`.", icon="🚨")
            except Exception as e:
                logging.exception("Unexpected error during pipeline execution.")
                st.error(f"An unexpected error occurred: {e}", icon="🔥")
