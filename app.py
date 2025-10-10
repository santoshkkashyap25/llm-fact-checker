# app.py
import streamlit as st
import os
import logging
import time
from dotenv import load_dotenv
from datetime import datetime

from config import APP_TITLE, APP_VERSION, MAX_INPUT_LENGTH
from pipeline import run_fact_checking_pipeline
from core.metrics import metrics_collector
from core.cache import query_cache
from core.vector_db import vector_db

# --- Logging Configuration ---
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=""
)

# --- Authentication ---
load_dotenv()
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    try:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface"]["api_token"]
        logger.info("Loaded API token from Streamlit secrets")
    except:
        logger.error("API token not found")
        st.error("Hugging Face API token not configured. Please set it in Streamlit secrets.")
        st.stop()

# --- Session State Initialization ---
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'last_query_time' not in st.session_state:
    st.session_state.last_query_time = None
if 'db_loaded' not in st.session_state:
    st.session_state.db_loaded = False

# --- Wake-up Handler for Streamlit Sleep ---
def handle_wake_up():
    """Handle app wake-up after Streamlit sleep"""
    if not st.session_state.db_loaded:
        with st.spinner("Waking up the system... Loading database..."):
            try:
                vector_db.load()
                st.session_state.db_loaded = True
                logger.info("Database loaded after wake-up")
                st.success("System ready!")
                time.sleep(1)
            except FileNotFoundError:
                st.error(
                    "**Database not found!**\n\n"
                    "Please run `python build_database.py` first to initialize the fact database."
                )
                st.stop()
            except Exception as e:
                st.error(f"Error loading database: {e}")
                st.stop()

# --- Initialize on first run ---
handle_wake_up()

# --- Sidebar ---
with st.sidebar:
    st.title("About")
    st.markdown(
        "This system uses Retrieval-Augmented Generation (RAG) to verify claims "
        "against a trusted fact database."
    )
    
# --- Main UI ---
st.title(f"{APP_TITLE}")
st.markdown(
    "Enter a statement below to verify its accuracy against our trusted fact database. "
    "The system will extract the key claim, retrieve relevant evidence, and provide a verdict."
)

# Example statements
with st.expander("Try these example statements"):
    examples = [
        "India met 241 GW peak power demand on 9th June 2025 with zero shortage.",
        "The Ayushman Bharat Pradhan Mantri Jan Arogya Yojana provides health insurance coverage of up to ₹5 lakh per family per year.",
        "IREDA was granted Navratna status by the Government of India.",
        "India has 28 states and 8 union territories.",
        "The Digital India initiative was launched in 2015."
    ]
    
    for i, ex in enumerate(examples):
        if st.button(f"Example {i+1}", key=f"ex_{i}"):
            st.session_state.example_text = ex

# Input area
input_text = st.text_area(
    "Enter a statement to verify:",
    value=st.session_state.get('example_text', ''),
    height=120,
    max_chars=MAX_INPUT_LENGTH,
    help=f"Maximum {MAX_INPUT_LENGTH} characters"
)

# Character counter
if input_text:
    st.caption(f"Characters: {len(input_text)}/{MAX_INPUT_LENGTH}")

# Verify button
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    verify_button = st.button("Verify Statement", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.example_text = ""
        st.rerun()

# Process verification
if verify_button:
    if not input_text.strip():
        st.warning("Please enter a statement to verify.")
        logger.warning("Empty input submitted")
    else:
        logger.info(f"User query #{st.session_state.query_count + 1}: {input_text[:100]}")
        
        with st.spinner("Analyzing statement... This may take 10-30 seconds."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                status_text.text("Extracting claim...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                status_text.text("Searching evidence database...")
                progress_bar.progress(50)
                
                # Run pipeline
                result = run_fact_checking_pipeline(input_text)
                
                status_text.text("Generating verdict...")
                progress_bar.progress(80)
                time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                # Update session state
                st.session_state.query_count += 1
                st.session_state.last_query_time = datetime.now()
                
                logger.info(
                    f"Query completed: Verdict={result['verdict']}, "
                    f"Confidence={result['confidence']}"
                )
                
                # Display results
                st.divider()
                
                # Verdict with color coding
                verdict_colors = {
                    'True': '🟢',
                    'False': '🔴',
                    'Unverifiable': '🟡'
                }
                verdict_icon = verdict_colors.get(result['verdict'], '⚪')
                
                st.subheader(f"{verdict_icon} Verdict: {result['verdict']}")
                
                # Metrics row
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Confidence", result['confidence'])
                with col_b:
                    st.metric("Evidence Found", len(result['evidence']))
                with col_c:
                    st.metric("Processing Time", result['performance']['total_time'])
                
                # Extracted claim
                with st.expander("Extracted Claim", expanded=True):
                    st.info(f"**{result['extracted_claim']}**")
                
                # Reasoning
                st.markdown("###Reasoning")
                st.write(result['reasoning'])
                
                # Evidence
                with st.expander(f"Retrieved Evidence ({len(result['evidence'])} items)"):
                    for i, (evidence, score) in enumerate(
                        zip(result['evidence'], result['evidence_scores'])
                    ):
                        st.markdown(f"**{i+1}.** (Similarity: {score})")
                        st.markdown(f"*{evidence}*")
                        st.divider()
                
                # Performance details
                with st.expander("Performance Breakdown"):
                    perf = result['performance']
                    st.write(f"- Claim Extraction: {perf['extraction_time']}")
                    st.write(f"- Evidence Retrieval: {perf['retrieval_time']}")
                    st.write(f"- LLM Verification: {perf['llm_time']}")
                    st.write(f"- **Total: {perf['total_time']}**")
                
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                st.error(f"{str(e)}")
            except Exception as e:
                logger.exception("Unexpected error during verification")
                st.error(
                    f"An unexpected error occurred: {str(e)}\n\n"
                    "Please try again."
                )

logger.info("App render completed")