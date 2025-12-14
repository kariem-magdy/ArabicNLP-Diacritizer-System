# src/app/app.py
import streamlit as st
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..config import cfg
from ..infer.infer_ensemble import EnsemblePredictor

# Paths
BILSTM_PATH = os.path.join(cfg.models_dir, "best_bilstm.pt")
TRANSFORMER_PATH = os.path.join(cfg.models_dir, "best_transformer") 
ARTIFACTS_PATH = cfg.processed_dir

st.set_page_config(page_title="Arabic Diacritization System", layout="wide")

@st.cache_resource
def get_ensemble_predictor():
    try:
        predictor = EnsemblePredictor(BILSTM_PATH, TRANSFORMER_PATH, ARTIFACTS_PATH)
        return predictor
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

predictor = get_ensemble_predictor()

st.title("🤖 Arabic Diacritization System")
st.markdown("Ensemble of **BiLSTM-CRF** and **AraBERT**.")

# Sidebar controls
st.sidebar.header("Configuration")
model_mode = st.sidebar.radio("Inference Mode", ["Ensemble", "BiLSTM Only", "Transformer Only"])

ensemble_weight = 0.5
if model_mode == "Ensemble":
    ensemble_weight = st.sidebar.slider("Ensemble Weight (0=BiLSTM, 1=Transformer)", 0.0, 1.0, 0.5)
elif model_mode == "BiLSTM Only":
    ensemble_weight = 0.0
elif model_mode == "Transformer Only":
    ensemble_weight = 1.0

# Input
text_input = st.text_area("Enter Arabic Text:", "ذهب الطالب الى المدرسة", height=100)

if st.button("Predict Tags"):
    if not predictor:
        st.error("Models not loaded.")
    elif not text_input.strip():
        st.warning("Enter text.")
    else:
        with st.spinner("Processing..."):
            words, b_tags, t_tags, final_tags = predictor.predict(text_input, ensemble_weight)
        
        # 1. Visual Result
        st.subheader("Result")
        result_html = """<div style="font-size: 1.5rem; line-height: 2.0; direction: rtl; text-align: right;">"""
        for w, tag in zip(words, final_tags):
            color = "#f0f2f6" if tag == "O" else "#fff3cd" 
            result_html += f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 4px; margin: 0 3px;" title="{tag}">{w}</span>'
        result_html += "</div>"
        st.markdown(result_html, unsafe_allow_html=True)
        
        # 2. Table Comparison
        if st.checkbox("Show Model Comparison (Last Char Tag)"):
            st.table({
                "Word": words,
                "BiLSTM": b_tags,
                "Transformer": t_tags,
                "Ensemble": final_tags
            })