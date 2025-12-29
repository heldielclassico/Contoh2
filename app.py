import streamlit as st
import dspy
import os
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YOLOv11 + DSPy Study", layout="wide")

# --- 1. HANDLING API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY")

# --- 2. FUNGSI LOAD MODEL (DIPERBAIKI) ---
@st.cache_resource
def load_yolo():
    weights_file = "yolo11n-world.pt"
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-world.pt"
    
    if not os.path.exists(weights_file):
        with st.spinner("Mengunduh model..."):
            r = requests.get(url)
            if r.status_code == 200:
                with open(weights_file, "wb") as f:
                    f.write(r.content)
            else:
                st.error("Gagal mengunduh model. Harap upload file .pt secara manual ke GitHub.")
                st.stop()
    return YOLO(weights_file)

# --- 3. MODUL DSPY ---
class VisualDescription(dspy.Signature):
    """Optimasi label objek untuk deteksi zero-shot."""
    object_name = dspy.InputField()
    context = dspy.InputField()
    refined_label = dspy.OutputField()

class ZeroShotOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        lm = dspy.LM(
            model="openrouter/google/gemini-2.0-flash-lite-001",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://localhost:8501"}
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 4. UI ---
target_class = st.sidebar.text_input("Objek:", "Safety Helmet")
env_context = st.sidebar.text_input("Konteks:", "Bright daylight")
uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])

if uploaded_file and st.sidebar.button("Proses"):
    img = Image.open(uploaded_file).convert("RGB")
    
    # Tahap DSPy
    optimizer = ZeroShotOptimizer()
    res_dspy = optimizer.forward(object_name=target_class, context=env_context)
    
    # Tahap YOLO
    model = load_yolo()
    
    # Perbandingan
    col1, col2 = st.columns(2)
    with col1:
        model.set_classes([target_class])
        out1 = model.predict(np.array(img), conf=0.25)[0]
        st.image(out1.plot()[:,:,::-1], caption="Baseline")
        
    with col2:
        model.set_classes([res_dspy.refined_label])
        out2 = model.predict(np.array(img), conf=0.25)[0]
        st.image(out2.plot()[:,:,::-1], caption="DSPy Optimized")
        
    st.info(f"Prompt Teroptimasi: {res_dspy.refined_label}")
