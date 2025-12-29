import streamlit as st
import dspy
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import requests

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YOLOv11 + DSPy Study", layout="wide")
st.title("🔬 Automated Prompt Optimization: YOLOv11 & DSPy")

# --- 1. HANDLING API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("🔑 API Key tidak ditemukan! Masukkan di Streamlit Secrets.")
    st.stop()

# --- 2. MODUL DSPY ---
class VisualDescription(dspy.Signature):
    """Mengubah label objek menjadi deskripsi visual spesifik."""
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Lingkungan atau kondisi gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat, visual, dan spesifik")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        lm = dspy.LM(
            model="google/gemini-2.0-flash-001", 
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False,
            extra_headers={"HTTP-Referer": "http://localhost:8501"}
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 3. FUNGSI DETEKSI ---
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n-world.pt")

def run_detection(image, labels):
    model = load_yolo()
    model.set_classes(labels)
    results = model.predict(image, conf=0.25, verbose=False)
    return results[0]

# --- 4. UI & LOGIKA UTAMA ---
target_class = st.text_input("Objek Target:", "Safety Helmet")
env_context = st.text_input("Konteks:", "Construction site")
uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'png', 'jpeg'])

if uploaded_file and st.button("Jalankan Analisis Komparatif"):
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # A. Optimasi via DSPy
    with st.spinner("Mengoptimalkan prompt..."):
        optimizer = PromptOptimizer()
        res = optimizer.forward(object_name=target_class, context=env_context)
        optimized_prompt = res.refined_label
        rationale = res.rationale

    # B. Komparasi Deteksi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline (Prompt Asli)")
        res_base = run_detection(img_array, [target_class])
        st.image(res_base.plot()[:, :, ::-1], use_container_width=True)
        st.write(f"Terdeteksi: {len(res_base.boxes)}")

    with col2:
        st.subheader("DSPy (Prompt Teroptimasi)")
        res_opt = run_detection(img_array, [optimized_prompt])
        st.image(res_opt.plot()[:, :, ::-1], use_container_width=True)
        st.write(f"Terdeteksi: {len(res_opt.boxes)}")

    # C. Detail Analisis (Bagian yang error sebelumnya)
    st.divider()
    with st.expander("📝 Lihat Analisis Semantik DSPy (Chain of Thought)"):
        st.markdown(f"**Alasan Optimasi:**\n{rationale}")
        st.markdown(f"**Hasil Prompt Akhir:** `{optimized_prompt}`")
