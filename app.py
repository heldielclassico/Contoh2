import streamlit as st
import dspy
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YOLOv11 + DSPy Optimizer", layout="wide")
st.title("🤖 Automated Prompt Optimization for YOLOv11")
st.markdown("Studi Komparatif Klasifikasi Zero-Shot menggunakan DSPy & OpenRouter")

# --- HANDLING API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("API Key tidak ditemukan! Tambahkan OPENROUTER_API_KEY di Secrets Streamlit.")
    st.stop()

# --- DEFINISI MODUL DSPy ---
class VisualDescription(dspy.Signature):
    """Optimasi label untuk deteksi objek zero-shot."""
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Konteks lingkungan atau atribut gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat, padat, dan spesifik secara visual")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Inisialisasi LM di dalam forward untuk menghindari ThreadError di Streamlit
        lm = dspy.LM(
            model="openai/google/gemini-2.0-flash-001", 
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- FUNGSI YOLO ---
@st.cache_resource
def load_yolo_model():
    # Menggunakan model World untuk dukungan Open Vocabulary (Zero-Shot)
    return YOLO("yolo11n-world.pt")

def process_detection(image, labels):
    model = load_yolo_model()
    # Update vocabulary model secara dinamis
    model.set_classes(labels)
    results = model.predict(image, conf=0.25)
    return results[0]

# --- ANTARMUKA PENGGUNA (UI) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input & Konfigurasi")
    uploaded_file = st.file_uploader("Pilih Gambar...", type=['jpg', 'jpeg', 'png'])
    target_class = st.text_input("Objek Target (Label Dasar):", "Safety Helmet")
    env_context = st.text_input("Konteks (Opsional):", "Worker in a construction site with sunlight")
    
    run_btn = st.button("Jalankan Optimasi & Deteksi")

if uploaded_file and run_btn:
    # 1. Tahap DSPy (Optimasi Prompt)
    with st.spinner("DSPy sedang merumuskan prompt terbaik via OpenRouter..."):
        try:
            optimizer = PromptOptimizer()
            dspy_result = optimizer.forward(object_name=target_class, context=env_context)
            optimized_label = dspy_result.refined_label
            reasoning = dspy_result.rationale
        except Exception as e:
            st.error(f"Gagal menghubungi OpenRouter: {e}")
            st.stop()

    # 2. Tahap YOLOv11 (Inference)
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("YOLOv11 sedang mendeteksi objek..."):
        # Bandingkan label asli vs label optimasi (Studi Komparatif)
        res_optimized = process_detection(img_array, [optimized_label])
        
        # Plot hasil
        img_res = res_optimized.plot()
        img_res_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)

    # 3. Menampilkan Hasil
    with col2:
        st.header("Hasil Deteksi")
        st.image(img_res_rgb, caption=f"Hasil dengan Prompt: {optimized_label}", use_container_width=True)
        
        with st.expander("Lihat Detail Optimasi DSPy"):
            st.write(f"**Reasoning (CoT):** {reasoning}")
            st.write(f"**Final Prompt:** `{optimized_label}`")

        # Metrik Sederhana
        num_objects = len(res_optimized.boxes)
        st.metric("Jumlah Objek Terdeteksi", num_objects)

elif run_btn and not uploaded_file:
    st.warning("Mohon unggah gambar terlebih dahulu.")
