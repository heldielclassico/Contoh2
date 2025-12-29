import streamlit as st
import dspy
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YOLOv11 + DSPy Study", layout="wide")
st.title("🔬 Automated Prompt Optimization: YOLOv11 & DSPy")
st.markdown("### Comparative Study on Zero-Shot Object Classification")

# --- 1. HANDLING API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("🔑 API Key tidak ditemukan! Masukkan di Streamlit Secrets.")
    st.stop()

# --- 2. MODUL DSPY ---
class VisualDescription(dspy.Signature):
    """Mengubah label objek menjadi deskripsi visual spesifik untuk model vision."""
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Konteks lingkungan gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat, visual, dan spesifik")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Inisialisasi LM di dalam forward agar thread-safe
        lm = dspy.LM(
            model="google/gemini-2.0-flash-001", 
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False,
            extra_headers={"HTTP-Referer": "http://localhost:8501"}
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 3. FUNGSI DETEKSI YOLOv11 ---
@st.cache_resource
def load_yolo():
    # Menggunakan model world untuk open vocabulary
    return YOLO("yolo11n-world.pt")

def run_detection(image, labels):
    model = load_yolo()
    model.set_classes(labels)
    results = model.predict(image, conf=0.25, verbose=False)
    return results[0]

# --- 4. ANTARMUKA PENGGUNA & LOGIKA ---
with st.sidebar:
    st.header("Konfigurasi Input")
    target_class = st.text_input("Objek Target:", "Safety Helmet")
    env_context = st.text_input("Konteks Gambar:", "Construction site, daytime")
    run_btn = st.button("Jalankan Analisis Komparatif")

uploaded_file = st.file_uploader("Upload gambar (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file and run_btn:
    # Load Image
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # A. Optimasi Prompt melalui DSPy
    with st.spinner("DSPy sedang merumuskan prompt terbaik..."):
        try:
            optimizer = PromptOptimizer()
            dspy_res = optimizer.forward(object_name=target_class, context=env_context)
            optimized_prompt = dspy_res.refined_label
            rationale = dspy_res.rationale
        except Exception as e:
            st.error(f"Gagal mengoptimalkan prompt: {e}")
            st.stop()

    # B. Visualisasi Komparatif
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Baseline (Prompt Asli)")
        res_base = run_detection(img_array, [target_class])
        st.image(res_base.plot()[:, :, ::-1], caption=f"Label: {target_class}", use_container_width=True)
        st.metric("Objek Terdeteksi", len(res_base.boxes))

    with col2:
        st.subheader("2. DSPy (Prompt Teroptimasi)")
        res_opt = run_detection(img_array, [optimized_prompt])
        st.image(res_opt.plot()[:, :, ::-1], caption=f"Label: {optimized_prompt}", use_container_width=True)
        st.metric("Objek Terdeteksi", len(res_opt.boxes))

    # C. Detail Analisis (Bagian yang sebelumnya error)
    st.divider()
    with st.expander("📝 Lihat Analisis Semantik DSPy (Chain of Thought)"):
        st.markdown(f"**Alasan Optimasi (Rationale):**\n{rationale}")
        st.markdown(f"**Prompt Akhir yang Digunakan:** `{optimized_prompt}`")
