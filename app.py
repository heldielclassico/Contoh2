import streamlit as st
import dspy
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. KONFIGURASI HALAMAN & API ---
st.set_page_config(page_title="YOLOv11 + DSPy Optimizer", layout="wide")
st.title("Automated Prompt Optimization for Zero-Shot YOLOv11")

# Ambil API Key dari Streamlit Secrets atau Environment Variable
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("API Key tidak ditemukan! Masukkan OPENROUTER_API_KEY di Streamlit Secrets.")
    st.stop()

# --- 2. KONFIGURASI DSPy ---
# Menggunakan Gemini atau GPT-4o-mini via OpenRouter
lm = dspy.LM(
    model="openai/google/gemini-2.0-flash-001", 
    api_key=api_key,
    api_base="https://openrouter.ai/api/v1",
    cache=False
)
dspy.configure(lm=lm)

# Signature untuk optimasi prompt visual
class VisualDescription(dspy.Signature):
    """Meningkatkan akurasi klasifikasi dengan deskripsi visual mendetail."""
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Konteks lingkungan gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat namun spesifik untuk model vision")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        return self.generator(object_name=object_name, context=context)

# --- 3. LOGIKA DETEKSI YOLOv11 ---
@st.cache_resource
def load_yolo():
    # Menggunakan model World untuk Open Vocabulary
    return YOLO("yolo11n-world.pt")

def run_inference(image, labels):
    model = load_yolo()
    model.set_classes(labels)
    results = model.predict(image)
    return results[0]

# --- 4. ANTARMUKA PENGGUNA (UI) ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])
    target_class = st.text_input("Objek yang ingin dicari:", "Safety Helmet")
    env_context = st.text_input("Konteks lingkungan:", "Construction site, bright daylight")
    
    run_btn = st.button("Jalankan Optimasi & Deteksi")

if run_btn and uploaded_file:
    # A. Tahap Optimasi Prompt dengan DSPy
    with st.spinner("Mengoptimalkan prompt via OpenRouter..."):
        optimizer = PromptOptimizer()
        dspy_res = optimizer.forward(object_name=target_class, context=env_context)
        optimized_text = dspy_res.refined_label
        
    st.info(f"**Prompt Asli:** {target_class}\n\n**Prompt Teroptimasi (DSPy):** {optimized_text}")

    # B. Tahap Deteksi dengan YOLOv11
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    
    with st.spinner("Menjalankan YOLOv11 Zero-Shot..."):
        # Kita uji dengan prompt hasil optimasi
        detection_results = run_inference(img_array, [optimized_text])
        
        # Plot hasil
        res_plotted = detection_results.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    with col2:
        st.image(res_rgb, caption="Hasil Deteksi Zero-Shot", use_container_width=True)
        
        # Tampilkan statistik deteksi
        if len(detection_results.boxes) > 0:
            st.success(f"Ditemukan {len(detection_results.boxes)} objek!")
        else:
            st.warning("Objek tidak terdeteksi. Coba sesuaikan konteks.")

elif run_btn and not uploaded_file:
    st.error("Silakan upload gambar terlebih dahulu.")
