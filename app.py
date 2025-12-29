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
st.subheader("Comparative Study: Zero-Shot Classification")

# --- 1. HANDLING API KEY & CONNECTION TEST ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def verify_openrouter(key):
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"}
        )
        return response.status_code == 200
    except:
        return False

if not api_key:
    st.error("🔑 API Key tidak ditemukan di Streamlit Secrets!")
    st.stop()

# --- 2. MODUL DSPY (THREAD-SAFE) ---
class VisualDescription(dspy.Signature):
    """Mengubah label objek menjadi deskripsi visual spesifik untuk model vision."""
    object_name = dspy.InputField(desc="Nama objek dasar (misal: 'Helm')")
    context = dspy.InputField(desc="Lingkungan atau kondisi gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat, visual, dan spesifik")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Konfigurasi model di dalam forward untuk keamanan thread Streamlit
        lm = dspy.LM(
            model="google/gemini-2.0-flash-001", # Tanpa prefix openai/ jika menggunakan api_base
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False,
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "DSPy-YOLO-Study"
            }
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 3. FUNGSI DETEKSI YOLOv11 ---
@st.cache_resource
def load_yolo():
    # Mengunduh model YOLOv11-World (mendukung Open Vocabulary)
    return YOLO("yolo11n-world.pt")

def run_detection(image, labels):
    model = load_yolo()
    model.set_classes(labels)
    results = model.predict(image, conf=0.25, verbose=False)
    return results[0]

# --- 4. ANTARMUKA PENGGUNA (UI) ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    target_class = st.text_input("Objek Target:", "Safety Vest")
    env_context = st.text_input("Konteks Gambar:", "Construction worker in low light")
    st.divider()
    if st.button("Uji Koneksi OpenRouter"):
        if verify_openrouter(api_key): st.success("Koneksi Berhasil!")
        else: st.error("Koneksi Gagal / User Not Found")

uploaded_file = st.file_uploader("Upload gambar untuk klasifikasi zero-shot", type=['jpg', 'png', 'jpeg'])

if uploaded_file and target_class:
    # Persiapan Gambar
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.divider()
    
    # PROSES 1: Optimasi Prompt
    with st
