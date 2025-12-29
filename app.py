import streamlit as st
import dspy
import os
from langchain_openai import ChatOpenAI
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- 1. KONFIGURASI API & MODEL ---
# Kita gunakan pola LangChain yang Anda berikan karena lebih stabil menangani OpenRouter
def get_llm_backend():
    api_key_secret = st.secrets["OPENROUTER_API_KEY"]
    
    # Inisialisasi dspy.LM dengan konfigurasi eksplisit
    return dspy.LM(
        model="openrouter/google/gemini-2.0-flash-lite-001",
        api_key=api_key_secret,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        extra_headers={
            "HTTP-Referer": "http://localhost:8501", # Diperlukan OpenRouter
            "X-Title": "DSPy-YOLO-Comparative-Study"
        }
    )

# --- 2. DEFINISI RISET DSPY ---
class VisualDescription(dspy.Signature):
    """
    Tugas: Ubah label objek menjadi deskripsi visual spesifik.
    Gunakan konteks tambahan untuk memperkaya deskripsi agar YOLOv11 lebih akurat.
    """
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Data tambahan dari sistem/sheets")
    refined_label = dspy.OutputField(desc="Deskripsi visual pendek untuk YOLO")

class ZeroShotOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Menggunakan ChainOfThought agar ada penalaran (Rationale) untuk paper riset
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Memaksa penggunaan backend LM yang sudah kita buat
        with dspy.context(lm=get_llm_backend()):
            return self.generator(object_name=object_name, context=context)

# --- 3. FUNGSI YOLO ---
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n-world.pt")

# --- 4. ANTARMUKA STREAMLIT ---
st.set_page_config(layout="wide")
st.title("🔬 Automated Prompt Optimization: DSPy & YOLOv11")

with st.sidebar:
    st.header("Input Parameter")
    user_input = st.text_input("Objek yang dicari:", "Safety Helmet")
    # Contoh data tambahan seperti pola yang Anda inginkan
    additional_data = st.text_area("Konteks (Data Sheets/System):", 
                                   "Kondisi area gelap, subjek menggunakan APD lengkap.")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and st.button("Jalankan Optimasi & Deteksi"):
    img = Image.open(uploaded_file).convert("RGB")
    
    # A. Tahap Optimasi Prompt (DSPy)
    with st.spinner("DSPy sedang mengoptimalkan prompt..."):
        try:
            optimizer = ZeroShotOptimizer()
            result = optimizer.forward(object_name=user_input, context=additional_data)
            optimized_prompt = result.refined_label
        except Exception as e:
            st.error(f"Gagal menghubungi OpenRouter: {str(e)}")
            st.stop()

    # B. Tahap Deteksi (YOLOv11)
    with st.spinner("YOLOv11 sedang mendeteksi..."):
        model = load_yolo()
        # Bandingkan performa
        # 1. Deteksi dengan Prompt Asli
        model.set_classes([user_input])
        res_base = model.predict(np.array(img), conf=0.25, verbose=False)[0]
        
        # 2. Deteksi dengan Prompt DSPy
        model.set_classes([optimized_prompt])
        res_opt = model.predict(np.array(img), conf=0.25, verbose=False)[0]

    # C. Tampilan Perbandingan (Comparative Study)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metode A: Baseline")
        st.image(res_base.plot()[:,:,::-1], caption=f"Prompt: {user_input}")
        st.write(f"Ditemukan: {len(res_base.boxes)} objek")

    with col2:
        st.subheader("Metode B: DSPy Optimized")
        st.image(res_opt.plot()[:,:,::-1], caption=f"Prompt: {optimized_prompt}")
        st.write(f"Ditemukan: {len(res_opt.boxes)} objek")

    st.divider()
    st.write("**Rationale (Penalaran AI):**")
    st.info(result.rationale)
