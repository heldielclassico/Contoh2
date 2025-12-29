import streamlit as st
import dspy
import os
from langchain_openai import ChatOpenAI
from ultralytics import YOLO
import numpy as np
from PIL import Image

# --- 1. INISIALISASI MODEL (Pola Anda) ---
def get_dspy_lm():
    api_key_secret = st.secrets["OPENROUTER_API_KEY"]
    
    # Kita gunakan dspy.LM dengan adapter OpenAI agar kompatibel dengan pola ChatOpenAI
    return dspy.LM(
        model="openrouter/google/gemini-2.0-flash-lite-001",
        api_key=api_key_secret,
        api_base="https://openrouter.ai/api/v1",
        cache=False,
        extra_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "DSPy-YOLO-Comparative-Study"
        }
    )

# --- 2. SIGNATURE & MODULE DSPY ---
class VisualDescription(dspy.Signature):
    """
    Instruction: Gunakan SYSTEM_PROMPT dari secrets untuk mengoptimalkan label.
    """
    object_name = dspy.InputField(desc="Nama objek")
    context = dspy.InputField(desc="Konteks data dari Google Sheets/Input")
    refined_label = dspy.OutputField(desc="Deskripsi visual untuk YOLOv11")

class ZeroShotOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Menggunakan konteks LM yang dikonfigurasi secara dinamis
        with dspy.context(lm=get_dspy_lm()):
            return self.generator(object_name=object_name, context=context)

# --- 3. LOGIKA UTAMA STREAMLIT ---
st.title("🔬 Automated Prompt Optimization for YOLOv11")

# Contoh fungsi ambil data seperti pola Anda
def get_sheet_data():
    return "Kondisi pencahayaan redup, objek berada di area konstruksi."

# Load YOLOv11-World
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n-world.pt")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])
user_input = st.text_input("Objek yang dicari:", "Safety Helmet")

if uploaded_file and st.button("Proses"):
    # Gabungkan data Sheets + Input User
    additional_data = get_sheet_data()
    
    # Tahap DSPy
    with st.spinner("Mengoptimalkan prompt..."):
        optimizer = ZeroShotOptimizer()
        # DSPy akan otomatis menggabungkan instruksi Signature + Input
        result = optimizer.forward(object_name=user_input, context=additional_data)
        optimized_prompt = result.refined_label

    # Tahap YOLO
    img = Image.open(uploaded_file).convert("RGB")
    model = load_yolo()
    model.set_classes([optimized_prompt]) # Set label hasil optimasi
    
    yolo_results = model.predict(np.array(img), conf=0.25)
    
    # Display Side by Side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prompt Optimasi")
        st.info(optimized_prompt)
        st.write(f"Rationale: {result.rationale}")
        
    with col2:
        st.subheader("Hasil Deteksi")
        st.image(yolo_results[0].plot()[:,:,::-1])
