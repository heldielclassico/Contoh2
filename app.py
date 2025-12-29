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

# --- 1. HANDLING API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("🔑 API Key tidak ditemukan!")
    st.stop()

# --- 2. MODUL DSPY ---
class VisualDescription(dspy.Signature):
    """Mengubah label objek menjadi deskripsi visual spesifik."""
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Konteks lingkungan gambar")
    refined_label = dspy.OutputField(desc="Deskripsi singkat, visual, dan spesifik")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # LiteLLM/OpenRouter Auth Fix
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        lm = dspy.LM(
            model="openrouter/google/gemini-2.0-flash-001",
            api_key=api_key,
            cache=False
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 3. FUNGSI DETEKSI YOLOv11 ---
@st.cache_resource
def load_yolo():
    return YOLO("yolo11n-world.pt")

def run_detection(image, labels):
    model = load_yolo()
    model.set_classes(labels)
    results = model.predict(image, conf=0.25, verbose=False)
    return results[0]

# --- 4. ANTARMUKA PENGGUNA ---
with st.sidebar:
    st.header("Konfigurasi")
    target_class = st.text_input("Objek Target:", "Safety Helmet")
    env_context = st.text_input("Konteks:", "Construction site")
    run_btn = st.button("Jalankan Analisis")

uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file and run_btn:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("Menghubungi OpenRouter..."):
        try:
            optimizer = PromptOptimizer()
            dspy_res = optimizer.forward(object_name=target_class, context=env_context)
            optimized_prompt = dspy_res.refined_label
            rationale = dspy_res.rationale
            
            # Tampilkan Hasil Komparasi
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline")
                res_base = run_detection(img_array, [target_class])
                st.image(res_base.plot()[:, :, ::-1], use_container_width=True)
            
            with col2:
                st.subheader("DSPy Optimized")
                res_opt = run_detection(img_array, [optimized_prompt])
                st.image(res_opt.plot()[:, :, ::-1], use_container_width=True)

            st.divider()
            with st.expander("Detail Analisis DSPy"):
                st.write(f"**Rationale:** {rationale}")
                st.write(f"**Final Prompt:** `{optimized_prompt}`")
                
        except Exception as e:
            st.error(f"⚠️ Kesalahan Autentikasi: {str(e)}")
            st.info("Tips: Pastikan API Key di Secrets sudah benar dan akun OpenRouter Anda memiliki kredit.")
