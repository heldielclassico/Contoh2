import streamlit as st
import dspy
import os
import urllib.request
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="YOLOv11 + DSPy Study", layout="wide")
st.title("🔬 Automated Prompt Optimization: YOLOv11 & DSPy")
st.markdown("### Zero-Shot Object Classification Comparative Study")

# --- 1. PENANGANAN API KEY ---
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error("🔑 API Key tidak ditemukan! Tambahkan OPENROUTER_API_KEY di Streamlit Secrets.")
    st.stop()

# --- 2. FUNGSI LOAD MODEL (Anti FileNotFoundError) ---
@st.cache_resource
def load_yolo():
    weights_file = "yolo11n-world.pt"
    # URL Asset Resmi YOLO11
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-world.pt"
    
    if not os.path.exists(weights_file):
        with st.spinner("Mengunduh model YOLOv11-World... Ini hanya dilakukan sekali."):
            try:
                urllib.request.urlretrieve(url, weights_file)
            except Exception as e:
                st.error(f"Gagal mengunduh file model: {e}")
                st.stop()
    
    return YOLO(weights_file)

# --- 3. MODUL DSPY ---
class VisualDescription(dspy.Signature):
    """
    Tugas: Ubah label objek menjadi deskripsi visual spesifik.
    Gunakan konteks tambahan agar YOLOv11 dapat mengenali objek lebih akurat.
    """
    object_name = dspy.InputField(desc="Nama objek dasar")
    context = dspy.InputField(desc="Konteks lingkungan gambar")
    refined_label = dspy.OutputField(desc="Deskripsi visual singkat untuk YOLO")

class ZeroShotOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(VisualDescription)

    def forward(self, object_name, context):
        # Gunakan provider 'openrouter/' dan sertakan referer untuk menghindari error 401
        lm = dspy.LM(
            model="openrouter/google/gemini-2.0-flash-lite-001",
            api_key=api_key,
            api_base="https://openrouter.ai/api/v1",
            cache=False,
            extra_headers={
                "HTTP-Referer": "https://localhost:8501",
                "X-Title": "DSPy-YOLO-Riset"
            }
        )
        with dspy.context(lm=lm):
            return self.generator(object_name=object_name, context=context)

# --- 4. ANTARMUKA PENGGUNA (UI) ---
with st.sidebar:
    st.header("Input Parameter")
    target_class = st.text_input("Objek yang dicari:", "Safety Helmet")
    env_context = st.text_input("Konteks (Data System):", "Construction site, bright daylight")
    st.divider()
    run_btn = st.button("Jalankan Analisis Komparatif")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and run_btn:
    # Persiapan Gambar
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    
    # A. Tahap DSPy (Optimasi)
    with st.spinner("DSPy sedang merumuskan prompt terbaik via OpenRouter..."):
        try:
            optimizer = ZeroShotOptimizer()
            result = optimizer.forward(object_name=target_class, context=env_context)
            optimized_prompt = result.refined_label
        except Exception as e:
            st.error(f"Gagal mengoptimalkan prompt: {str(e)}")
            st.stop()

    # B. Tahap Deteksi (Komparasi)
    with st.spinner("YOLOv11 sedang melakukan inferensi..."):
        model = load_yolo()
        
        # 1. Baseline
        model.set_classes([target_class])
        res_base = model.predict(img_np, conf=0.25, verbose=False)[0]
        
        # 2. Optimized
        model.set_classes([optimized_prompt])
        res_opt = model.predict(img_np, conf=0.25, verbose=False)[0]

    # C. Tampilan Hasil
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metode A: Baseline")
        st.image(res_base.plot()[:,:,::-1], caption=f"Prompt: {target_class}")
        st.metric("Objek Terdeteksi", len(res_base.boxes))

    with col2:
        st.subheader("Metode B: DSPy Optimized")
        st.image(res_opt.plot()[:,:,::-1], caption=f"Prompt: {optimized_prompt}")
        st.metric("Objek Terdeteksi", len(res_opt.boxes))

    st.divider()
    with st.expander("📝 Lihat Detail Penalaran (Rationale)"):
        st.write(f"**Alasan AI:** {result.rationale}")
        st.write(f"**Prompt Akhir:** `{optimized_prompt}`")
