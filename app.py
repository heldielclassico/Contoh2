import dspy
from ultralytics import YOLO
import os

# --- Konfigurasi OpenRouter via DSPy ---
# Anda bisa mengganti model dengan 'anthropic/claude-3.5-sonnet' atau 'google/gemini-pro-1.5'
openrouter_model = "openai/gpt-4o-mini" 
api_key = "YOUR_OPENROUTER_API_KEY"

# Konfigurasi LLM menggunakan adapter OpenAI yang diarahkan ke OpenRouter
lm = dspy.LM(
    f'openai/{openrouter_model}',
    api_key=api_key,
    api_base="https://openrouter.ai/api/v1",
    extra_headers={
        "HTTP-Referer": "http://localhost:3000", # Opsional untuk OpenRouter
        "X-Title": "DSPy Zero-Shot Research"
    }
)
dspy.configure(lm=lm)

# --- Definisi Arsitektur DSPy ---
class ZeroShotSignature(dspy.Signature):
    """Mengoptimalkan label tekstual agar lebih mudah dikenali oleh model Vision (YOLOv11)."""
    base_label = dspy.InputField(desc="Nama objek (contoh: 'helm')")
    environment = dspy.InputField(desc="Konteks gambar (contoh: 'proyek konstruksi, siang hari')")
    refined_prompt = dspy.OutputField(desc="Deskripsi visual singkat yang sangat spesifik untuk deteksi")

class PromptOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Menggunakan ChainOfThought untuk penalaran sebelum menghasilkan prompt
        self.optimizer = dspy.ChainOfThought(ZeroShotSignature)

    def forward(self, label, context):
        return self.optimizer(base_label=label, environment=context)

# --- Integrasi YOLOv11 ---
def run_comparative_inference(image_path, labels):
    # Menggunakan YOLO11 World (model Open Vocabulary)
    model = YOLO("yolo11n-world.pt") 
    
    # Set label hasil optimasi ke model
    model.set_classes(labels)
    
    # Jalankan prediksi
    results = model.predict(image_path, save=True)
    return results

# --- Eksekusi Utama ---
if __name__ == "__main__":
    # 1. Inisialisasi Optimizer
    dspy_module = PromptOptimizer()
    
    # 2. Kasus Uji: Objek yang sering sulit dideteksi secara general
    target_object = "safety vest"
    context_info = "high-visibility clothing in a busy warehouse with artificial lighting"

    print(f"--- Mengoptimalkan Prompt via OpenRouter ({openrouter_model}) ---")
    prediction = dspy_module.forward(label=target_object, context=context_info)
    
    optimized_label = prediction.refined_prompt
    print(f"Hasil Optimasi: {optimized_label}")

    # 3. Jalankan YOLOv11
    # Kita bandingkan label dasar vs label hasil optimasi DSPy
    final_labels = [optimized_label, "person", "forklift"]
    run_comparative_inference("warehouse_scene.jpg", final_labels)
