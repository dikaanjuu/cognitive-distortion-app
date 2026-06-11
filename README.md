# Cognitive Distortion Detector

Aplikasi Streamlit untuk klasifikasi cognitive distortion
berbahasa Indonesia menggunakan IndoBERT Base P2.

## Alur Prediksi

1. Binary classification: membedakan No Distortion dan Distortion.
2. Jika terdeteksi Distortion, multiclass classification menentukan satu dari 11 jenis distorsi.
3. Setiap tahap memakai ensemble seed 42, 123, dan 2025.
4. Checkpoint diambil dari Hugging Face Hub dan dimuat secara sequential agar penggunaan RAM lebih ringan.

## Menjalankan Aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py