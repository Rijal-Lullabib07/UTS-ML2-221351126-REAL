import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# Load model & scaler
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
interpreter = tf.lite.Interpreter(model_path="student-performance.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Konfigurasi halaman
st.set_page_config(page_title="🎓 Prediksi Akademik Siswa", layout="centered")

# Styling
st.markdown("""
    <style>
        body { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
        .main {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        .result-card {
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1.5rem;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: gray;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# Judul
st.markdown("<h1 style='text-align:center;'>📚 Prediksi Performa Akademik Siswa</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Masukkan data penting siswa untuk melihat prediksi tingkat akademik.</p>", unsafe_allow_html=True)

# Input Data
col1, col2 = st.columns(2)
with col1:
    school = st.selectbox("🏫 Sekolah", ["GP - Gabriel Pereira", "MS - Monte Santo"])
    sex = st.selectbox("🧑 Jenis Kelamin", ["Perempuan", "Laki-laki"])
    age = st.slider("📅 Usia", 15, 22, 17)
    studytime = st.slider("⏱️ Jam Belajar/Minggu", 1, 4, 2)
with col2:
    G1 = st.slider("📊 Nilai G1", 0, 20, 10)
    G2 = st.slider("📊 Nilai G2", 0, 20, 10)
    G3 = st.slider("📊 Nilai G3", 0, 20, 10)

failures = st.slider("❌ Pengulangan Kelas", 0, 3, 0)

# Encode input
label_map = {
    'school': {'GP': 0, 'MS': 1},
    'sex': {'Perempuan': 0, 'Laki-laki': 1},
}

user_input = {
    'school': label_map['school'][school[:2]],
    'sex': label_map['sex'][sex],
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'G1': G1,
    'G2': G2,
    'G3': G3
}

# Tambahkan default kolom yang diperlukan oleh scaler
default_values = {col: 0 for col in scaler.feature_names_in_}
default_values.update(user_input)
df_input = pd.DataFrame([default_values])[scaler.feature_names_in_]
input_scaled = scaler.transform(df_input).astype(np.float32)

# Prediksi
if st.button("🔮 Prediksi Sekarang"):
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    result = label_encoder.inverse_transform([np.argmax(output)])[0]

    if result == "high":
        st.markdown("<div class='result-card' style='background:#d4edda; color:#155724;'>🌟 Performa Akademik: <b>TINGGI</b></div>", unsafe_allow_html=True)
        st.balloons()
    elif result == "medium":
        st.markdown("<div class='result-card' style='background:#fff3cd; color:#856404;'>⚖️ Performa Akademik: <b>SEDANG</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-card' style='background:#f8d7da; color:#721c24;'>🚨 Performa Akademik: <b>RENDAH</b></div>", unsafe_allow_html=True)

    # Tentang Sekolah (Langsung Setelah Prediksi)
    st.markdown("---")
    st.markdown("### 🏫 Tentang Sekolah")
    st.markdown("""
    - **GP (Gabriel Pereira)**: Sekolah umum dengan fokus kuat pada pembelajaran akademik.
    - **MS (Monte Santo)**: Sekolah yang menekankan pengembangan karakter dan sosial.
    """)

# Footer
st.markdown("<div class='footer'>© 2025 Aplikasi Prediksi Akademik — Dibuat untuk masa depan pendidikan - RIJAL LULLABIB🌍</div>", unsafe_allow_html=True)
