import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Ganti dari pickle ke joblib
from keras.models import load_model

st.title("🔮 Prediksi LSTM 10 Menit ke Depan")

# Load model
try:
    model = load_model("model.h5")
except Exception as e:
    st.error(f"Gagal memuat model.h5: {e}")
    st.stop()

# Load scaler
try:
    scaler = joblib.load("scaler.joblib")  # Ganti dari pickle ke joblib
except FileNotFoundError:
    st.error("File scaler.joblib tidak ditemukan. Pastikan file ini ada di repository.")
    st.stop()
except Exception as e:
    st.error(f"Gagal memuat scaler: {e}")
    st.stop()

# Upload data
uploaded_file = st.file_uploader("📁 Upload data sensor (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("Kolom 'tag_value' tidak ditemukan dalam file CSV.")
        st.stop()

    st.subheader("📄 Data Asli (10 Terakhir):")
    st.dataframe(df.tail(10))

    window_size = 60
    if len(df) < window_size:
        st.warning(f"Data kurang dari {window_size} baris. Minimal harus {window_size} untuk prediksi.")
        st.stop()

    # Preprocessing
    values = df['tag_value'].values[-window_size:].reshape(-1, 1)
    scaled = scaler.transform(values)
    X_input = np.array([scaled])  # Shape: (1, 60, 1)

    # Prediction
    prediction = model.predict(X_input)
    prediction_rescaled = scaler.inverse_transform(prediction)

    # Show result
    st.subheader("📈 Prediksi 10 Menit Kedepan:")
    st.line_chart(prediction_rescaled.flatten())

