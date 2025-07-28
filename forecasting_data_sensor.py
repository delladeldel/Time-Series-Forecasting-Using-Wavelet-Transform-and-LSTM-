import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model

st.title("Prediksi LSTM 10 Menit ke Depan")

# Load model
model = load_model("model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Upload data
uploaded_file = st.file_uploader("Upload data sensor (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Asli:")
    st.dataframe(df.tail(10))

    # --- Preprocess ---
    # (Pastikan kolom yang dipakai sama dengan waktu training)
    # Contoh: ambil 60 data terakhir
    window_size = 60
    values = df['tag_value'].values[-window_size:].reshape(-1, 1)
    scaled = scaler.transform(values)
    X_input = np.array([scaled])  # LSTM expects shape (batch, timesteps, features)

    # --- Prediksi ---
    prediction = model.predict(X_input)
    prediction_rescaled = scaler.inverse_transform(prediction)

    st.subheader("Prediksi 10 Menit Kedepan:")
    st.line_chart(prediction_rescaled.flatten())
