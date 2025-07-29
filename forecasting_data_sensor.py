import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Wavelet+LSTM", layout="wide")
st.title("📈 Prediksi Time Series dengan Wavelet Transform + LSTM")

# Load model & scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("model.h5")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# Fungsi membuat sequence sliding window
def create_sequences(data, window_size=60):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
    return np.array(X)

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'ddate' dan 'tag_value'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("CSV harus memiliki kolom 'ddate' dan 'tag_value'.")
    else:
        # Preprocessing
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        # Normalisasi
        scaled_data = scaler.transform(df[['tag_value']])

        # Windowing input untuk prediksi historis
        window_size = 60
        X_input = create_sequences(scaled_data, window_size)

        if len(X_input) == 0:
            st.warning("Data terlalu sedikit untuk dibuat sequence (minimal 60 baris).")
        else:
            # Prediksi historis
            y_pred_scaled = model.predict(X_input, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            df_pred = df.iloc[window_size:].copy()
            df_pred['predicted'] = y_pred

            # 🔮 Prediksi 10 Menit ke Depan (60 langkah karena data per 10 detik)
            n_future_steps = 60
            last_sequence = scaled_data[-window_size:].reshape(1, window_size, 1)
            future_predictions_scaled = []

            for _ in range(n_future_steps):
                next_pred_scaled = model.predict(last_sequence, verbose=0)[0]
                future_predictions_scaled.append(next_pred_scaled)
                last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred_scaled]], axis=1)

            future_predictions = scaler.inverse_transform(future_predictions_scaled)

            last_time = df['ddate'].iloc[-1]
            time_step = pd.Timedelta(seconds=10)
            future_dates = [last_time + (i + 1) * time_step for i in range(n_future_steps)]

            df_future = pd.DataFrame({
                'ddate': future_dates,
                'predicted': future_predictions.flatten()
            })

            # 🗃️ Tabel data historis + prediksi
            st.subheader("🗃️ Data Asli dan Prediksi Historis")
            st.dataframe(df_pred[['ddate', 'tag_value', 'predicted']].head(20))

            # 📊 Visualisasi historis
            st.subheader("📊 Visualisasi Prediksi Historis vs Data Aktual")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df_pred['ddate'], df_pred['tag_value'], label="Aktual", color='blue')
            ax1.plot(df_pred['ddate'], df_pred['predicted'], label="Prediksi", color='orange')
            ax1.set_xlabel("Waktu")
            ax1.set_ylabel("Nilai")
            ax1.legend()
            st.pyplot(fig1)

            # 🔮 Visualisasi prediksi masa depan
            st.subheader("🔮 Prediksi 10 Menit ke Depan")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(df['ddate'], df['tag_value'], label="Aktual", color='blue')
            ax2.plot(df_pred['ddate'], df_pred['predicted'], label="Prediksi Historis", color='orange')
            ax2.plot(df_future['ddate'], df_future['predicted'], label="Prediksi Masa Depan", color='green', linestyle='--')
            ax2.set_xlabel("Waktu")
            ax2.set_ylabel("Nilai")
            ax2.legend()
            st.pyplot(fig2)

            # 📄 Tabel hasil prediksi ke depan
            st.subheader("📄 Hasil Prediksi 10 Menit ke Depan")
            st.dataframe(df_future.head(10))
else:
    st.info("Silakan unggah file CSV untuk melakukan prediksi.")
