import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Wavelet+LSTM", layout="wide")
st.title("📈 Prediksi Time Series dengan Wavelet Transform + LSTM")

# Load model dan scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("model.h5")
    scaler = joblib.load("scaler.joblib")  # Ganti sesuai nama file scaler kamu
    return model, scaler

model, scaler = load_model_and_scaler()

# Upload CSV
uploaded_file = st.file_uploader("📁 Unggah file CSV dengan kolom 'ddate' dan 'tag_value'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("⚠️ CSV harus memiliki kolom 'ddate' dan 'tag_value'.")
    else:
        # Parsing waktu dan sort
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        # Tampilkan data terakhir
        st.subheader("📄 Data Terakhir")
        st.dataframe(df.tail(10))

        # Pastikan cukup data
        if len(df) < 60:
            st.warning("⚠️ Minimal dibutuhkan 60 baris data untuk prediksi.")
            st.stop()

        # Ambil 60 data terakhir dan normalisasi
        last_60 = df['tag_value'].values[-60:].reshape(-1, 1)
        scaled_input = scaler.transform(last_60)

        # Prediksi 10 langkah ke depan (1 langkah = 10 detik)
        input_seq = scaled_input.copy()
        preds = []

        for _ in range(10):  # 10 langkah ke depan
            X_input = np.array([input_seq])  # shape (1, 60, 1)
            next_scaled = model.predict(X_input)
            preds.append(next_scaled[0][0])
            input_seq = np.append(input_seq, next_scaled)[1:]

        # Invers transform
        preds = np.array(preds).reshape(-1, 1)
        preds_inverse = scaler.inverse_transform(preds)

        # Buat timestamp prediksi
        last_time = df['ddate'].iloc[-1]
        future_times = [last_time + pd.Timedelta(seconds=10 * (i+1)) for i in range(10)]

        # Buat DataFrame hasil
        result_df = pd.DataFrame({
            "timestamp": future_times,
            "prediksi": preds_inverse.flatten()
        })

        # Tampilkan hasil prediksi
        st.subheader("📊 Hasil Prediksi 10x10 Detik ke Depan")
        st.dataframe(result_df)

        # Visualisasi hasil
        st.subheader("📈 Grafik Prediksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_df['timestamp'], result_df['prediksi'], marker='o', label="Prediksi", color='orange')
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Prediksi 10 Langkah ke Depan (10 Detik per Titik)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Tombol download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Prediksi (.csv)", data=csv,
                           file_name="hasil_prediksi.csv", mime="text/csv")
else:
    st.info("📌 Silakan unggah file CSV terlebih dahulu.")
