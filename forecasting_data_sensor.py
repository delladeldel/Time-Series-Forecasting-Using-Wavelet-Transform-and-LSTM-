import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi LSTM", layout="centered")
st.title("🔮 Prediksi LSTM 10 Menit ke Depan")

# Load model
try:
    model = load_model("model.h5")
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model.h5: {e}")
    st.stop()

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat scaler.pkl: {e}")
    st.stop()

# Upload file
uploaded_file = st.file_uploader("📁 Upload file CSV dengan kolom 'tag_value'", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("❌ Kolom 'tag_value' tidak ditemukan dalam file.")
        st.stop()

    st.subheader("📄 Data Terakhir (10 Baris):")
    st.dataframe(df.tail(10))

    window_size = 60
    if len(df) < window_size:
        st.warning(f"⚠️ Data minimal harus {window_size} baris.")
        st.stop()

    # Preprocessing
    data_values = df['tag_value'].values[-window_size:].reshape(-1, 1)
    scaled_values = scaler.transform(data_values)
    X_input = np.array([scaled_values])  # Bentuk: (1, 60, 1)

    # Prediksi
    prediction = model.predict(X_input)

    try:
        prediction_rescaled = scaler.inverse_transform(prediction)
    except Exception as e:
        st.error(f"❌ Gagal inverse transform hasil prediksi: {e}")
        st.stop()

    # Siapkan data hasil prediksi
    if prediction_rescaled.shape[1] == 1:
        pred_array = [prediction_rescaled[0][0]] * 10
    else:
        pred_array = prediction_rescaled[0]

    pred_df = pd.DataFrame(pred_array, columns=["Prediksi"])

    # Tampilkan tabel
    st.subheader("📈 Hasil Prediksi (10 Menit ke Depan):")
    st.dataframe(pred_df)

    # Tampilkan line_chart (Streamlit)
    st.subheader("📊 Grafik Sederhana (Line Chart Streamlit):")
    st.line_chart(pred_df)

    # Tampilkan Matplotlib chart
    st.subheader("📉 Grafik Prediksi (Matplotlib):")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(pred_array)+1), pred_array, marker='o', color='blue', label='Prediksi')
    plt.title("Prediksi Nilai Sensor untuk 10 Menit ke Depan")
    plt.xlabel("Menit ke-")
    plt.ylabel("Nilai Prediksi")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # Download tombol
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil Prediksi (.csv)", data=csv, file_name="prediksi_10_menit.csv", mime="text/csv")
