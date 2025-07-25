import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# CONFIG
WINDOW_SIZE = 60
N_PREDICT = 60
MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.pkl'

@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("📈 Forecasting 10 Menit ke Depan")
st.write("Upload minimal 60 data terakhir dalam 1 kolom CSV")

uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Format file tidak dikenali.")
        st.stop()

    try:
        data = df.iloc[:, 0].values.astype(float)
    except:
        st.error("Pastikan file hanya memiliki satu kolom angka di kolom pertama.")
        st.stop()

    last_60 = data[-WINDOW_SIZE:].reshape(-1, 1)
    last_scaled = scaler.transform(last_60)
    input_seq = last_scaled.reshape(1, WINDOW_SIZE, 1)

    predictions = []
    current_input = input_seq.copy()

    for _ in range(N_PREDICT):
        pred = model.predict(current_input, verbose=0)[0][0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    st.subheader("Grafik Prediksi 10 Menit ke Depan")
    fig, ax = plt.subplots()
    ax.plot(range(len(data)), data, label="Data Aktual")
    ax.plot(range(len(data), len(data)+N_PREDICT), predictions, color='red', label="Prediksi")
    ax.axvline(len(data)-1, color='gray', linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Tabel Prediksi")
    pred_df = pd.DataFrame({
        "Langkah ke-": list(range(1, N_PREDICT+1)),
        "Prediksi": predictions
    })
    st.dataframe(pred_df)

else:
    st.info("Silakan upload file CSV dengan minimal 60 data.")
