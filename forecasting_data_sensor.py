# Upload CSV dan validasi
uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'ddate' dan 'tag_value'", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("CSV harus memiliki kolom 'ddate' dan 'tag_value'.")
    else:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        if len(df) < 60:
            st.warning("Minimal butuh 60 data terakhir untuk prediksi.")
            st.stop()

        st.subheader("📄 Data Terakhir")
        st.dataframe(df.tail(10))

        # Ambil 60 data terakhir dan normalisasi
        last_60 = df['tag_value'].values[-60:].reshape(-1, 1)
        scaled_input = scaler.transform(last_60)

        input_seq = scaled_input.copy()
        preds = []

        for _ in range(10):  # prediksi 10 langkah ke depan
            X_input = np.array([input_seq])  # shape (1, 60, 1)
            next_scaled = model.predict(X_input)
            preds.append(next_scaled[0][0])  # simpan prediksi
            input_seq = np.append(input_seq, next_scaled)[1:]  # geser window

        # Kembalikan ke bentuk asli (inverse transform)
        preds = np.array(preds).reshape(-1, 1)
        preds_inverse = scaler.inverse_transform(preds)

        # Buat waktu prediksi: tiap 10 detik dari data terakhir
        last_time = df['ddate'].iloc[-1]
        future_times = [last_time + pd.Timedelta(seconds=10 * (i+1)) for i in range(10)]

        result_df = pd.DataFrame({
            "timestamp": future_times,
            "prediksi": preds_inverse.flatten()
        })

        # Tampilkan hasil
        st.subheader("🧮 Hasil Prediksi 10x10 Detik ke Depan")
        st.dataframe(result_df)

        # Visualisasi
        st.subheader("📊 Grafik Prediksi (10 Detik per Titik)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_df['timestamp'], result_df['prediksi'], marker='o', label="Prediksi", color='orange')
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title("Prediksi 10 Langkah ke Depan (10 Detik per Titik)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Tombol download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil Prediksi (.csv)", data=csv, file_name="prediksi_10_langkah.csv", mime="text/csv")
