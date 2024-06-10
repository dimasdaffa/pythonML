import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Membaca data dari file CSV
data = pd.read_csv("warung_padang_data.csv")

# 2. Menampilkan tabel data penjualan
print("Tabel Data Penjualan:")
print(data)

# Menampilkan tabel visualisasi data penjualan
plt.figure(figsize=(10, 6))
for col in data.columns[2:]:
    plt.plot(data['hari'], data[col], label=col)
plt.xlabel('Hari')
plt.ylabel('Penjualan')
plt.title('Visualisasi Data Penjualan per Menu')
plt.legend()
plt.grid(True)
plt.show()

# 3. Memisahkan fitur dan target
X = data[["hari"]].copy()  # Membuat salinan untuk menghindari view
y_cols = [
    "rendang",
    "gulai_ayam",
    "sayur_asam",
    "sambal_ijo",
    "telur_dadar",
    "total_penjualan",
]

# 4. Membuat data baru untuk prediksi bulan depan
data_baru = pd.DataFrame({"hari": range(1, 61)})

# 5. Mengonversi tipe data kolom 'hari' menjadi string
X["hari"] = X["hari"].astype(str)
data_baru["hari"] = data_baru["hari"].astype(str)

# 6. Menggabungkan nilai-nilai 'hari' dari data pelatihan dan data baru
all_hari_values = pd.concat([X["hari"], data_baru["hari"]])

# 7. Melakukan label encoding pada semua nilai 'hari'
label_encoder = LabelEncoder()
all_hari_values = label_encoder.fit_transform(all_hari_values)

# 8. Memisahkan kembali nilai-nilai 'hari' untuk data pelatihan dan data baru
X["hari"] = all_hari_values[: len(X)]
data_baru["hari"] = all_hari_values[len(X) :]

# 9. Melakukan One-hot encoding pada kolom 'hari'
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X)

# 10. Membagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, data[y_cols], test_size=0.2, random_state=42
)

# 11. Membangun model Linear Regression untuk setiap jenis makanan
models = {}
for col in y_cols:
    model = LinearRegression()
    model.fit(X_train, y_train[col])
    models[col] = model

# 12. Melakukan One-hot encoding pada data baru
data_baru_encoded = encoder.transform(data_baru[["hari"]])

# 13. Melakukan prediksi penjualan untuk bulan depan
prediksi_penjualan = {}
for col in y_cols:
    if (
        col != "total_penjualan"
    ):  # hindari memprediksi total penjualan berdasarkan harga
        prediksi_penjualan[col] = models[col].predict(data_baru_encoded)
        total_prediksi = sum(prediksi_penjualan[col])
        print(
            f"Prediksi Total Penjualan {col.capitalize()} untuk Bulan Depan: {int(total_prediksi)} porsi"
        )

        # Menghitung rata-rata penjualan per hari pada bulan sebelumnya
        rata_rata_penjualan_sebelumnya = data[col].mean()

        # Menghitung persentase peningkatan produksi yang disarankan
        peningkatan_produksi = (
            (total_prediksi / 60 - rata_rata_penjualan_sebelumnya)
            / rata_rata_penjualan_sebelumnya
            * 100
        )
        print(
            f"Saran Peningkatan Produksi {col.capitalize()}: {peningkatan_produksi:.2f}%"
        )

        # Simulasi untuk makanan yang tidak terlalu laku (Telur Dadar)
        if col == "telur_dadar" and peningkatan_produksi < 0:
            print(
                f"Makanan {col.capitalize()} tidak terlalu laku, disarankan untuk mengurangi produksi."
            )

# 14. Menghitung total prediksi penjualan untuk bulan depan
total_penjualan = sum(
    sum(prediksi_penjualan[col]) for col in y_cols if col != "total_penjualan"
)
print(f"\nPrediksi Total Penjualan untuk Bulan Depan: {int(total_penjualan)}")


# 15. Menampilkan tabel data penjualan per menu dan plot regresi
for col in y_cols:
    # Menampilkan tabel data penjualan per menu
    print(f"\nTabel Data Penjualan {col.capitalize()}:")
    print(data[["hari", col]])

    # Plot regresi untuk setiap jenis makanan
    plt.figure(figsize=(10, 6))
    hari_asli = np.arange(1, len(data) + 1)
    hari_prediksi = np.arange(1, 61)

    # Plot data asli
    plt.scatter(hari_asli, data[col], color="blue", label="Data Asli")

    # Plot prediksi penjualan
    if col != "total_penjualan":
        plt.plot(hari_prediksi, prediksi_penjualan[col], color="red", label="Prediksi")

    plt.xlabel("Hari")
    plt.ylabel("Penjualan")
    plt.title(f"Regresi Linear Penjualan {col.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.show()
