import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Membaca data dari file CSV
data = pd.read_csv('warung_padang_data.csv')

# Memisahkan fitur dan target
X = data[['hari']]  # Menggunakan 'hari' sebagai fitur
y_cols = ['rendang', 'gulai_ayam', 'sayur_asam', 'sambal_ijo', 'telur_dadar']

# Membagi data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X, data[y_cols], test_size=0.2, random_state=42)

# Membangun model Linear Regression untuk setiap jenis makanan
models = {}
for col in y_cols:
    model = LinearRegression()
    model.fit(X_train, y_train[col])
    models[col] = model

# Membuat data baru untuk prediksi bulan depan
data_baru = pd.DataFrame({'hari': range(1, 31)})

# Melakukan prediksi penjualan untuk bulan depan
prediksi_penjualan = {}
for col in y_cols:
    prediksi_penjualan[col] = models[col].predict(data_baru)
    total_prediksi = sum(prediksi_penjualan[col])
    print(f"Prediksi Total Penjualan {col.capitalize()} untuk Bulan Depan: {int(total_prediksi)} porsi")

    # Menghitung rata-rata penjualan per hari pada bulan sebelumnya
    rata_rata_penjualan_sebelumnya = data[col].mean()

    # Menghitung persentase peningkatan produksi yang disarankan
    peningkatan_produksi = (total_prediksi / 30 - rata_rata_penjualan_sebelumnya) / rata_rata_penjualan_sebelumnya * 100
    print(f"Saran Peningkatan Produksi {col.capitalize()}: {peningkatan_produksi:.2f}%")

    # Simulasi untuk makanan yang tidak terlalu laku (Telur Dadar)
    if col == 'telur_dadar' and peningkatan_produksi < 0:
        print(f"Makanan {col.capitalize()} tidak terlalu laku, disarankan untuk mengurangi produksi.")

# Menghitung total prediksi penjualan untuk bulan depan
total_penjualan = sum(data['total_penjualan'])
print(f"\nPrediksi Total Penjualan untuk Bulan Depan: Rp {int(total_penjualan)}")