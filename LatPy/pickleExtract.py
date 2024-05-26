import pickle

# Membuka file untuk membaca dalam mode biner
with open("dict.pickle", "rb") as pickle_masuk:
    # Memuat dictionary dari file
    contoh_dictionary = pickle.load(pickle_masuk)

# Mencetak dictionary yang telah dimuat
print(contoh_dictionary)
