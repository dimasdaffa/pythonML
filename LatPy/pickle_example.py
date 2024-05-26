import pickle

# Contoh dictionary
contoh_dictionary = {1: "6", 2: "2", 3: "f"}

# Membuka file untuk menulis dalam mode biner
with open("dict.pickle", "wb") as pickle_keluar:
    # Menyimpan dictionary ke dalam file
    pickle.dump(contoh_dictionary, pickle_keluar)
