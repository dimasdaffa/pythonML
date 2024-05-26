import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nama', required=True, help="Masukkan Nama Anda")
parser.add_argument('-t', '--tanggallahir', required=True, help="Masukkan Tanggal Lahir Anda (format: dd-mm-yyyy)")
args = parser.parse_args()

try:
    tanggal_lahir = datetime.datetime.strptime(args.tanggallahir, '%d-%m-%Y').date()
    usia = (datetime.date.today() - tanggal_lahir).days // 365

    if usia < 30:
        panggilan = "Kakak"
    else:
        panggilan = "Bapak"

    print(f"Terima kasih telah menggunakan panggilargparse.py, {panggilan} {args.nama}")
    print(f"Tanggal lahir Anda: {tanggal_lahir}")
    print(f"Usia Anda: {usia} tahun")
except ValueError:
    print("Format tanggal lahir tidak valid. Gunakan format dd-mm-yyyy.")