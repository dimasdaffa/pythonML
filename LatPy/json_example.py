import json

# Contoh JSON:
x = '{ "nama": "Buchori", "umur": 22, "kota": "New York" }'

# Parsing JSON:
y = json.loads(x)

# Mencetak nilai dari kunci "umur"
print(y["umur"])
