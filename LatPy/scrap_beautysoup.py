from urllib.request import urlopen
from bs4 import BeautifulSoup

# Pengambilan konten
url = "https://www.youtube.com/watch?v=ZlL4QLFHywI"
page = urlopen(url)
html = page.read().decode("utf-8")

# Membuat objek BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Mencetak judul halaman
print(soup.title.string)
