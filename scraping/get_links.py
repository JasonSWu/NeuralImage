from bs4 import BeautifulSoup
import requests

url="https://zh.wikisource.org/wiki/%E6%9D%B1%E5%9D%A1%E5%85%A8%E9%9B%86"
base_url = "https://zh.wikisource.org"
html = requests.get(url).text

soup = BeautifulSoup(html, "lxml")
#with open("out.txt", "wb") as f:
#    f.write(soup.prettify(encoding="utf-8"))
with open("poem_urls.txt", "w") as f:
    for ol in soup.find_all("ol"):
        for a in ol.find_all("a"):
            f.write(base_url + a.get("href") + "\n")
    with open("wonky_volumes.txt", "rb") as wonky:
        wonky_soup = BeautifulSoup(wonky, "lxml")
        for small in wonky_soup.find_all("small"):
            for a in small.find_all("a"):
                f.write(base_url + a.get("href") + "\n")