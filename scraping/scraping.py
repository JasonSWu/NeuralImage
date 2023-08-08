from bs4 import BeautifulSoup
import requests
import re
import pickle

def remove_numbers_in_brackets(input_string):
    # Define a regular expression pattern to match numbers within brackets
    pattern = r'\[\d+\]| ' #The space after "|" is a special invisible unicode(maybe utf-8 idk)character
    
    # Use the sub() function from re module to replace matches with an empty string
    output_string = re.sub(pattern, '', input_string)
    
    return output_string

poems = open("poems.html", "w", encoding="utf-8")
with open("poem_urls.txt","r") as f:
    for line in f.readlines():
        poem_soup = BeautifulSoup(requests.get(line[:-1]).text, "lxml")
        title = poem_soup.find("span", {"class": "mw-page-title-main"})
        if title is None:
            continue
        title = "<p>" + title.getText() + "</p>"
        poem = poem_soup.find("div", {"class": "poem"})
        if poem is None:
            poem = poem_soup.find("div", {"class": "mw-parser-output"})
        if poem is None:
            continue
        poem = poem.find_all("p")
        text = "\n".join(section.getText() for section in poem)
        text_cleaned = "\n<p>\n" + remove_numbers_in_brackets(text) + "</p>"
        poems.write("<li>\n" + title + text_cleaned + "\n</li>\n")

poems.close()        