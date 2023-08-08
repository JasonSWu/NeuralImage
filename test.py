from langchain.chat_models import ChatOpenAI
import os, dotenv

with open("poem_urls.txt") as f:
    for line in f.readlines():
        print(line[-1])
        input()