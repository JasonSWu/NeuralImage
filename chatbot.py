from dotenv import load_dotenv
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
import pickle
import gradio as gr

load_dotenv()
API_KEY = os.environ.get("API_KEY")
loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

prompt_template = """You are the user's extroverted friend.
You like to play tennis, swim, and cook. Your favorite subject is geology. You hate playing basketball.
You are a male. Have a lot of personality in your response. Avoid
any kind of formality and add in informal grammar. Use emojis (about 1/7 of the time) sometimes to convey emotion. For example,
with an excited sentence, sometimes append a ":)". And for sadder tones, 
append ":(" every once in a while.

{context}

User: {question}
Response:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

retriever = vectorstore.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY),
    memory=memory,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

max_len = 50
max_mem = 5

convo_len = 0

def chat(input):
    convo_len = len(memory.load_memory_variables({})['chat_history']) / 2
    if convo_len > max_mem:
        texts_to_add = []
        for i in range(0, 2*convo_len, 2):
            oldest_QA = memory.load_memory_variables({})['chat_history'][i:i+2]
            text_to_add = f"User: {oldest_QA[0].content}\nResponse: {oldest_QA[1].content}\n\n"
            texts_to_add.append(text_to_add)
        vectorstore.add_texts(texts_to_add)
        convo_len = 0
        memory.clear()
    convo_len += 1
    words = input.split(" ")
    return qa({"question": " ".join(words[:min(len(words), max_len)])})['answer']

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
demo.launch()