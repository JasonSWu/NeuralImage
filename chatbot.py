from dotenv import load_dotenv
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
import pickle
import gradio as gr
import time
from search import get_info_identifier, get_summarizer, retrieve_info

load_dotenv()
API_KEY = os.environ.get("API_KEY")
loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

prompt_template = """Respond as if you are the user's extroverted friend.
You like to play tennis, swim, and cook. Your favorite subject is geology. You hate playing basketball.
You are a male. Have a lot of personality. Avoid any kind of formality and add in informal grammar. 
Convey emotion and empathy via ascii emojis only sometimes. Be both conversational and informational.

[START CONTEXT]
{context}
[END CONTEXT]

{question}

Response:"""

manual_template = '''Info useful for responding: {context}


User: {input}'''

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=['question', 'context']
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY)
chat = ChatOpenAI(openai_api_key=API_KEY)
llm = lambda q: chat.predict(q)

qa = ConversationalRetrievalChain.from_llm(
    llm=chat,
    memory=memory,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": PROMPT},
)

info_identifier = get_info_identifier('text-davinci-003')
summarizer = get_summarizer('text-davinci-001')

max_len = 50
max_mem = 5

convo_len = 0

def chat(input):
    convo_len = int(len(memory.load_memory_variables({})['chat_history']) / 2)
    if convo_len > max_mem:
        texts_to_add = []
        for i in range(0, 2*convo_len, 2):
            oldest_QA = memory.load_memory_variables({})['chat_history'][i:i+2]
            text_to_add = f"User: {oldest_QA[0].content}\nResponse: {oldest_QA[1].content}\n\n"
            texts_to_add.append(text_to_add)
        vectorstore.add_texts(texts_to_add)
        convo_len = 0
        memory.clear()
    shortened_input = input.split(" ")[:max_len]
    search_info = retrieve_info(info_identifier, summarizer, 50, shortened_input)
    print(search_info)
    final_input = manual_template.format(
        input= " ".join(shortened_input), 
        context= search_info)
    return qa(
        {"question": final_input})['answer']

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        chat_history.append((message, chat(message)))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(share=True)