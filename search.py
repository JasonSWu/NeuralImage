import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

def get_info_identifier(model='text-davinci-001'):
    load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]

    template = """
    If the chat requires only personal information to respond to,
    then output nothing. If the chat requires specialized knowledge, 
    output a question as if you are asking for this information from a search engine.

    chat: {question}
    """

    llm = OpenAI(model=model, temperature=0.6) #create question variation. For example, if current events in general are asked about
    prompt_template = PromptTemplate(template=template, input_variables=["question"])
    identifier = LLMChain(llm=llm, prompt=prompt_template)
    return (lambda question: identifier({"question": question})['text'])

def get_summarizer(model='text-davinci-001'):
    load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]

    search = GoogleSearchAPIWrapper()

    tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

    template = """
    Summarize the response using the question's context. In other words, try to reduce
    the length as much as possible while maintaining information related to the question

    question: {question}
    text: {answer}
    """

    llm = OpenAI(model=model, temperature=0)
    prompt_template = PromptTemplate(template=template, input_variables=["answer", "question"])
    summarizer = LLMChain(llm=llm, prompt=prompt_template)
    return (lambda question: summarizer({'answer':tool.run(question), "question": question})['text'])

def retrieve_info(info_identifier, summarizer, max_words, input: str):
    identifier = info_identifier(input)
    if identifier.find("Nothing") == -1:
        return " ".join(summarizer(identifier).split(" ")[:max_words])
    return "None"