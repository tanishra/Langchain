from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=['perm']
)

loader = TextLoader("cricket.txt",encoding='utf-8')

docs = loader.load()

print(docs)

print(type(docs))

print(len(docs))

print(docs[0])

chain = prompt | llm | parser

response = chain.invok({'poem' : docs[0].page_content})

print(response)