from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

loader = PyPDFLoader("dl-curriculum.pdf")

document = loader.load()

print(document)
print(len(document))
print(document[0].page_content)
print(document[0].metadata)