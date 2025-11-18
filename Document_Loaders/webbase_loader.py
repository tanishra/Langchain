from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question \n {question}  from the following text - \n {text}",
    input_variables=['question','text']
)

url = "https://www.amazon.in/Apple-MacBook-13-inch-10-core-Unified/dp/B0DZDDQ429/ref=sr_1_1_sspa?crid=1TP95ARFCCLX8&dib=eyJ2IjoiMSJ9.cxg64j71asHtIpoHuVSkM5Ehy5n1Z-roHbMIJOwdSih6CjJVJBk1BsqnUrnEbSw4EOrYiCsHcYSJlI1ggCzQGXZGVVnP7kEmpef0S0pqfjbVKje_VkxDPr2ODZ-T8FBS-94Evlh2gM-ZAO5K64gJ5XF46QibS9xnLhCCNV-rUOXRbHP7dmXyO-SADriRI6C8sT16_2duMnVTd9OLCtilDUPkhhi9KGa9OTM7JlF2EpI.7_evF5zv0EXJtdoPi951a4GGkJwhUU89b4b457n40yQ&dib_tag=se&keywords=macbook%2Bair%2Bm4&qid=1763483207&sprefix=macbook%2Caps%2C213&sr=8-1-spons&aref=jT69Sp1NzW&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs)

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


chain = prompt | llm | parser

result = chain.invoke({'question' : "What is the product?", 'text' : docs[0].page_content})

print(result)