# OpenAIEmbeddings class from LangChain’s OpenAI integration
from langchain_openai import OpenAIEmbeddings  
from dotenv import load_dotenv  

# Load API Key
load_dotenv()  

# OpenAI embedding model with specific model name and dimension size
embedding_model = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1000)  


result = embedding_model.embed_query("Lucknow is the capital of UttarPradesh")  

print(str(result))  

# For documents
documents = [
    "Delhi is the capital of India",
    "Lucknow is the capital of UttarPradesh",
    "Paris is the capital of France"
]

result = embedding_model.embed_documents(documents)

print(str(result))
