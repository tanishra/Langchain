# OpenAI LLM wrapper from LangChain
from langchain_openai import OpenAI
from dotenv import load_dotenv       

# Load environment variables
load_dotenv()  

# Initialize the OpenAI-compatible LLM 
llm = OpenAI(model='gpt-4.1-mini')  

# Send a prompt to the model and get its response
response = llm.invoke("What is the capital of India?")  

print(response) 
