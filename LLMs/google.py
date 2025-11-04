# Google wrapper from langchain
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load API
load_dotenv()

# Initialize Google LLM
llm = GoogleGenerativeAI(model='gemini-2.5-pro')

# Send message to model and get its response
response = llm.invoke("what is the currency of China?")

print(response)