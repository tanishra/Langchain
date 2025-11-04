# ChatGoogle class for chat-based LLMs from langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API
load_dotenv()

# Initialize chat-based LLM
chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

# Send message to model and get its response
response = chat_model.invoke("What is the capital of China?")

print(response.content)