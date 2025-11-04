# ChatAnthropic class for chat-based LLMs from LangChain
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Initialize chat-based LLM
chat_model = ChatAnthropic(model='claude-3.5-sonnet-20241022')

# Send message to model and get its response
response = chat_model.invoke("What is the color of sun?")

print(response.content)