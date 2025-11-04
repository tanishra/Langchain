# Anthropic LLM wrapper from langchain
from langchain_anthropic import AnthropicLLM
from dotenv import load_dotenv

# Load API
load_dotenv()

# Initialize Anthropic LLM
llm = AnthropicLLM(model='claude-3.5-sonnet-20241022')

# Send a prompt to the model and get its response
response = llm.invoke('What is the color of sun?')

print(response)