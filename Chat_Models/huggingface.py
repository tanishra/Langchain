from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()

huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the HuggingFace model endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  
    task="text-generation", 
    huggingfacehub_api_token=huggingface_api_token  # Pass the token to authenticate requests
)

# Initialize the ChatHuggingFace wrapper around the LLM
chat_model = ChatHuggingFace(llm=llm)

# Use the model to generate a response for the prompt
response = chat_model.invoke("What is the capital of Russia?")

# Print the model's response
print(response.content)
