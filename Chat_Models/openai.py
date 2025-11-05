# ChatOpenAI class for chat-based LLMs from LangChain
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv          

# Load API key
load_dotenv()  

# Initialize the chat-based LLM
chat_model = ChatOpenAI(model='gpt-4.1-mini')  

# Send a user message to the model and get its response
response = chat_model.invoke("Who is the prime minister of India?")  

print(response.content)  


chat_model = ChatOpenAI(model='gpt-4.1-mini', streaming=True)  

# Function to handle streaming response
def stream_response(prompt):
    response = chat_model.invoke(prompt)
    
    # Iterating over the streamed responses
    for chunk in response:
        print(chunk['text'], end='')

stream_response("Write 10 lines on India")