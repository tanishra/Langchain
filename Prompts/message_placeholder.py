from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Make sure that chat_history.txt exists
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

# Load chat_history
chat_history = []
with open('chat_history.txt') as f:
    chat_history.append(f.readlines())

# Create prompt
prompt = chat_template.invoke({'chat_history' : chat_history, "query" : "where is my refund"})