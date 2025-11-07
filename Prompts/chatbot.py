from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

chat_history = [
    SystemMessage(content="You are a helpful assistant")
]

while True:
    user_input = input("User : ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "exit" or "quit" or "q":
        break

    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))

    print("AI : ",result.content)