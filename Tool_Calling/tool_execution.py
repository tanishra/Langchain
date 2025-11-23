from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Tool Creation
@tool
def multiply(a : int, b : int) -> int:
    """Multiply two numbers"""
    return a * b

# LLM
llm = ChatOpenAI(model='gpt-4o-mini')

# Tool Binding
llm_with_tools = llm.bind_tools([multiply])

# llm_with_tools.invoke("Hello")
query = HumanMessage("can you multiply 4 by 6?")

messages = [query]

result = llm_with_tools.invoke(messages)

# llm_with_tools.invoke(results.tool_calls[0]['args'])

messages.append(result)

tool_message = llm_with_tools.invoke(result.tool_calls[0])

messages.append(tool_message)

final_result = llm_with_tools.invoke(messages).content