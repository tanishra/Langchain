from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
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

llm_with_tools.invoke("Can you multiply 2 by 5?").tool_calls[0]

# LLM does not actually run the tool - It just suggests the tool and the input arguments.
# The actual execution is handled by LangChain or programmer.