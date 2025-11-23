from langchain_community.tools import tool

@tool
def add(a : int, b : int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a : int, b : int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def divide(a : int, b : int) -> int:
    """Divide two numbers"""
    return a // b

@tool
def subtract(a : int,b : int) -> int:
    """Subtract two numbers"""
    return a - b

class MathToolKit:
    def get_tools(self):
        return [add, subtract, multiply, divide]


toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "->", tool.description)