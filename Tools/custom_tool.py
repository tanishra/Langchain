from langchain_core.tools import tool

# Create a function
def multiply(a, b):
    """Multiply two integers """
    return a * b

# Add type hints
def multiply(a : int, b : int) -> int:
    """Multiply two integers """
    return a * b

# Add tool decorator
@tool
def multiply(a : int, b : int) -> int:
    """Multiply two integers"""
    return a * b

result = multiply.invoke({'a' : 2, 'b' : 5})

print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)

print(multiply.args_schema.model_json_schema())