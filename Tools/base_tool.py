from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a : int = Field(required=True,description="First number to multiply")
    b : int = Field(required=True,description="second number to multiply")

class MultiplyTool(BaseTool):
    name : str = 'multiply'
    description : str = "Multiply two numbers"
    args_schema : Type[BaseModel] = MultiplyInput

    def _run(self,a : int,b : int) -> int:
        return a * b
    
multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a' : 5, 'b' : 6})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)