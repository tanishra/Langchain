from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
import requests
from typing import Annotated
from dotenv import load_dotenv
import os
import json

load_dotenv()

exchange_rate_api = os.getenv("EXCHANGE_RATE_API_KEY")

@tool
def get_conversion_factor(base_currency : str, target_currency : str) -> float:
    """This function fetches the currency conversion factor between a base currency and a target currency"""
    url = f"https://v6.exchangerate-api.com/v6/{exchange_rate_api}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)

    return response.json()

@tool
def convert(base_currency_value : float, conversion_factor : Annotated[float, InjectedToolArg]) -> float:
    """Given a currency conversion rate, this function calculates the target currency value from a given base currency value"""
    return base_currency_value * conversion_factor

llm = ChatOpenAI(model='gpt-4o-mini')

llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

messages = [HumanMessage("What is the conversion factor between USD and INR, based on that can you convert 10USD to INR?")]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    # print(tool_call)
    # Execute the first tool and get the value of the conversion factor
    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # print(tool_message1)
        # Fetch the conversion rate
        # tool_message1.content['conversion_rate']  It is a  json
        conversion_factor = json.loads(tool_message1.content)['conversion_rate']
        # Append Tool message to messages list
        messages.append(tool_message1)

    # Execute the second tool using the conversion factor from the first tool
    if tool_call['name'] == 'convert':
        # Fetch the current argument
        tool_call['args']['conversion_factor'] = conversion_factor
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

# print(messages)

result = llm_with_tools.invoke(messages)

print(result.content)