from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_classic import hub
from dotenv import load_dotenv
# from langchain.agents import AgentExecutor  # Deprecated
import requests
import os

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_api_version = os.getenv("OPENAI_API_VERSION")

search_tool = DuckDuckGoSearchRun()

llm = AzureChatOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=azure_openai_api_version,
    azure_deployment="gpt-4o-mini",       
)

# Pull the ReAct prompt from the langchain Hub
prompt = hub.pull("hwchase17/react").template

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=prompt
)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=[search_tool],
#     verbose=True
# )

response = agent.invoke({"input" : "3 ways to reach goa from delhi"})
print(response)