from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables= ['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

# | LCEL - Langchain Expression Language
chain = prompt | model | parser

result = chain.invoke({"topic" : "Blackhole"})

print(result)

chain.get_graph().print_ascii()