from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

llm = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt,llm,parser)

result = chain.invoke({'topic' : "AI"})

print(result)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

chain2 = RunnableSequence(prompt,llm,parser,prompt2,llm,parser)

result2 = chain2.invoke({'topic' : "AI"})

print(result2)