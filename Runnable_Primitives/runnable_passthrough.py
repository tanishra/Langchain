from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the summary of the following {texxt}",
    input_variables=['text']
)

chain1 = RunnableSequence(prompt1,llm,parser)

chain2 = RunnableSequence(prompt2,llm,parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : chain2
})

final_chain = RunnableSequence(chain1,parallel_chain)

result = final_chain.invoke({'topic' : "Blackhole"})

print(result)