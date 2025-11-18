from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1,llm,parser),
    'linkedin' : RunnableSequence(prompt2,llm,parser)
})

result = parallel_chain.invoke({'topic' : "Blackhole"})

print(result)
print(result['tweet'])
print(result['linkedin'])