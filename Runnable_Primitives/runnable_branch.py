from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed report on this {topic}",
    input_variables=['topic']
)

chain1 = RunnableSequence(prompt1,llm,parser)

def word_count(text):
    return len(text.split())


prompt2 = PromptTemplate(
    template="Summarize the following text \n {text}",
    input_variables=['text']
)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500, RunnableSequence(prompt2,llm,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(chain1,branch_chain)

result = final_chain.invoke({'topic' : "Blackhole"})

print(result)