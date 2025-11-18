from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=['topic']
)

chain1 = RunnableSequence(prompt1,llm,parser)

def word_count(text):
    return len(text.split())

word_counter_runnable = RunnableLambda(word_count)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : word_counter_runnable
})

final_chain = RunnableSequence(chain1,parallel_chain)

result = final_chain.invoke({'topic' : "AI"})

print(result)

final_result = """joke - {} \n word count - {}""".format(result['joke'],result['word_count'])

print(final_result)