from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    repo_id="google/gemma-2-2b-it",
    task='text-generation'
)

model = ChatHuggingFace(llm= llm)

# Prompt - 1 (Detailed Report)
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables= ['topic']
)

prompt1 = template1.invoke({"topic" : "blackhole"})

result1 = model.invoke(prompt1)

# Prompt - 2 (Summary)
template2 = PromptTemplate(
    template="Write a 5 lines summary on the following text. \n {text}",
    input_variables= ['text']
)

prompt2 = template2.invoke({"text" : result1.content})

result2 = model.invoke(prompt2)

print(result2.content)



# StrOutput
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "blackhole"})

print(result)