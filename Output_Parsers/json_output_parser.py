from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation'
)

model =  ChatHuggingFace(llm= llm)


parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, city and age of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke()

# result = model.invoke(prompt)

# # print(result)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))


chain = template | model | parser

result = chain.invoke({})

print(result)

# You can't enforce a schema on json parser - To solve this we use Structured Output Parser