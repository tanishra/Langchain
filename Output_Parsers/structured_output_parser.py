from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation'
)

model =  ChatHuggingFace(llm= llm)

schema = [
    ResponseSchema(name='fact-1',description="fact 1 about the topic"),
    ResponseSchema(name='fact-2',description="fact 2 about the topic"),
    ResponseSchema(name='fact-3',description="fact 3 about the topic"),
]
parser = StructuredOutputParser.from_response_schema(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic}. \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.invoke({'topic' : 'blackhole'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({'topic' : 'blackhole'})

print(result)


# StructuredOutputParser don't  do data validation - To solve this we use Pydantic Output Parser