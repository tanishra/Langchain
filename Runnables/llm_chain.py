from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    template="Suggest a catchy blot title about {topic}.",
    input_variables=['topic']
)

# Create the chain
chain = prompt | llm

# Run the chain
topic = input("Enter a topic")
output = chain.invoke({"topic" : topic})

print("Generated Blot Title : ",output)