from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

prompt = PromptTemplate(
    template="Suggest a catchy blog title about {topic}",
    input_variables=['topic']
)

topic = input("Enter a topic")

formatted_prompt = prompt.invoke(topic=topic)

blog_title = llm.invoke(formatted_prompt)

print("Generated Blog Title : ",blog_title)