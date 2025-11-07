from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} expert"),
#     HumanMessage(content="Explain in simple terms, what is {topic}")
# ])


# You can also use ChatPromptTemplate.from_messages()
chat_template = ChatPromptTemplate([
    ("system","You are a helpful {domain} expert"),
    ("human","Explain in simple terms, what is {topic}")
])
prompt = chat_template.invoke({
    "domain" : "cricket",
    "topic" : "ODI"
})