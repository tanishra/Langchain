from langchain_core.prompts import PromptTemplate

def create_context_prompt():
    """
    Creates and returns a PromptTemplate that instructs the assistant
    to answer only using the provided transcript context.
    
    Returns:
        PromptTemplate: A reusable prompt template.
    """
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question : {query}
        """,
        input_variables=['context', 'query']
    )
    
    return prompt