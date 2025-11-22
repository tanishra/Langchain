from langchain_openai import ChatOpenAI

def run_llm(prompt: str, model_name: str = "gpt-4.1-nano"):
    """
    Sends a formatted prompt to ChatOpenAI and returns the response content.

    Args:
        prompt (str): The final prompt string created using PromptTemplate.
        model_name (str): Optional OpenAI chat model.

    Returns:
        str: The LLM-generated content.
    """
    llm = ChatOpenAI(model=model_name)
    result = llm.invoke(prompt)
    return result.content
