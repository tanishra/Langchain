# Necessary classes from LangChain's HuggingFace integration
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  

# Create a text generation pipeline from a Hugging Face model
llm = HuggingFacePipeline.from_model_id(  
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
    task="text-generation", 
    pipeline_kwargs=dict(  
        temperature=0.5,  
        max_new_tokens=100 
    )
)

# Wrap the Hugging Face model into a LangChain-compatible chat model
model = ChatHuggingFace(llm=llm)  

response = model.invoke("Who is the prime minister of India?")  

print(response.content)
