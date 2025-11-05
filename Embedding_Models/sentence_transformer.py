from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

result = embedding_model.embed_query(text)

print(str(result))

# For documents
documents = [
    "Delhi is the capital of India",
    "Paris is the capital of India",
    "Lucknow is the capital of UttarPradesh"
]

vector = embedding_model.embed_documents(documents)

print(str(vector))