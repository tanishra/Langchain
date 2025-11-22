from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def store_chunks_in_vectorstore(chunks, embedding_model: str = "text-embedding-3-small"):
    """
    Receives transcript chunks and stores them in a FAISS vectorstore.
    
    Args:
        chunks (List[str]): List of text chunks.
        embedding_model (str): OpenAI embedding model name.
        
    Returns:
        dict: {
            "vector_store": FAISS object,
            "num_chunks": int,
            "embedding_model": str
        }
    """
    # Convert chunks into Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return {
        "vector_store": vector_store,
        "num_chunks": len(chunks),
        "embedding_model": embedding_model
    }
