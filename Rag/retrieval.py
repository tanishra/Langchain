def retrieve_relevant_documents(query: str, vector_store, k: int = 4):
    """
    Retrieves relevant documents from a FAISS vector store using similarity search.

    Args:
        query (str): User's search query.
        vector_store (FAISS): The existing FAISS vector store object.
        k (int): Number of results to return.

    Returns:
        List[Document]: A list of the most relevant documents.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)
    return docs