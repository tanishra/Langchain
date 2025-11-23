from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_transcript_into_chunks(transcript: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits a transcript string into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        transcript (str): The full transcript text.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        List[str]: A list of chunk texts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    documents = splitter.create_documents([transcript])
    
    # Return only the text content of each chunk
    return [doc.page_content for doc in documents]