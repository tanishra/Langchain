from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Load the document
loader = TextLoader("doc.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings and store in FAISS
vectorstore = FAISS.from_documents(docs,OpenAIEmbeddings())

# Create a retriever (fetches relevant documents)
retriever = vectorstore.as_retriever()

# Manually retrieve relevant documents
query = "What are the key takeaways from the document?"
retrieved_docs = retriever._get_relevant_documents(query)

# Combine retrieved text into a single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Initialize the LLM
llm = ChatOpenAI()

# Manually pass retrieved text to LLM
prompt = f"Based on the following text, answer the question: {query} \n \n {retrieved_text}"
answer = llm.invoke(prompt)

# Print the answer
print("Answer : ",answer)