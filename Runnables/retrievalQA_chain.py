from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import retrieval_qa
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

# Initialize the LLM
llm = ChatOpenAI()

# Create RetrievalQAChain
qa_chain = retrieval_qa(llm=llm,retriever=retriever)

# Ask a question
query = "What are the key takeaways from the documents?"
answer = qa_chain.run(query)

print(answer)