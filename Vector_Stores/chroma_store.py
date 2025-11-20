from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1,doc2,doc3,doc4,doc5]

vectore_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='tanish_db',
    collection_name='tanish'
)

vectore_store.add_documents(docs)

vectore_store.get(include=['embeddings','documents','metadata'])

similarity_search = vectore_store.similarity_search(
    query="Who is among these is an all rounder",
    k=2
)

print(similarity_search)

similarity_search_with_score = vectore_store.similarity_search_with_score(
    query="Who is the player from chennai superkings?",
    k=2
)

print(similarity_search_with_score)

filter = vectore_store.similarity_search_with_score(
    query="",
    filter={'team' : "mumbai Indians"}
)

print(filter)