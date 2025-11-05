from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding_model = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=400)

documents = [
    "Virat Kohli is an Indian international cricketer and the former captain of the Indian national cricket team. He is a right-handed batsman and an occasional medium-pace bowler. He currently represents Royal Challengers Bengaluru in the IPL and Delhi in domestic cricket",
    "MS Dhoni is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper. Widely regarded as one of the most prolific wicket-keeper batsmen and captains, he represented the Indian cricket team and was the captain of the side in limited overs formats from 2007 to 2017 and in test cricket from 2008 to 2014.",
    "Sachin Tendulkar is an Indian former international cricketer who captained the Indian national team. Often dubbed the 'God of Cricket' in India, he is widely regarded as one of the greatest cricketers of all time as well as one of the greatest batsmen of all time.",
    "Rohit Sharma is an Indian international cricketer and the former captain of the Indian national cricket team. He is a right-handed batsman who plays for Mumbai Indians in Indian Premier League and for Mumbai in domestic cricket. In the year 2024 and 2025 he announced his retirement from T20Is and Test Cricket.",
    "Jasprit Bumrah is an Indian cricketer who plays for the India national team in all formats of the game and has captained India in Tests and T20Is. He is widely regarded as one of the world's best active all-format pace bowlers, and one of the best fast bowlers ever. Bumrah plays for Gujarat in domestic cricket and for Mumbai Indians in the Indian Premier League."
]

query = "Who is Virat Kohli?"

# Embeddings for all documents in the list
documents_embeddings = embedding_model.embed_documents(documents)

# Embedding for the user query
query_embeddings = embedding_model.embed_query(query)

# Compute cosine similarity between the query embedding and all document embeddings
scores = cosine_similarity([query_embeddings], documents_embeddings)[0]

# Get the index and similarity score of the most relevant document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is: ",score)

