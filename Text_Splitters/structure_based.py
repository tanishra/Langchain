from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
Virat Kohli is a legendary Indian cricketer widely regarded as one of the greatest batsmen of all time. Hailing from Delhi, he captained the Indian national team across all three international formats and led the Under-19 team to a World Cup victory in 2008. Kohli is renowned for his aggressive batting style, exceptional consistency, and intense passion for the game, earning him the nickname King Kohli.

His career is studded with numerous records, including the most individual hundreds in One Day International (ODI) matches, surpassing Sachin Tendulkar's record of 49 in the 2023 ODI World Cup. He also holds the record for the most runs in a single ODI World Cup edition, with 765 in 2023. Kohli was a key member of the Indian squads that won the 2011 ODI World Cup, the 2013 ICC Champions Trophy, the 2024 T20 World Cup, and the 2025 ICC Champions Trophy.
In the Indian Premier League (IPL), Kohli is the highest run-scorer and led the Royal Challengers Bengaluru to their first IPL title in 2025. Following the 2024 T20 World Cup triumph, Kohli retired from T20 International cricket and announced his retirement from Test cricket in May 2025. He remains an inspirational figure known for his dedication, fitness, and ability to thrive under pressure.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)

chunks = splitter.split_documents(text)

print(len(chunks))
print(chunks[0])
print(chunks[0].page_content)