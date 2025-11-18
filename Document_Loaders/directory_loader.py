from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()  # Normal Load

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

for doc in docs:
    print(doc.metadata)

# docs = loader.lazy_load() # Lazy Load

# for doc in docs:
#     print(doc.metadata)
