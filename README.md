# LangChain
**LangChain** is a framework designed to simplify the development of applications powered by **language models** (like OpenAI’s GPT, Anthropic’s Claude, or open-source models like LLaMA and Falcon).  
It provides modular components to help developers connect **LLMs, prompts, tools, memory, agents, retrievers, and more** into structured, production-ready pipelines.

## 📚 Table of Contents

- [Goals of This Repository](#goals-of-this-repository)
- [Current Progress](#current-progress)
  - [Models Component](#-models-component)
    - [Language Models](#-language-models)
    - [Embedding Models](#-embedding-models)
  - [🧩 Semantic Search Example](#-semantic-search-example)
  - [🗣️ Prompts Component](#-prompts-component)
  - [📦 Structured Output](#-structured-output)
  - [🧮 Output Parsers](#-output-parsers)
  - [🔗 Chains Component](#-chains-component)
  - [⚡️ Runnables Component](#-runnable-component)
  - [📄 Document Loaders](#-document-loaders)
  - [✂️ Text Splitters](#-text-splitters)
  - [🧮 Vector Store](#-vector-store)
  - [🔍 Retrievers](#-retrievers)
  - [🧠 Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [🛠️ Tools Component](#tool-component)
- [Upcoming Topics](#-upcoming-topics)
- [Installation & Setup](#-installation--setup)
- [Technologies Used](#-technologies-used)
- [Contribution](#-contribution)

---

## 🎯 Goals of This Repository

- Learn and understand **LangChain’s main components** through practical coding examples.
- Build small, functional mini-projects using these components.
- Document the journey for **educational and reference** purposes.

---

## ✅ Current Progress

### 1️⃣ **Models Component**

This section focuses on **LangChain’s Model abstraction**, covering both **Language Models** and **Embedding Models**.

#### 💬 Language Models
Explored how to use different types of language models for text generation and conversation.

- **Closed-Source Models:**
  - OpenAI (e.g., `gpt-4`, `gpt-3.5-turbo`)
  - Anthropic (e.g., `claude`)
- **Open-Source Models:**
  - Hugging Face models (e.g., `TinyLlama`, `Mistral`, `Falcon`)
  - Integration via `langchain_huggingface`

**Topics covered:**
- Text generation
- Temperature & token control
- Model invocation using `invoke()` and `stream()`
- Comparing performance between models

#### 🧭 Embedding Models
Explored how embeddings represent text as numerical vectors for semantic understanding.

- **Closed-Source Embeddings:**
  - OpenAI Embeddings (`text-embedding-3-small`, `text-embedding-3-large`)
- **Open-Source Embeddings:**
  - Hugging Face embeddings (`sentence-transformers`, etc.)

**Topics covered:**
- Creating embeddings for text
- Embedding documents and queries
- Calculating similarity using **cosine similarity**

---

### 🧩 Semantic Search Example

Implemented a **simple semantic search** using OpenAI embeddings.

**Workflow:**
1. Embed a small collection of text documents.  
2. Embed a user query.  
3. Use **cosine similarity** to find the most semantically similar document.  
4. Return the top-matching document and similarity score.

**Concepts demonstrated:**
- How vector representations can power information retrieval.
- Practical usage of embeddings for semantic similarity.

---

### 🗣️ Prompts Component

Learned about how **LangChain** manages and structures **prompts** — the input instructions given to language models.

#### 🧱 Key Concepts
- **ChatPromptTemplate** — Helps design reusable prompt templates for conversations or structured inputs.  
  Example:
  ```python
  from langchain.prompts import ChatPromptTemplate

  prompt = ChatPromptTemplate.from_template("Translate the following text into French: {text}")
  formatted_prompt = prompt.format_messages(text="Hello, how are you?")

- **MessagePlaceHolder:** Acts as a placeholder for inserting previous messages or dynamic context (useful in chat applications).
- **Messages:** Learned how LangChain structures interactions through message types:
 - **SystemMessage** – sets rules or behavior of the assistant
 - **HumanMessage** – user input
 - **AIMessage** - model response

---


#### 🧠 Mini Projects
- **Chatbot** — Built a simple chatbot using `ChatPromptTemplate` and message history.
- **Research Paper Summarizer** — Created a summarization tool that accepts a research paper as input and outputs a concise summary using prompt templates.

**Concepts covered:**
- Designing reusable prompt templates.
- Managing context in prompts.
- Using placeholders for dynamic message injection.


---


### 📦 Structured Output

Explored how to get **structured and reliable outputs** from LLMs instead of free-form text.

#### 🧩 Importance
Structured output ensures that responses from models can be programmatically parsed and integrated into applications (e.g., JSON, dictionaries, typed objects).

#### 🛠️ Methods Learned
- **TypedDict** — Used Python’s `typing.TypedDict` to define expected data structures and guide LLM output.
  ```python
  from typing import TypedDict

  class MovieInfo(TypedDict):
      title: str
      genre: str
      rating: float
      ```
- **Pydantic** — Used Pydantic models to enforce schema validation and easily fetch structured outputs from LLMs.
  ```python
  from pydantic import BaseModel
  
  class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str
    ```

  These techniques help ensure that model outputs are consistent, machine-readable, and validated, which is critical for production-grade applications.

  ---

### 🧮 Output Parsers

Learned about **Output Parsers** in LangChain, which help transform LLM text responses into structured formats for further processing.

#### 🔍 Types of Output Parsers
- **StrOutputParser** — Parses and returns the output as plain text (useful for simple responses).
- **JsonOutputParser** — Parses model output formatted as JSON strings into Python dictionaries.
- **StructuredOutputParser** — Enforces a predefined schema for the output using format instructions.
- **PydanticOutputParser** — Leverages Pydantic models to parse and validate the LLM’s structured responses.

#### 🧠 Key Learnings
- How to attach output parsers to LLM chains for structured responses.
- The importance of combining prompt templates with output parsers for reliable pipelines.
- Handling model parsing errors gracefully.

**Example:**
```python
from langchain.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template("Return a JSON object with 'name' and 'age' fields for a fictional person.")

chain = prompt | parser
result = chain.invoke({})
print(result)
```

--- 


## ⚙️ Installation & Setup 
1. **Clone this repository:**
  ```bash
   git clone https://github.com/tanishra/Langchain.git
   cd Langchain
   ```
2. **Create and activate a virtual environment:**
  ```python
  python -m venv venv
  source venv/bin/activate     # For Linux/Mac
  venv\Scripts\activate
  ```
3. **Install dependencies:**
  ```bash
  ip install -r requirements.txt
  ```
4. **Add your API keys to .env file:**
  ```bash
  OPENAI_API_KEY=your_openai_api_key_here
  HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
  ANTHROPIC_API_KEY=your_anthropic_api_key_here
  GOOGLE_API_KEY=your_google_api_key_here
  ````
---

### 🔗 Chains Component

Chains are powerful abstractions that connect multiple components — **models, prompts, parsers, and logic** — into a **pipeline** for more complex workflows.

---

#### ⚙️ Types of Chains Explored

##### 1️⃣ **Simple Chains**
A basic combination of a **prompt** and a **model**.  
Input flows directly into the prompt, which then sends formatted input to the model.

**Example:**
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Create the prompt template
prompt = PromptTemplate.from_template(
    "What is a good name for a company that makes {product}?"
)

# Instantiate the chat LLM
llm = ChatOpenAI(model="gpt-4-turbo")

# Initialize Parser
parser = StrOutputParser()

# Compose the chain using the ‘pipe’ style
chain = prompt | llm | parser

# Invoke the chain
response = chain.invoke({"product": "AI-powered drones"})
print(response)

```

#### 2️⃣ **Sequential Chains**
Execute multiple chains in sequence, where the output of one chain becomes the input to the next.

**Example:**
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize the model
llm = ChatOpenAI(model="gpt-4-turbo")

# Define prompts
prompt1 = PromptTemplate.from_template("Generate a company name for {product}")
prompt2 = PromptTemplate.from_template("Write a tagline for {company_name}")

# Define output parser
parser = StrOutputParser()

# Create first chain: product -> company_name
chain1 = prompt1 | llm | parser

# Create second chain: company_name -> tagline
chain2 = prompt2 | llm | parser

# Combine chains sequentially
overall_chain = chain1 | chain2

# Run the pipeline
result = overall_chain.invoke({"product": "AI chatbots"})

print(result)

```

#### 3️⃣ Parallel Chains
Run multiple chains simultaneously on the same input and gather all outputs together.
Useful for generating multiple perspectives or information types from a single input.

**Example:**
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Initialize the model
llm = ChatOpenAI(model="gpt-4-turbo")

# Define prompts
summary_prompt = PromptTemplate.from_template("Summarize: {text}")
sentiment_prompt = PromptTemplate.from_template("What is the sentiment of this text: {text}?")

# Output parser
parser = StrOutputParser()

# Define parallel branches
summary_chain = summary_prompt | llm | parser
sentiment_chain = sentiment_prompt | llm | parser

# Run both in parallel
chain = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain
)

# Invoke
result = chain.invoke({"text": "LangChain makes building with LLMs super efficient!"})
print(result)

```

#### 4️⃣ Conditional Chains
Choose which chain to execute based on input conditions.
Enables dynamic logic flow in your pipelines.

**Example**
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# Initialize the model
llm = ChatOpenAI(model="gpt-4-turbo")

# Output parser
parser = StrOutputParser()

# Define the two branches
weather_chain = (
    PromptTemplate.from_template("Provide weather info for {query}") |
    llm |
    parser
)

general_chain = (
    PromptTemplate.from_template("Answer the query: {query}") |
    llm |
    parser
)

# Define branching logic
branch = RunnableBranch(
    (lambda x: "weather" in x["query"].lower(), weather_chain),
    (lambda x: True, general_chain)  # fallback branch (like "else")
)

# Invoke the chain
result = branch.invoke({"query": "What’s the weather like in Paris?"})
print(result)

```

#### 🧠 Key Learnings
- How to compose multiple LLM components into reusable pipelines.
- Differences between Sequential, Parallel, and Conditional logic.
- How to control data flow between chains using LangChain’s Runnable interfaces.
- How to design multi-step workflows with structured input and output.

---

### ⚡️ Runnables Componenet
Runnables define how data flows through pipelines and give extremely fine-grained control over execution.

Runnables are executable components that take an input → process it → return output.

They can represent:
- Models
- Prompts
- Parsers
- Functions
- Conditional logic
- Parallel pipelines
- Entire workflows

#### Why Runnables?
Before runnables, LangChain relied on rigid chain classes.

Runnables solve key limitations by offering:

- High modularity
- Flexible composition
- Support for parallel & conditional execution
- Ability to add custom Python functions
- Unified interface across LLMs, embeddings, retrievers, tools, prompts

#### How Runnables Work
Every runnable supports these methods:

- .invoke(input) → run synchronously
- .ainvoke(input) → async
- .batch(inputs) → multiple inputs
- .stream(input) → output streaming
- | (pipe operator) → compose components like Unix pipelines

#### 🧱 Runnable Primitives Explored

#### 1️⃣ RunnableSequence
A sequence of steps executed one after another.
Output of step A becomes input to step B.

```python
from langchain_core.runnables import RunnableSequence, RunnableLambda

pipeline = RunnableSequence(
    steps=[
        RunnableLambda(lambda x: x * 2),
        RunnableLambda(lambda x: x + 10)
    ]
)

print(pipeline.invoke(5))  # Output: 20
```
#### 2️⃣ RunnableParallel
Runs multiple tasks simultaneously on the same input.

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

parallel_ops = RunnableParallel(
    square=RunnableLambda(lambda x: x ** 2),
    cube=RunnableLambda(lambda x: x ** 3),
)

print(parallel_ops.invoke(4))
```

#### 3️⃣ RunnableLambda
Wraps a simple Python function as a runnable.

Great for custom logic, preprocessing, filtering, and data transformations.

```python
from langchain_core.runnables import RunnableLambda

double = RunnableLambda(lambda x: x * 2)
print(double.invoke(7))  # Output: 14
```

#### 4️⃣ RunnablePassThrough
Passes the input unchanged but allows multiple runnable branches to process it.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassThrough
from langchain_core.runnables import RunnableLambda

pipeline = (
    RunnablePassThrough()
    | RunnableParallel(
        original=RunnableLambda(lambda x: x),
        doubled=RunnableLambda(lambda x: x * 2),
    )
)

print(pipeline.invoke(5))
```

#### 5️⃣ RunnableBranch
Implements conditional logic — similar to if / elif / else.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

branch = RunnableBranch(
    (lambda x: x > 50, RunnableLambda(lambda x: "High value")),
    (lambda x: x > 20, RunnableLambda(lambda x: "Medium value")),
    (lambda x: True, RunnableLambda(lambda x: "Low value"))
)

print(branch.invoke(30))

```
##### 6️⃣ Task-Specific Runnables
These wrap core LangChain modules as runnables:
- RunnablePrompt
- RunnableLLM
- RunnableEmbeddings
- RunnableRetriever
- RunnableTool

This makes ANY LangChain component plug-and-play inside runnable pipelines.

#### **Example – Prompt → LLM → Parser**
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Explain {topic} in simple words.")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

pipeline = prompt | llm | parser

print(pipeline.invoke({"topic": "quantum mechanics"}))
```

##### 🎯 Key Learnings
- Runnables power the entire execution engine of LangChain.
- You can build complex pipelines using sequence, parallel, and conditional logic.
- RunnableLambda allows inserting custom Python logic anywhere in the flow.
- RunnableParallel helps run multiple tasks at once.
- RunnableBranch enables dynamic decision-making inside workflows.
- All LLM components (models, retrievers, tools, prompts) convert into reusable runnables.
- Runnables = the most powerful and flexible abstraction in LangChain.

### 📄 Document Loaders
Document Loaders are the entry point of any retrieval or RAG pipeline.

They help you load data from various sources—files, web pages, APIs, cloud storage, or entire directories—into a structured Document format that LangChain understands.

Document Loaders allow you to quickly ingest raw text, PDFs, CSV files, websites, and more so they can be processed, chunked, embedded, and searched.

#### Why Document Loaders?
LangChain provides a unified interface for loading data from many formats.

Key advantages:
- Consistent Document structure regardless of source
- Automatic handling of metadata
- Ability to process large datasets
- Easily combine multiple loaders for multi-source ingestion
- Supports local + remote sources (files, URLs, APIs, GitHub repos, etc.)

Document loaders are essential for Semantic Search, RAG, Chat-with-your-data, and AI agents that rely on external information.

#### Types of Document Loaders

#### 1️⃣ Text Loader
Loads simple .txt files.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt")
docs = loader.load()

print(docs)
```

#### 2️⃣ PDF Loader
PDFs often contain multi-column text, images, and complex layouts.

LangChain provides robust PDF loaders like PyPDFLoader.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()

print(docs[0].page_content[:500])
```

#### 3️⃣ CSV Loader
Loads CSV files row-by-row as separate documents.

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="data.csv")
docs = loader.load()

print(docs[0])
```

#### 4️⃣ Web-Based Loader
Loads HTML content from a URL using WebBaseLoader.

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
docs = loader.load()

print(docs[0].page_content[:300])
```

#### 5️⃣ Directory Loader
Loads all files in a folder, while automatically selecting appropriate loaders based on file extensions.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "my_docs/",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()
print(len(docs))
```

#### 🧠 How Document Loaders Work
Each loader outputs a list of Document objects:

```python
{
  "page_content": "...actual text...",
  "metadata": {
      "source": "filename",
      "page": 1,
      ...
  }
}
```

This uniform structure allows:
- Chunking
- Embedding
- Semantic search
- Vector storage
- Query understanding

No matter what the original file format was.

---

### ✂️ Text Splitter
Text Splitters in LangChain are used to divide a document or text into smaller, more manageable chunks. These chunks can be processed individually for a variety of purposes, such as indexing, embedding, or semantic search. Text splitting is particularly useful when working with large documents that need to be broken down into smaller sections before being processed by models or passed to other components in the pipeline.

LangChain provides several types of text splitters, each designed to split documents based on different criteria, such as length, structure, or semantic meaning. These splitters help ensure that the chunks are meaningful and usable for downstream tasks like semantic search or question-answering.

#### Why Use Text Splitters?

- **Handling Large Texts**: Split long documents into smaller pieces for more efficient processing.
- **Optimizing Search**: Create smaller chunks for better relevance in semantic search.
- **Data Preprocessing**: Prepare text for embeddings by splitting into manageable units.

#### Types of Text Splitters

LangChain provides the following text splitters:

1️⃣ **Length-Based Splitter**  
This splitter divides the text into chunks based on a specified length. It is useful for ensuring that chunks are a manageable size for embedding or processing.

```python
from langchain_text_splitters import CharacterTextSplitter

# Split text into chunks of 1000 characters
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
docs = splitter.split_text("Your very long document goes here.")

print(docs)
```

2️⃣ Structure-Based Splitter
This splitter divides the text based on predefined structure or delimiters, such as paragraphs, sentences, or specific tags. It helps in ensuring that chunks align with logical breaks in the text.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split text by paragraphs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30
)
docs = splitter.split_text("Your text with multiple paragraphs goes here.")

print(docs)
```

3️⃣ Document-Structure-Based Splitter
This splitter uses the structure of the document (e.g., sections, headers) to break it into chunks. It is ideal for documents that follow a clear structure, such as reports, guides, or books.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

docs = splitter.split_text(text)

print(docs)
```

4️⃣ Markdown-Based Splitter
If your documents are in Markdown format, this splitter will break the text based on Markdown headings. It can be useful for processing Markdown files where sections are clearly defined by # tags.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

docs = splitter.split_text(text)

print(docs)
```

5️⃣ Semantic Meaning-Based Splitter
This splitter divides the text based on its semantic content rather than just structure or length. It uses models to identify meaningful sections of the text that are semantically relevant. This is ideal for more advanced use cases like summarization or extracting key concepts from long documents.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])

print(docs)
```


#### 🧮 Vector Store

A **Vector Store** is a storage system designed to store and manage vector embeddings. Vector embeddings are high-dimensional representations of text, images, or other data types that are generated by machine learning models. These embeddings capture semantic meaning in a way that allows for efficient similarity search, clustering, and other tasks in natural language processing (NLP) and AI.

In LangChain, a **Vector Store** is an essential component for tasks like semantic search, retrieval-augmented generation (RAG), and other applications that require understanding the meaning of documents or queries. The Vector Store allows you to index and retrieve vectors based on their similarity, enabling you to perform advanced search operations such as finding similar documents or retrieving relevant context for question-answering.

#### Why is a Vector Store Required?

Vector Stores are required for tasks where you need to:

- **Store high-dimensional vectors**: Machine learning models, such as those for text or image embeddings, generate high-dimensional vectors that represent semantic meaning. Vector Stores allow you to efficiently store these vectors.
- **Perform similarity search**: You need a way to compare vectors efficiently and retrieve the most relevant ones for a query. This is where vector stores shine—they allow fast, approximate nearest neighbor (ANN) search.
- **Enable RAG (Retrieval-Augmented Generation)**: By using vector stores, you can augment your generation models with real-time information retrieval, enabling more accurate and context-aware responses.
- **Efficient clustering**: Vector stores can be used to cluster similar items together, which is useful for organizing large datasets or building recommendation systems.

#### Vector Store vs. Vector Database

While the terms "vector store" and "vector database" are sometimes used interchangeably, there is a subtle difference between the two:

- **Vector Store**: Refers specifically to a storage system that is optimized for storing and searching vector embeddings. It usually provides functionalities like indexing, similarity search, and vector retrieval. A vector store is often a part of a broader system, like LangChain, that uses it for specific tasks like semantic search or document retrieval.
  
- **Vector Database**: A vector database is a broader term that often includes more advanced capabilities, such as transactional support, distributed storage, and advanced querying features beyond similarity search. While a vector store might focus primarily on vector search, a vector database might also support metadata queries, data governance, and scalability features.

In short, a **Vector Store** is more focused on the retrieval and management of vector embeddings, while a **Vector Database** is a more feature-rich, enterprise-level system that includes support for broader data storage and management tasks.

#### Examples of Vector Stores in LangChain
LangChain supports integration with several popular vector stores, each offering unique features suited to different use cases. Below are some of the vector stores that can be easily used with LangChain for managing and querying vector embeddings:

1️⃣ **FAISS (Facebook AI Similarity Search)**  
   - **Description**: FAISS is a fast, open-source library developed by Facebook AI Research for efficient similarity search and clustering of high-dimensional vectors. It is commonly used for tasks like document retrieval and approximate nearest neighbor (ANN) search.
   - **Use Case**: Ideal for on-premise, lightweight deployments where you need fast, efficient similarity search of vectors.

2️⃣ **Pinecone**  
   - **Description**: Pinecone is a managed vector database that provides scalable, high-performance vector search capabilities. It supports automatic scaling and can handle millions of vectors, making it suitable for large-scale applications.
   - **Use Case**: Best for cloud-based applications where you need to scale vector search operations effortlessly, with features like automatic indexing and management.

3️⃣ **Chroma**  
   - **Description**: Chroma is a fast, open-source vector store that supports both disk-based and in-memory storage. It is designed to be simple to use and lightweight, making it an ideal choice for developers looking for an easy-to-integrate vector store.
   - **Use Case**: Great for smaller or mid-sized applications where ease of use and fast setup are key considerations.

4️⃣ **Weaviate**  
   - **Description**: Weaviate is an open-source vector search engine that supports both semantic search and hybrid search. It also integrates seamlessly with machine learning models to create a knowledge graph with vector-based queries.
   - **Use Case**: Ideal for applications that require hybrid search capabilities (combining both traditional keyword-based search and semantic vector search), as well as large-scale knowledge graphs.

5️⃣ **Qdrant**  
   - **Description**: Qdrant is an open-source vector search engine optimized for high-dimensional vector search. It features advanced filtering and search functionalities to support complex use cases, such as recommendation systems and personalized search.
   - **Use Case**: Well-suited for applications where you need to filter vectors based on multiple criteria in addition to vector similarity.


#### Key Features of Vector Stores in LangChain

- **Efficient Similarity Search**: Vector stores allow you to quickly search for vectors that are similar to a query vector, making them a powerful tool for semantic search and information retrieval.
- **Scalability**: Many vector stores (like Pinecone and Weaviate) are designed to scale seamlessly, handling millions or even billions of vectors.
- **Integration with LangChain Pipelines**: Vector stores integrate easily with LangChain’s pipelines, enabling complex workflows like Retrieval-augmented Generation (RAG), semantic search, and document classification.
- **Support for Various Embeddings**: LangChain’s vector stores can store and query embeddings from various models, including OpenAI, Hugging Face, or custom models.

---

### 🔍 Retrievers

Retrievers in LangChain are used to fetch relevant documents or information from a knowledge base or a vector store. They work by querying the data (e.g., vector stores, documents, or external APIs) and retrieving the most relevant pieces of information based on a user's query. The retrieved information is typically then passed to a model or another component for further processing, such as in Retrieval-Augmented Generation (RAG).

Retrievers are crucial in enabling applications that involve semantic search, question-answering, or any task where context from an external knowledge base is required. LangChain supports various types of retrievers, each designed for specific use cases and data sources.

#### Types of Retrievers

LangChain offers a variety of retrievers, each suited for different data sources and retrieval needs. Below are some of the most common types of retrievers available:

1️⃣ **Wikipedia Retriever**  
   - **Description**: The Wikipedia Retriever is designed to fetch relevant documents or information directly from Wikipedia. It can query the Wikipedia API to retrieve articles based on a given query, making it useful for answering general knowledge questions.
   - **Use Case**: Ideal for applications that require general knowledge and facts from Wikipedia.

2️⃣ **Vector Store Retriever**  
   - **Description**: A Vector Store Retriever performs similarity search on a vector store (e.g., FAISS, Pinecone, Chroma). It retrieves the most relevant documents or vectors from a vector store by comparing the similarity of the query vector with stored vectors.
   - **Use Case**: Useful when your data is stored as vector embeddings, and you need to find semantically similar documents based on vector search.

3️⃣ **MMR (Maximum Marginal Relevance) Retriever**  
   - **Description**: The MMR Retriever aims to find documents that are both relevant to the query and diverse from each other. It selects the top documents based on both relevance and the diversity of the returned results, which helps to reduce redundancy in the results.
   - **Use Case**: Best for applications where you want diverse yet relevant results, such as when you need to avoid returning similar or repetitive documents.

4️⃣ **Multi-Query Retriever**  
   - **Description**: The Multi-Query Retriever can perform multiple queries in parallel. It is designed to send multiple queries to the same or different retrievers and aggregate their results. This is useful when you want to query different sources or combine different retrieval techniques.
   - **Use Case**: Ideal for complex systems where you need to query multiple sources or multiple retrieval strategies simultaneously, such as cross-referencing results or combining information from different retrieval methods.

5️⃣ **Contextual Compression Retriever**  
   - **Description**: The Contextual Compression Retriever improves retrieval results by compressing and rephrasing the context of a query. It modifies the query in a way that preserves its meaning while potentially improving the retrieval of relevant documents, especially when the original query is vague or ambiguous.
   - **Use Case**: Useful when working with vague or complex queries where the exact intent might not be clear, and a refined query could improve retrieval accuracy.

#### How Retrievers Work

Each retriever in LangChain operates by executing a query against a data source (such as a vector store, Wikipedia, or multiple retrievers) and returning a list of relevant documents or results. These results are then typically passed to a downstream model or used in a pipeline for tasks like summarization, question-answering, or document ranking.

The retriever typically outputs a list of documents, each containing metadata (such as the source, relevance score, or position in the result set), which can then be processed for downstream tasks.

```python
{
  "page_content": "...relevant document content...",
  "metadata": {
      "source": "source_name",
      "score": 0.95,
      "relevance": "high"
  }
}
```

### 🧠 Retrieval Augmented Generation (RAG)

#### What is RAG?

**Retrieval Augmented Generation (RAG)** is an architecture where a Large Language Model (LLM) is combined with an external knowledge source (usually a vector database).  
Instead of relying only on the LLM’s internal knowledge, RAG retrieves relevant contextual information from documents and feeds it to the model at query time.

RAG = **Retriever + LLM**

This allows the LLM to generate answers grounded in **real, external, updated information**.

#### Why is RAG Used?

RAG solves several real-world problems:

1. **Overcomes LLM Knowledge Cutoff** -   LLMs cannot know everything after their last training date.RAG gives them **live**, **custom**, and **domain-specific** knowledge.

2. **Reduces Hallucinations** - Since the LLM answers only from retrieved context, responses become more accurate and explainable.

3. **Personal & Private Data Usage** You can use:
    - private company documents  
    - internal knowledge bases  
    - proprietary research  without ever training or fine-tuning an LLM.

4. **Scalable & Flexible** - You can update the vector database anytime with new documents, and the LLM will immediately use them.


#### Detailed Steps Involved in RAG

RAG generally follows this pipeline:


#### **1. Document Loading**
Raw documents (PDFs, text files, transcripts, URLs, YouTube content, etc.) are loaded using:
- LangChain Loaders  
- Custom scrapers  
- APIs  

Example:  
Loading YouTube transcript → storing text.


#### **2. Text Splitting**
Documents are often too large for embeddings.  
They are chunked into smaller segments using something like:

- `RecursiveCharacterTextSplitter`
- `TokenTextSplitter`

This ensures:
- better retrieval accuracy  
- manageable embedding cost  
- coherent context windows  


#### **3. Embedding Generation**
Each chunk is converted into a vector using an embedding model such as:

- `text-embedding-3-small`
- `text-embedding-3-large`

These embeddings represent semantic meaning.


#### **4. Vector Store Storage**
Chunks + embeddings are stored in a vector database, such as:

- **FAISS**
- ChromaDB
- Pinecone

This enables fast similarity search.


#### **5. Retrieval**
At query time, the user question is embedded.  
The vector store retrieves the most relevant documents using similarity search.

Example techniques:
- `k-NN search`
- Cosine similarity
- Dot-product search


#### **6. Prompt Construction**
The retrieved documents are inserted into a structured prompt template:
```python
"""You are a helpful assistant.
Answer only from the provided context.
If the context is insufficient, say "I don't know".
Context:
{context}
Question:
{query}"""
```
This constrains the model to grounded knowledge.

#### **7. LLM Generation**
The prompt is passed into an LLM such as:
- `gpt-4o`
- `gpt-4.1`
- `gpt-4-turbo`

The model uses the retrieved context to generate an accurate, grounded answer.


#### **8. Answer Returned to User**
The final answer is context-enriched, fact-based, and non-hallucinated.


RAG makes LLMs smarter, grounded, and capable of using **your own data** reliably.


### 🛠️ Tools Component
LangChain **Tools** allow language models to interact with external systems by executing functions, APIs, database queries, code, and more.  
They bridge the gap between **natural language** and **actions**, enabling agents to perform meaningful tasks in the real world.


#### 🔍 What Are Tools in LangChain?

Tools are **functions wrapped with metadata** (name, description, schema) that help LLMs understand:

- *When* to use the tool  
- *How* to use the tool  
- *What input* it expects  
- *What output* it produces  

Tools allow LLMs to:  
- Make calculations  
- Query APIs  
- Search documents  
- Access databases  
- Run code  
- Trigger workflows  
- And more...


#### 🧰 Why Are Tools Important?

LLMs operate on text only.  
Tools give them **capabilities beyond text**, enabling:

- **Enhanced reasoning**  
- **Improved accuracy** (e.g., math, retrieval)  
- **Autonomous agents**  
- **Function calling** like OpenAI/Anthropic models  
- **Actionable workflows**  

Tools are the backbone of **agents**, **function calling**, and **complex pipelines**.

#### 🛠️ Creating Tools in LangChain

LangChain provides multiple ways to define tools.  
Below are the most common and most powerful methods.


#### 1. 🏷️ Creating Tools Using the `@tool` Decorator

This is the simplest way to create a tool.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
```

##### ✔ Features
- Auto-generates metadata
- Automatically infers schema from type hints
- Very convenient for quick utilities

#### 2. 🧱 Creating Tools Using StructuredTool
This method is used when your tool needs:
- Strong input validation
- Pydantic schemas
- Clear argument structures
- Detailed documentation for LLMs

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)
```

##### ✔ Why use StructuredTool?
- Strict control over inputs
- Better error handling
- Compatible with complex applications
- Recommended for production-grade tools

#### 3. ⚙️ Creating Custom Tools Using BaseTool
Use this when you need full control over:
- Custom logic
- Validation
- Setup/teardown
- Async behavior

```python
from langchain_core.tools import BaseTool

class PowerTool(BaseTool):
    name = "power"
    description = "Raises a number to a power."

    def _run(self, base: int, exponent: int):
        return base ** exponent

    async def _arun(self, base: int, exponent: int):
        return base ** exponent
```

##### ✔ When to use BaseTool?
- Custom behavior
- Tools requiring initialization
- Complex async tools
- Wrapper over external services

#### 🧰 Toolkits in LangChain

A **Toolkit** is a pre-built collection of tools designed for a specific domain.

#### Examples include:

| Toolkit | Purpose |
|--------|----------|
| **SQL Toolkit** | Query SQL databases with tools like `QuerySQLDataBaseTool` |
| **Browser Toolkit** | Tools for web navigation and scraping |
| **File Toolkit** | Read/write local file content |
| **VectorStore Toolkit** | Search and retrieve from vector DBs |

---

##### **Why Toolkits?**

- Ready-made bundles  
- Reduce setup time  
- Build powerful agents quickly  

---

## 🧰 Technologies Used 
- **Python 3.11** 
- **LangChain** 
- **OpenAI API** 
- **Hugging Face Transformers** 
- **Google API** 
- **Anthropic API** 
- **Huggingface Token** 
- **scikit-learn** *(for cosine similarity)*


----


## 💡 Contribution 
Contributions are always welcome! 🙌 If you would like to improve this repository, fix issues, or add new LangChain examples: 
1. **Fork** the repository 
2. **Create a new branch** for your feature or fix
  ```bash
   git checkout -b feature-name
   ```
3. **Commit your changes:**
  ```bash
    git commit -m "Add detailed explanation for retrievers module"
  ```
4. **Push to your fork:**
  ```bash
    git push origin feature-name
  ```
5. **Open a Pull Request** — describe what you’ve done and why it improves the repo