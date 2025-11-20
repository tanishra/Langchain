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