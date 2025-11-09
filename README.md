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


## ⚙️ Installation & Setup 1. **Clone this repository:**
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