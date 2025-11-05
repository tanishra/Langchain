# LangChain

## 📚 Table of Contents

- [About LangChain](#about-langchain)
- [Goals of This Repository](#goals-of-this-repository)
- [Current Progress](#current-progress)
  - [1️⃣ Models Component](#1️⃣-models-component)
    - [Language Models](#language-models)
    - [Embedding Models](#embedding-models)
  - [🧩 Semantic Search Example](#-semantic-search-example)
- [Upcoming Topics](#upcoming-topics)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [How to Use](#how-to-use)
- [Technologies Used](#technologies-used)
- [Learning Resources](#learning-resources)
- [Disclaimer](#disclaimer)

---

## 🧠 About LangChain

**LangChain** is a framework designed to simplify the development of applications powered by **language models** (like OpenAI’s GPT, Anthropic’s Claude, or open-source models like LLaMA and Falcon).  
It provides modular components to help developers connect **LLMs, prompts, tools, memory, agents, retrievers, and more** into structured, production-ready pipelines.

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

## 🔜 Upcoming Topics

The repository will expand to include more LangChain components as I continue learning:

| Component | Description |
|------------|-------------|
| **Prompts** | Designing and managing prompt templates effectively. |
| **Chains** | Combining multiple components to form LLM pipelines. |
| **Memory** | Adding conversation history to make chatbots contextual. |
| **Output Parsers** | Structuring and validating model outputs. |
| **Runnables** | Executing modular and composable LLM workflows. |
| **Retrievers** | Fetching relevant data from vector stores or databases. |
| **Agents** | Enabling LLMs to reason and use external tools dynamically. |
| **Mini Projects** | Building small, complete applications integrating multiple components. |

---

## ⚙️ Installation & Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/tanishra/Langchain.git
   cd Langchain
   ```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Add your API keys to .env file:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
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


---

## 💡 Contribution

Contributions are always welcome! 🙌  
If you would like to improve this repository, fix issues, or add new LangChain examples:

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



