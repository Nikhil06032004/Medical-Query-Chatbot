<div align="center">

# 🏥 Healthcare RAG Assistant

### AI-Powered Medical Information Retrieval System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green?style=for-the-badge)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-orange?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-purple?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **production-grade Retrieval-Augmented Generation (RAG)** healthcare assistant that provides accurate, contextual, and explainable medical information by combining Large Language Models with domain-specific medical documents.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Usage](#-usage) · [Project Structure](#-project-structure) · [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Usage](#-usage)
  - [Web Interface (Streamlit)](#web-interface-streamlit)
  - [CLI Interface](#cli-interface)
  - [Ingesting Your Own Documents](#ingesting-your-own-documents)
- [Module Reference](#-module-reference)
- [Key Design Decisions](#-key-design-decisions)
- [Known Limitations & Roadmap](#-known-limitations--roadmap)
- [Contributing](#-contributing)
- [Medical Disclaimer](#️-medical-disclaimer)
- [License](#-license)

---

## 🔍 Overview

The **Healthcare RAG Assistant** addresses a critical gap in accessible medical information: users often struggle to get accurate, source-backed answers to healthcare questions from generic AI tools. This system solves that by grounding every response in **your own curated medical document corpus**, preventing hallucination and enabling transparent citation of sources.

The system supports two usage modes:
- A **Streamlit web UI** for end-users and demonstrations
- A **CLI interface** for developers and automated pipelines

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Context-Aware Q&A** | Answers grounded in retrieved medical documents via RAG |
| 📄 **PDF Ingestion** | Automatically loads and indexes PDF medical literature |
| 🧠 **Semantic Search** | FAISS vector store for fast, accurate similarity retrieval |
| ⚡ **Ultra-Fast Inference** | Groq-hosted LLaMA 3.1 8B for sub-second response times |
| 🖥️ **Dual Interface** | Professional Streamlit web UI + interactive CLI |
| 📚 **Source Transparency** | Every answer includes collapsible source document references |
| 🧾 **Disclaimer Enforcement** | Built-in medical disclaimer on every response |
| 📊 **Session Statistics** | Live query count, document count, and model settings in sidebar |
| 📥 **Chat Export** | Download full conversation history as a `.txt` file |
| 🔑 **Flexible API Key Management** | Supports `.env` file or runtime UI input |
| 🗂️ **Structured Logging** | File + console logging with timestamps across all modules |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                      │
│          Streamlit Web UI          CLI (terminal)           │
└─────────────────────┬───────────────────────────────────────┘
                      │  User Question
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                           │
│                                                             │
│   ┌──────────────┐     ┌────────────────────────────────┐  │
│   │   Retriever  │────▶│  ChatPromptTemplate            │  │
│   │  (FAISS k=5) │     │  (System + Human messages)     │  │
│   └──────┬───────┘     └───────────────┬────────────────┘  │
│          │                             │                    │
│          │  Top-K Chunks               │  Prompt + Context  │
│          ▼                             ▼                    │
│   ┌──────────────┐     ┌────────────────────────────────┐  │
│   │  FAISS Index │     │  ChatGroq (LLaMA 3.1-8B)       │  │
│   │  (db_faiss)  │     │  Temp: 0.7 | Tokens: 2048      │  │
│   └──────────────┘     └───────────────┬────────────────┘  │
│                                        │                    │
│                                        │  StrOutputParser   │
└────────────────────────────────────────┼────────────────────┘
                                         ▼
                                  Structured Answer
                                 + Source References

┌─────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                        │
│                                                             │
│  PDF Files ──▶ DirectoryLoader ──▶ RecursiveTextSplitter   │
│                  (PyPDFLoader)       chunk=500, overlap=50   │
│                                           │                  │
│                                           ▼                  │
│                              HuggingFaceEmbeddings           │
│                           (all-MiniLM-L6-v2, 384-dim)       │
│                                           │                  │
│                                           ▼                  │
│                              FAISS VectorStore.save_local    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **LLM** | Groq / LLaMA 3.1 | 8B Instant | Response generation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 5.0.0 | Semantic text encoding |
| **Vector Store** | FAISS (CPU) | 1.11.0 | Similarity search |
| **Orchestration** | LangChain | 0.3.26 | RAG pipeline management |
| **Web UI** | Streamlit | 1.46.1 | Interactive frontend |
| **PDF Loader** | PyPDF + LangChain DirectoryLoader | 5.7.0 | Document ingestion |
| **Text Splitting** | RecursiveCharacterTextSplitter | — | Chunking strategy |
| **Environment** | python-dotenv | 1.1.1 | Secret management |
| **Deep Learning** | PyTorch | 2.7.1 | Embedding model backend |
| **Package Manager** | `uv` (with `pyproject.toml`) | — | Fast dependency resolution |

---

## 📁 Project Structure

```
Medical-Query-Chatbot/
│
├── 📂 Data/                          # Place your medical PDF files here
│   └── *.pdf                         # Input documents for ingestion
│
├── 📂 vectorstore/
│   └── db_faiss/                     # Pre-built FAISS index (binary)
│       ├── index.faiss               # Vector index
│       └── index.pkl                 # Metadata & docstore
│
├── 📂 logs/                          # Application log files
│   ├── rag_assistant.log             # CLI interface logs
│   └── streamlit_rag.log             # Web UI logs
│
├── 📄 memory_for_LLM.py              # Document ingestion & FAISS builder
├── 📄 Connect_memory_with_LLM.py     # CLI RAG interface & HealthcareRAGAssistant class
├── 📄 Medical_chatbot.py             # Streamlit web application (main UI)
├── 📄 main.py                        # Package entry point stub
│
├── 📄 requirements.txt               # Pinned production dependencies (uv-generated)
├── 📄 pyproject.toml                 # Project metadata & build config
├── 📄 uv.lock                        # Reproducible lock file
├── 📄 .python-version                # Pinned Python version
├── 📄 .gitignore                     # Git exclusions
└── 📄 README.md                      # This file
```

---

## ✅ Prerequisites

- **Python** 3.9 or higher (see `.python-version`)
- A free **Groq API key** — obtain one at [console.groq.com](https://console.groq.com)
- `uv` package manager (recommended) or `pip`
- At least one medical PDF in the `Data/` directory before running ingestion

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Nikhil06032004/Medical-Query-Chatbot.git
cd Medical-Query-Chatbot
```

### 2. Create & Activate a Virtual Environment

Using `uv` (recommended — significantly faster):
```bash
pip install uv
uv venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

Or with standard `venv`:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# With uv (uses uv.lock for reproducible installs)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

> ⚠️ **Note:** The full install includes PyTorch (~2 GB). This is required for the embedding model. Expect a few minutes on first install.

### 4. Set Up Your API Key

Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

Alternatively, you can enter the key directly in the Streamlit sidebar at runtime.

### 5. Add Medical Documents

Place your PDF files in the `Data/` directory:

```bash
cp /path/to/your/medical_books/*.pdf Data/
```

### 6. Build the Vector Store

```bash
python memory_for_LLM.py
```

Expected output:
```
Loaded 342 PDF pages
Created 1,204 text chunks
FAISS vector store saved successfully
```

### 7. Launch the Application

**Streamlit Web UI:**
```bash
streamlit run Medical_chatbot.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

**CLI Interface:**
```bash
python Connect_memory_with_LLM.py
```

---

## ⚙️ Configuration

All tuneable parameters are centralized in the `Config` class present in both `Medical_chatbot.py` and `Connect_memory_with_LLM.py`:

```python
class Config:
    MODEL_NAME      = "llama-3.1-8b-instant"   # Groq model identifier
    TEMPERATURE     = 0.7                        # Response creativity (0=deterministic, 1=creative)
    MAX_TOKENS      = 2048                       # Maximum response length
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DB_FAISS_PATH   = "vectorstore/db_faiss"    # Vector store location
    TOP_K_RESULTS   = 5                          # Chunks retrieved per query
    LOG_FILE        = "logs/streamlit_rag.log"  # Log output path
```

To switch models, update `MODEL_NAME` to any Groq-supported identifier (e.g., `"llama-3.1-70b-versatile"`).

For the ingestion pipeline, chunk parameters are set in `memory_for_LLM.py`:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between adjacent chunks
)
```

---

## 💡 Usage

### Web Interface (Streamlit)

1. Run `streamlit run Medical_chatbot.py`
2. Open [http://localhost:8501](http://localhost:8501)
3. Configure your Groq API key in the **Settings** sidebar (or use `.env`)
4. Type a question in the chat input at the bottom
5. View the answer with collapsible source references
6. Use **Export Chat** to download your conversation

**Example questions:**
- "What are the early warning signs of Type 2 diabetes?"
- "How does hypertension affect kidney function long-term?"
- "What are the recommended vaccination schedules for adults?"

### CLI Interface

```bash
python Connect_memory_with_LLM.py
```

Available commands during a session:

| Command | Action |
|---|---|
| Any text | Submit a healthcare question |
| `help` | Show usage instructions |
| `clear` | Clear the terminal screen |
| `exit` / `quit` / `q` | End the session |
| `Ctrl+C` | Force quit |

### Ingesting Your Own Documents

1. Place PDF files in the `Data/` directory
2. Run `python memory_for_LLM.py`
3. The script will:
   - Load all PDFs via `PyPDFLoader`
   - Split pages into 500-character chunks with 50-character overlap
   - Encode chunks using `all-MiniLM-L6-v2` (384-dimensional embeddings)
   - Save the FAISS index to `vectorstore/db_faiss/`
4. Restart the chatbot — it will automatically use the new index

---

## 📖 Module Reference

### `memory_for_LLM.py` — Ingestion Pipeline

| Function | Description |
|---|---|
| `load_pdf_files(data_path)` | Loads all PDFs from a directory using `DirectoryLoader` + `PyPDFLoader` |
| `create_chunks(documents)` | Splits documents using `RecursiveCharacterTextSplitter` |
| `get_embedding_model()` | Returns a `HuggingFaceEmbeddings` instance |
| *(script body)* | Orchestrates load → chunk → embed → save workflow |

### `Connect_memory_with_LLM.py` — CLI Interface

| Class / Function | Description |
|---|---|
| `Config` | Centralized configuration dataclass |
| `StreamingCallbackHandler` | LangChain callback that logs LLM start/end events |
| `HealthcareRAGAssistant.__init__` | Validates env, initializes LLM, retriever, and chain |
| `HealthcareRAGAssistant.query(question)` | Executes RAG query; returns structured `dict` with answer, sources, timestamp |
| `HealthcareRAGAssistant.format_response(result)` | Formats query result for terminal display with banner and disclaimer |
| `main()` | Interactive REPL loop with command handling |

### `Medical_chatbot.py` — Streamlit Web App

| Function | Description |
|---|---|
| `initialize_embeddings()` | `@st.cache_resource` — loads HuggingFace model once per session |
| `initialize_vectorstore(embeddings)` | `@st.cache_resource` — loads FAISS index once per session |
| `initialize_llm()` | Creates `ChatGroq` instance from env or session-state API key |
| `create_rag_chain(llm, retriever)` | Builds the full LangChain LCEL pipeline |
| `format_source_documents(docs)` | Renders HTML collapsible source cards |
| `render_sidebar(...)` | Sidebar with API config, system status, model info, and chat export |
| `main()` | App entry point — wires all components and handles chat loop |

---

## 🎯 Key Design Decisions

**Why FAISS over a hosted vector DB?** FAISS is entirely local — no external service, no latency, no data leaving the machine. This is important in healthcare contexts where document privacy matters.

**Why `sentence-transformers/all-MiniLM-L6-v2`?** It strikes a balance between embedding quality and performance — fast enough to run on CPU, with 384-dimensional vectors that capture semantic meaning well for medical text.

**Why Groq?** Groq's LPU hardware delivers dramatically faster inference than GPU-hosted APIs, making conversational latency feel near-instant even on complex prompts.

**Why `@st.cache_resource` on embeddings and vectorstore?** These are expensive to initialize (~5–15 seconds each). Caching them means only the first user session pays the startup cost; subsequent interactions and rerenders are immediate.

**Dual interface rationale:** `Medical_chatbot.py` (Streamlit) is for end-users; `Connect_memory_with_LLM.py` (CLI) is for developers testing prompts, integrating into pipelines, or running in headless server environments.

---



**Roadmap / Suggested Improvements:**

| Priority | Enhancement |
|---|---|
| 🔴 High | Add `ConversationBufferMemory` for multi-turn context |
| 🔴 High | Implement automatic index rebuild when new PDFs are added to `Data/` |
| 🟡 Medium | Add Docker support (`Dockerfile` + `docker-compose.yml`) |
| 🟡 Medium | Implement user authentication for the Streamlit app |
| 🟡 Medium | Add a REST API layer (FastAPI) alongside the existing interfaces |
| 🟢 Low | Support additional document formats (DOCX, TXT, HTML) |
| 🟢 Low | Add evaluation metrics (RAGAS or similar) |
| 🟢 Low | CI/CD pipeline with GitHub Actions |

---



---

## ⚠️ Medical Disclaimer

> **This application is for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the guidance of a qualified healthcare provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking it because of information provided by this system.

---



---

<div align="center">

Built with ❤️ by [Nikhil](https://github.com/Nikhil06032004) · Powered by LangChain, Groq & FAISS

</div>
