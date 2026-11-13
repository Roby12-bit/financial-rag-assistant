# Enterprise Financial RAG Assistant

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline built in Python that allows users to query multi-year corporate financial and operational reports using semantic search and Google's Gemini generative models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Modular-orange)
![Google GenAI](https://img.shields.io/badge/Google-GenAI-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-yellow)

---

## Features

- **Local Embeddings:** Utilizes Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) running on CPU to vectorize documents securely and efficiently without external API cost.
- **Persistent Vector Store:** Leverages `ChromaDB` to store and load pre-computed embeddings, avoiding unnecessary re-indexing on every script execution.
- **Advanced Text Splitting:** Uses `RecursiveCharacterTextSplitter` with optimized chunk sizes and overlaps for precise context boundary handling.
- **Google GenAI Integration:** Powered by Google's latest `google-genai` SDK and `gemini-3.8-flash` for fast, accurate conversational synthesis.
- **Clean Interactive CLI:** A user-friendly command-line loop with system instructions ensuring professional, conversational, and grounded responses complete with page-level citations.

---

## Tech Stack

- **Orchestration:** LangChain (`langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`)
- **LLM:** Google Gemini (`gemini-3.8-flash`)
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers (Hugging Face)
- **Document Loading:** PyPDFLoader

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Roby12-bit/financial-rag-assistant.git](https://github.com/Roby12-bit/financial-rag-assistant.git)
cd financial-rag-assistant
