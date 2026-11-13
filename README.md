# Enterprise Financial RAG Assistant

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline built in Python that allows users to query multi-year corporate financial and operational reports using semantic search and Google's Gemini generative models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Modular-orange)
![Google GenAI](https://img.shields.io/badge/Google-GenAI-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-yellow)

---

## Features

- **Local Embeddings:** Utilizes Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) running on CPU to vectorize documents securely and efficiently without external API costs.
- **Persistent Vector Store:** Leverages `ChromaDB` to store and load pre-computed embeddings on disk, avoiding redundant re-indexing on every execution.
- **Optimized Text Chunking:** Uses `RecursiveCharacterTextSplitter` with tailored chunk sizes and overlaps to preserve financial table and narrative context boundaries.
- **Google GenAI Integration:** Powered by Google's modern `google-genai` SDK and `gemini-3.5-flash-lite` for fast, accurate conversational synthesis.
- **Grounded Interactive CLI:** A clean command-line interface with strict system instructions ensuring accurate, conversational answers complete with page-level citations.

---

## Project Structure
```text
financial-rag-assistant/
│
├── venv/                       # Source code and dataset
│   ├── generate_embeddings.py  # Vectorizes the PDF and builds the Chroma database
│   ├── rag.py                  # Interactive CLI assistant script
│   └── report.pdf              # Sample multi-year corporate financial report
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

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
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
```
# On Windows:
```bash
venv\Scripts\activate
```

# On macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
```text
-Press the Windows Key, type Environment Variables, and select Edit the system environment variables.
-Click the Environment Variables... button at the bottom right.
-Under your user variables, click New...
-Variable name: GEMINI_API_KEY
-Variable value: your_api_key_name
-Click OK
```

## How To Run
-Navigate into the venv directory before executing the scripts so relative file paths resolve properly with
```bash
cd venv
```

### Step 1: Generate the Vector Database
Parse report.pdf, split the text into chunks, compute embeddings, and build the persistent Chroma store:
```bash
python generate_embeddings.py
```

### Step 2: Query the Financial Assistant
Start the interactive CLI loop to ask questions about the annual report:
```bash
python rag.py
```
