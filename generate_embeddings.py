import os
import shutil
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "./report.pdf"
PERSIST_DIR = "./chroma_db_nccn"

if "--rebuild" in sys.argv and os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Database already exists. Use --rebuild to recreate it.")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print("Creating new vector database...")
    docs = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )

print(f"Total vectors in store: {vectorstore._collection.count()}")