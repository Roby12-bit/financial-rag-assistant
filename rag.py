import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types

PERSIST_DIR = "./chroma_db_nccn"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    sys.exit("Set your key first, e.g.:  export GEMINI_API_KEY='your-key'")
if not os.path.isdir(PERSIST_DIR):
    sys.exit("Database not found. Run generate_embeddings.py first.")

# Load everything once, not on every question
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You answer questions about a document using only the provided context. "
    "Be friendly and conversational, explain complex terms simply, and answer "
    "in complete sentences. Mention page numbers when useful. If the context "
    "does not contain the answer, say you couldn't find it in the document "
    "instead of guessing."
)

def get_context(query, k=6):
    results = vector_db.similarity_search(query, k=k)
    return "\n\n".join(
        f"[Page {r.metadata.get('page', 0) + 1}] {r.page_content}" for r in results
    )

def generate_answer(query, context):
    prompt = f"QUESTION: {query}\n\nCONTEXT:\n{context}\n\nANSWER:"
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return response.text
    except Exception as e:
        return f"Sorry, the API call failed: {e}"

print("Ask questions about your PDF. Type 'exit' to quit.")
while True:
    try:
        query = input("\nQuery: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting gracefully...")
        break
    if not query:
        continue
    if query.lower() in {"exit", "quit"}:
        break
    print(generate_answer(query, get_context(query)))