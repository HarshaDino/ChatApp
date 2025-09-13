import os
import uuid
from pathlib import Path

MEDIA_ROOT = Path(__file__).resolve().parent.parent / 'media'
CHROMA_DIR = Path(__file__).resolve().parent.parent / 'chroma_db'
os.makedirs(CHROMA_DIR, exist_ok=True)

def extract_text_from_file(filepath: str) -> str:
    filepath = str(filepath)
    lower = filepath.lower()
    text = ''
    if lower.endswith('.pdf'):
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            pages = [p.extract_text() or '' for p in reader.pages]
            text = '\n'.join(pages)
        except Exception as e:
            text = ''
    elif lower.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(filepath)
            paragraphs = [p.text for p in doc.paragraphs]
            text = '\n'.join(paragraphs)
        except Exception as e:
            text = ''
    else:
        # fallback for txt or others
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception:
            text = ''
    return text or ''

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# Embedding & vector store helpers (lazy imports)
def get_embedding_model(model_name: str = None):
    model_name = model_name or os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(model_name)
        return emb
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

def get_chroma_client_and_collection(collection_name: str = 'documents'):
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(CHROMA_DIR)))
        # create or get collection
        coll = None
        try:
            coll = client.get_collection(name=collection_name)
        except Exception:
            coll = client.create_collection(name=collection_name)
        return client, coll
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

def embed_texts(texts, embedder=None):
    # returns list of embeddings (list of floats)
    if embedder is None:
        embedder = get_embedding_model()
    return embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
