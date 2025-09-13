# Django RAG Chat (Minimal)

This is a **minimal** Django + DRF project that demonstrates a Retrieval-Augmented-Generation (RAG)
workflow with **document upload** and a **chat endpoint** — **no OpenAI API key required**.

The project uses:
- `sentence-transformers` for embeddings (`all-MiniLM-L6-v2` by default)
- `chromadb` for local vector storage (DuckDB+Parquet persistence)
- `transformers` (text2text generation, default `google/flan-t5-small`) for answer generation
- `pypdf` / `python-docx` for extracting text from uploaded documents

> ⚠️ Note: This scaffold **does not include any model files**. The first run will download models from Hugging Face.

## Quick start (local)
1. Create & activate a Python venv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows (PowerShell)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run migrations & create superuser (optional):
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```
4. Start the Django dev server:
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```
5. Open `http://127.0.0.1:8000/` in your browser. Upload a document and ask questions.

## Environment variables (optional)
- `MODEL_NAME` — Name of the Hugging Face model to use for generation (default: google/flan-t5-small)
- `EMBEDDING_MODEL` — Name of the sentence-transformers embedding model (default: all-MiniLM-L6-v2)

## Hosting publicly (free / demo)
- This app can be deployed to **Render / Railway / Fly.io** as a small demo. Be aware:
  - Model downloads may exceed platform network allowances/timeouts.
  - CPU-only generation will be slow for larger models.
  - For a production-ready system you typically use a dedicated model-serving service or smaller hosted model endpoints.

## Files included
- `rag_django/` - Django project
- `chatapp/` - Django app (views, urls, utils)
- `templates/index.html` - simple frontend (chat + upload)
- `static/js/app.js` - frontend JS

## Support
If you want, I can:
- Help adapt this to use `llama-cpp-python` / Ollama (for local GGUF models)
- Provide a Render/Railway deployment config (Dockerfile / service file)
