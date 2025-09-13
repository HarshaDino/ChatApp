import os, uuid, json
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from .models import Document
from .serializers import DocumentSerializer
from . import utils

MEDIA_ROOT = settings.MEDIA_ROOT

def index(request):
    return render(request, 'index.html', {})

class UploadView(APIView):
    def post(self, request, format=None):
        f = request.FILES.get('file')
        if not f:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        doc = Document.objects.create(file=f, original_name=getattr(f, 'name', None))
        # extract text and index
        doc_path = doc.file.path
        text = utils.extract_text_from_file(doc_path)
        chunks = utils.chunk_text(text)
        if chunks:
            embedder = utils.get_embedding_model()
            embeddings = utils.embed_texts(chunks, embedder=embedder)
            client, coll = utils.get_chroma_client_and_collection()
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{'doc_id': doc.id, 'source': doc.original_name or doc.file.name, 'chunk_index': i} for i in range(len(chunks))]
            coll.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
            client.persist()
        serializer = DocumentSerializer(doc)
        return Response(serializer.data)

class ChatView(APIView):
    def post(self, request, format=None):
        question = request.data.get('question')
        if not question:
            return Response({'error': 'Missing question'}, status=status.HTTP_400_BAD_REQUEST)
        # embed question
        embedder = utils.get_embedding_model()
        q_emb = embedder.encode([question], show_progress_bar=False, convert_to_numpy=True).tolist()[0]
        client, coll = utils.get_chroma_client_and_collection()
        try:
            result = coll.query(query_embeddings=[q_emb], n_results=3)
            # result contains 'documents' key typically
            docs = []
            # chroma may return nested lists
            for docs_list in result.get('documents', []):
                docs.extend(docs_list)
            for metas_list in result.get('metadatas', []):
                metas = metas_list
            # build context
            context = ''
            if docs:
                # join top docs
                context = '\n\n---\n\n'.join(docs[:3])
            # generate answer using a local transformers model (text2text)
            answer = generate_answer(question, context)
            return Response({'answer': answer, 'context': context, 'sources': metas[:3] if 'metas' in locals() else []})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def generate_answer(question: str, context: str) -> str:
    # Use a small instruction-following model via transformers (text2text)
    model_name = os.environ.get('MODEL_NAME', 'sshleifer/tiny-gpt2')
    prompt = f"Use the following context to answer the question. If the answer is not contained in the context, reply with: 'I don't know — answer not found in uploaded documents.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        # load pipeline lazily
        pipe = pipeline('text2text-generation', model=model_name, tokenizer=model_name, device_map='auto' if False else None)
        out = pipe(prompt, max_new_tokens=256, do_sample=False)
        if isinstance(out, list):
            return out[0].get('generated_text', '').strip()
        return str(out).strip()
    except Exception as e:
        # fallback simple reply if transformers are not available or model fails
        return "Sorry — model unavailable. Please check server logs or install the required models.\n" + str(e)
