"""RAG pipeline: chunking, embedding, ingestion, querying."""

import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq

from config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
)
from endee_client import EndeeClient


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file object or path."""
    reader = PdfReader(pdf_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest(
    text: str,
    doc_name: str,
    model: SentenceTransformer,
    client: EndeeClient,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Chunk → embed → store. Returns number of chunks stored."""
    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return 0

    embeddings = model.encode(chunks, normalize_embeddings=True)
    seed = f"{doc_name}-{int(time.time())}"
    ids = [f"{seed}-{i}" for i in range(len(chunks))]
    metas = [{"doc": doc_name, "text": c} for c in chunks]

    ok = client.insert_vectors(ids, embeddings, metas)
    return len(chunks) if ok else 0


def search(
    question: str,
    model: SentenceTransformer,
    client: EndeeClient,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    vec = model.encode([question], normalize_embeddings=True)[0]
    return client.search(vec, top_k=top_k)


def generate_answer(
    question: str,
    sources: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is not set. Add it to your `.env` file."

    context = "\n\n".join(src["text"] for src in sources) if sources else ""
    if not context:
        return "No relevant documents found. Please upload and ingest a document first."

    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        "You are a helpful assistant. Use ONLY the context below to answer.\n"
        "If the context does not contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=600,
    )
    return resp.choices[0].message.content