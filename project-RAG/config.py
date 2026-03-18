"""Shared configuration for the RAG pipeline."""

import os
from dotenv import load_dotenv

load_dotenv()


def resolve_endee_url() -> str:
    explicit = os.getenv("ENDEE_URL")
    if explicit:
        return explicit.rstrip("/")
    hostport = os.getenv("ENDEE_HOSTPORT")
    if hostport:
        if hostport.startswith(("http://", "https://")):
            return hostport.rstrip("/")
        return f"http://{hostport}".rstrip("/")
    return "http://localhost:8080"


ENDEE_URL = resolve_endee_url()
INDEX_NAME = os.getenv("INDEX_NAME", "rag_documents")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 500
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.3