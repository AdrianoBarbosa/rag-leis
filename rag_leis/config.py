"""Configuração central do projeto. Tudo que é "botão de ajuste" do RAG fica aqui."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # markdown baixado do repo ctresb/leis
STORAGE_DIR = DATA_DIR / "storage"  # índices persistidos
QDRANT_DIR = STORAGE_DIR / "qdrant"
BM25_DIR = STORAGE_DIR / "bm25"
NODES_FILE = STORAGE_DIR / "nodes.jsonl"  # chunks gerados, úteis para inspecionar

REPO = "ctresb/leis"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/main"

# Embeddings: bge-m3 é multilíngue (vai bem em português) e roda local no Ollama.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
COLLECTION = "leis"

# Chunking: artigos maiores que isso são quebrados em pedaços com sobreposição.
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64

DOMINIOS = ["trabalhista", "tributario", "inventario"]
