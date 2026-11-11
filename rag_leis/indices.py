"""Os dois índices do projeto e como abri-los.

- Vetorial (Qdrant + bge-m3): acha trechos pelo SIGNIFICADO. "mandar embora sem motivo"
  encontra "dispensa sem justa causa".
- Lexical (BM25): acha trechos pelas PALAVRAS. Bom para termos exatos ("FGTS",
  "art. 1.829", "Simples Nacional") que embeddings às vezes diluem.

Usar os dois e fundir os resultados é a tal busca híbrida.

Obs.: o Qdrant em modo local (sem servidor) trava a pasta para um processo só. Se o
Streamlit estiver aberto, feche-o antes de rodar ingest/avaliar/perguntar.
"""

from dataclasses import dataclass

import Stemmer
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .config import BM25_DIR, COLLECTION, EMBED_MODEL, OLLAMA_URL, QDRANT_DIR

# Sem isso o LlamaIndex tenta usar OpenAI como padrão em qualquer lugar que
# não receber um LLM explícito.
Settings.llm = None


def embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_URL, embed_batch_size=32)


def stemmer() -> Stemmer.Stemmer:
    # "demitido", "demissão", "demitir" -> mesmo radical, então o BM25 casa as três.
    return Stemmer.Stemmer("portuguese")


def qdrant_store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(collection_name=COLLECTION, client=client)


@dataclass
class Indices:
    vetorial: VectorStoreIndex
    bm25: BM25Retriever
    client: QdrantClient


def abrir_indices() -> Indices:
    if not QDRANT_DIR.exists() or not BM25_DIR.exists():
        raise FileNotFoundError("Índices não encontrados. Rode: uv run python -m rag_leis.ingest")
    client = QdrantClient(path=str(QDRANT_DIR))
    vetorial = VectorStoreIndex.from_vector_store(qdrant_store(client), embed_model=embed_model())
    bm25 = BM25Retriever.from_persist_dir(str(BM25_DIR))
    bm25.stemmer = stemmer()  # o stemmer não é salvo junto com o índice
    return Indices(vetorial, bm25, client)
