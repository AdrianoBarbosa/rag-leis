"""Indexação: markdown -> chunks -> embeddings no Qdrant + índice BM25.

Uso: uv run python -m rag_leis.ingest
"""

import shutil
import time

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient

from .config import BM25_DIR, NODES_FILE, QDRANT_DIR
from .indices import embed_model, qdrant_store, stemmer
from .parsing import carregar_nos, salvar_nos


def main() -> None:
    nos = carregar_nos()
    salvar_nos(nos)
    print(f"{len(nos)} chunks (inspecione em {NODES_FILE})")

    # Reindexação completa: mais simples de entender do que atualização incremental.
    shutil.rmtree(QDRANT_DIR, ignore_errors=True)
    shutil.rmtree(BM25_DIR, ignore_errors=True)

    inicio = time.perf_counter()
    client = QdrantClient(path=str(QDRANT_DIR))
    storage = StorageContext.from_defaults(vector_store=qdrant_store(client))
    # Aqui acontece o trabalho pesado: cada chunk passa pelo bge-m3 e vira um vetor
    # de 1024 dimensões, gravado no Qdrant junto com os metadados.
    VectorStoreIndex(nos, storage_context=storage, embed_model=embed_model(), show_progress=True)
    client.close()
    print(f"Índice vetorial pronto em {time.perf_counter() - inicio:.0f}s")

    bm25 = BM25Retriever.from_defaults(nodes=nos, stemmer=stemmer(), language="pt")
    bm25.persist(str(BM25_DIR))
    print(f"Índice BM25 pronto em {BM25_DIR}")


if __name__ == "__main__":
    main()
