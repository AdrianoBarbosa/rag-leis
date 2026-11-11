"""Pergunta pelo terminal. Sem --provedor, mostra só o que foi recuperado (ótimo para
estudar a recuperação isolada, sem o LLM no meio).

Exemplos:
  uv run python -m rag_leis.perguntar "quem herda se não há filhos?"
  uv run python -m rag_leis.perguntar "quem herda se não há filhos?" -p "Ollama (local)" -m qwen3:8b
  uv run python -m rag_leis.perguntar "prazo para pagar rescisão" -d trabalhista --modo bm25
"""

import argparse

from .config import DOMINIOS
from .engine import PESO_VETORIAL_PADRAO, montar_query_engine, montar_retriever
from .indices import abrir_indices
from .llms import PROVEDORES, criar_llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pergunta")
    ap.add_argument("-p", "--provedor", choices=list(PROVEDORES))
    ap.add_argument("-m", "--modelo")
    ap.add_argument("-k", "--top-k", type=int, default=8)
    ap.add_argument("-d", "--dominio", action="append", choices=DOMINIOS)
    ap.add_argument("--modo", default="hibrido", choices=["hibrido", "vetorial", "bm25"])
    ap.add_argument("--peso-vetorial", type=float, default=PESO_VETORIAL_PADRAO)
    ap.add_argument("--variacoes", type=int, default=1, help="multi-query: nº de buscas geradas pelo LLM")
    args = ap.parse_args()

    indices = abrir_indices()
    llm = None
    if args.provedor:
        modelo = args.modelo or PROVEDORES[args.provedor].sugestoes[0]
        llm = criar_llm(args.provedor, modelo)

    retriever = montar_retriever(indices, args.modo, args.top_k, args.dominio, llm, args.variacoes, args.peso_vetorial)

    if llm is None:
        for i, r in enumerate(retriever.retrieve(args.pergunta), 1):
            md = r.node.metadata
            print(f"\n[{i}] score={r.score:.3f}  {md['norma']} {md['artigo']}  ({md['dominio']})")
            print(f"    {md['hierarquia']}")
            print("    " + r.node.text[:300].replace("\n", " ") + "...")
    else:
        resposta = montar_query_engine(retriever, llm, streaming=True).query(args.pergunta)
        resposta.print_response_stream()
        print("\n\nFontes:")
        for r in resposta.source_nodes:
            md = r.node.metadata
            print(f"  - {md['norma']} {md['artigo']}  {md['fonte']}")
    indices.client.close()


if __name__ == "__main__":
    main()
