"""Recuperação + geração: as duas metades do RAG.

    pergunta ──► retriever(s) ──► top-k trechos ──► prompt com os trechos ──► LLM ──► resposta
                 vetorial
                 BM25        ──► fusão (RRF)
"""

import copy
from typing import Literal

from llama_index.core import PromptTemplate
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .config import DOMINIOS
from .indices import Indices

Modo = Literal["hibrido", "vetorial", "bm25"]

# Escolhido com rag_leis.avaliar: equilibra perguntas coloquiais (vetorial vai melhor)
# e referências literais como "art. 1.830" ou "Súmula 437" (BM25 vai melhor).
PESO_VETORIAL_PADRAO = 0.7

PROMPT_RESPOSTA = PromptTemplate(
    """Você é um assistente de pesquisa jurídica em direito brasileiro, focado em direito \
trabalhista, tributário e sucessões/inventário.

Responda à pergunta usando SOMENTE os trechos de legislação e súmulas abaixo.

Regras:
- Fundamente cada afirmação citando a fonte entre colchetes, no formato \
[norma, artigo], por exemplo [Código Civil, Art. 1.829] ou [Súmula 331 do TST].
- Se os trechos não bastarem para responder, diga o que falta. Não complete com \
conhecimento externo e não invente artigos.
- Seja objetivo e use linguagem acessível. Use tópicos quando ajudar.
- Termine lembrando que a resposta é informativa e não substitui um advogado.

Trechos recuperados:
---------------------
{context_str}
---------------------

Pergunta: {query_str}
Resposta:"""
)

# Usado no modo multi-query: o LLM reescreve a pergunta de outras formas antes de buscar.
PROMPT_VARIACOES = """Você ajuda a pesquisar legislação brasileira. Gere {num_queries} \
variações de busca para a pergunta abaixo, uma por linha, usando termos técnicos \
jurídicos que provavelmente aparecem no texto da lei. Não numere as linhas.
Pergunta: {query}
Variações:
"""


def _filtro_dominio(dominios: list[str]) -> MetadataFilters | None:
    if not dominios or set(dominios) == set(DOMINIOS):
        return None
    return MetadataFilters(filters=[MetadataFilter(key="dominio", value=dominios, operator=FilterOperator.IN)])


def montar_retriever(
    indices: Indices,
    modo: Modo = "hibrido",
    top_k: int = 8,
    dominios: list[str] | None = None,
    llm: LLM | None = None,
    num_queries: int = 1,
    peso_vetorial: float | None = PESO_VETORIAL_PADRAO,
) -> BaseRetriever:
    filtros = _filtro_dominio(dominios or [])

    # O filtro vai para o Qdrant, que já busca só dentro do domínio (pré-filtro).
    vetorial = indices.vetorial.as_retriever(similarity_top_k=top_k, filters=filtros)

    # O BM25 é compartilhado entre requisições, então trabalhamos numa cópia. A máscara
    # zera o peso dos chunks fora do domínio escolhido.
    bm25 = copy.copy(indices.bm25)
    bm25.similarity_top_k = top_k
    if filtros:
        bm25.corpus_weight_mask = [int(c["dominio"] in dominios) for c in bm25.corpus]

    if num_queries <= 1 and modo != "hibrido":
        return vetorial if modo == "vetorial" else bm25

    retrievers = {"hibrido": [vetorial, bm25], "vetorial": [vetorial], "bm25": [bm25]}[modo]
    # Duas formas de fundir as listas:
    # - reciprocal_rerank (RRF): soma 1/(60 + posição) de cada lista. Ignora a escala
    #   dos scores (cosseno e BM25 são incomparáveis) e trata os retrievers como iguais.
    # - relative_score: normaliza os scores de cada lista para 0..1 e faz média
    #   ponderada, então dá para confiar mais em um retriever do que no outro.
    pesos = None
    if modo == "hibrido" and peso_vetorial is not None:
        pesos = [peso_vetorial, 1 - peso_vetorial]
    return QueryFusionRetriever(
        retrievers,
        llm=llm,
        mode="relative_score" if pesos else "reciprocal_rerank",
        retriever_weights=pesos,
        similarity_top_k=top_k,
        num_queries=num_queries,
        query_gen_prompt=PROMPT_VARIACOES,
        use_async=False,
    )


def montar_query_engine(retriever: BaseRetriever, llm: LLM, streaming: bool = False) -> RetrieverQueryEngine:
    # response_mode="compact": junta o máximo de trechos que cabe no contexto do LLM
    # numa única chamada (e só faz chamadas extras de "refine" se não couber).
    return RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        text_qa_template=PROMPT_RESPOSTA,
        response_mode="compact",
        streaming=streaming,
    )
