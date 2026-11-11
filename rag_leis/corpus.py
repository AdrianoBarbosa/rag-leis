"""Manifesto do corpus: quais normas entram, em qual domínio e quais artigos.

Curar o corpus é a primeira decisão de qualidade de um RAG. Indexar o Código Civil
inteiro para responder sobre inventário só traz ruído (contratos, família, empresa...),
então recortamos por faixa de artigos.

Ficaram de fora de propósito:
- Lei 13.467/2017 (reforma trabalhista) e Lei 11.441/2007 (inventário extrajudicial):
  são leis de alteração ("o art. X passa a vigorar com..."). O texto compilado da CLT
  e do CPC/2015 (art. 610) já incorpora o que importa, e indexar as duas versões
  faria o retriever trazer redação antiga.
- ITCMD: as normas gerais estão na LC 227/2026, mas alíquotas e detalhes são de lei
  estadual. Para incluir um estado, adicione estaduais/<UF>/... aqui.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fonte:
    path: str                      # caminho dentro do repo ctresb/leis
    dominio: str                   # trabalhista | tributario | inventario
    nome: str                      # nome curto usado nas citações
    artigos: tuple[int, int] | None = None  # faixa inclusiva; None = norma inteira


FONTES: list[Fonte] = [
    # ---- Trabalhista
    Fonte("federal/decretos-lei/1943/del-5452-1943-05-01.md", "trabalhista", "CLT"),
    Fonte("federal/leis-ordinarias/1990/lei-8036-1990-05-11.md", "trabalhista", "Lei 8.036/1990 (FGTS)"),
    Fonte("constituicao.md", "trabalhista", "CF/88", artigos=(6, 11)),
    # ---- Tributário
    Fonte("federal/leis-ordinarias/1966/lei-5172-1966-10-25.md", "tributario", "CTN"),
    # A LC 214/2025 (IBS/CBS) está quebrada no repo (só um trecho de retificação).
    # A LC 227/2026 regulamenta o IBS/CGIBS (Livro I) e o ITCMD (Livro II).
    Fonte("federal/leis-complementares/2026/lcp-227-2026-01-13.md", "tributario", "LC 227/2026 (IBS/CGIBS)", artigos=(1, 145)),
    Fonte("federal/leis-complementares/2006/lcp-123-2006-12-14.md", "tributario", "LC 123/2006 (Simples Nacional)"),
    Fonte("constituicao.md", "tributario", "CF/88", artigos=(145, 162)),
    # ---- Inventário e sucessões
    Fonte("federal/leis-ordinarias/2002/lei-10406-2002-01-10.md", "inventario", "Código Civil", artigos=(1639, 1688)),
    Fonte("federal/leis-ordinarias/2002/lei-10406-2002-01-10.md", "inventario", "Código Civil", artigos=(1784, 2027)),
    Fonte("federal/leis-ordinarias/2015/lei-13105-2015-03-16.md", "inventario", "CPC", artigos=(610, 673)),
    Fonte("federal/leis-ordinarias/1980/lei-6858-1980-11-24.md", "inventario", "Lei 6.858/1980"),
    Fonte("federal/leis-complementares/2026/lcp-227-2026-01-13.md", "inventario", "LC 227/2026 (ITCMD)", artigos=(146, 164)),
]

# Súmulas: diretórios inteiros. As do TST são todas trabalhistas; as do STJ são
# classificadas por palavra-chave (ver parsing.classificar_sumula_stj).
SUMULAS = {
    "jurisprudencia/tst/sumulas": "TST",
    "jurisprudencia/stj/sumulas": "STJ",
}
