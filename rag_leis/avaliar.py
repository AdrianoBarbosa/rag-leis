"""Avaliação da RECUPERAÇÃO, sem LLM.

Se o artigo certo não chega no top-k, nenhum LLM salva a resposta. Por isso a primeira
métrica a acompanhar é: para perguntas com resposta conhecida, o trecho certo apareceu?

- hit@k: fração das perguntas em que algum trecho esperado está entre os k primeiros.
- MRR: média de 1/posição do primeiro acerto (1.0 = sempre em 1º lugar).

Uso: uv run python -m rag_leis.avaliar [--k 8]
Adicione perguntas suas ao GABARITO: é o jeito mais honesto de medir cada mudança.
"""

import argparse

from .engine import PESO_VETORIAL_PADRAO, montar_retriever
from .indices import abrir_indices

# (pergunta, [(prefixo da norma, artigo esperado), ...]). Basta acertar um deles.
GABARITO: list[tuple[str, list[tuple[str, str]]]] = [
    # Inventário e sucessões
    ("Qual a ordem de vocação hereditária na sucessão legítima?", [("Código Civil", "Art. 1.829")]),
    ("Quem são os herdeiros necessários?", [("Código Civil", "Art. 1.845")]),
    ("Que parte da herança pertence obrigatoriamente aos herdeiros necessários?", [("Código Civil", "Art. 1.846")]),
    ("Em quanto tempo deve ser aberto o inventário depois do falecimento?", [("CPC", "Art. 611")]),
    ("Quando o inventário pode ser feito em cartório por escritura pública?", [("CPC", "Art. 610")]),
    ("Quem pode ser nomeado inventariante e em que ordem?", [("CPC", "Art. 617")]),
    ("O saldo de FGTS e PIS de quem morreu pode ser pago aos dependentes sem inventário?", [("Lei 6.858", "Art. 1º"), ("Lei 6.858", "Art. 1")]),
    ("O que acontece com a herança quando o falecido não deixa herdeiros conhecidos?", [("Código Civil", "Art. 1.819"), ("Código Civil", "Art. 1.844")]),
    # Trabalhista
    ("Quantos dias de férias o empregado tem depois de 12 meses de trabalho?", [("CLT", "Art. 130")]),
    ("Quais faltas do empregado permitem demissão por justa causa?", [("CLT", "Art. 482")]),
    ("Qual o percentual do depósito mensal do FGTS sobre o salário?", [("Lei 8.036", "Art. 15")]),
    ("A empresa que terceiriza responde pelas dívidas trabalhistas da prestadora?", [("Súmula 331 do TST", "")]),
    ("Qual a antecedência mínima do aviso prévio?", [("CLT", "Art. 487")]),
    ("Qual o prazo de prescrição para cobrar créditos trabalhistas na Justiça?", [("CF/88", "Art. 7"), ("CLT", "Art. 11")]),
    # Tributário
    ("Qual a definição legal de tributo?", [("CTN", "Art. 3")]),
    ("Quais situações suspendem a exigibilidade do crédito tributário?", [("CTN", "Art. 151")]),
    ("Em quanto tempo o Fisco perde o direito de lançar o tributo (decadência)?", [("CTN", "Art. 173"), ("CTN", "Art. 150")]),
    ("Qual o limite de faturamento para ser microempresa ou empresa de pequeno porte?", [("LC 123", "Art. 3")]),
    ("Para qual Estado é devido o ITCMD na herança de um imóvel?", [("CF/88", "Art. 155"), ("LC 227/2026 (ITCMD)", "")]),
    # Referências literais: aqui a busca por palavra (BM25) costuma ajudar
    ("O que diz o art. 1.830 do Código Civil?", [("Código Civil", "Art. 1.830")]),
    ("Súmula 437 do TST intervalo intrajornada", [("Súmula 437 do TST", "")]),
    ("art. 135 do CTN responsabilidade de sócios e administradores", [("CTN", "Art. 135")]),
    ("art. 659 CPC arrolamento sumário", [("CPC", "Art. 659")]),
    ("Súmula 112 do STJ depósito integral suspende exigibilidade", [("Súmula 112 do STJ", "")]),
]


def acertou(metadata: dict, esperados: list[tuple[str, str]]) -> bool:
    return any(
        metadata["norma"].startswith(norma) and (not artigo or metadata["artigo"] == artigo)
        for norma, artigo in esperados
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra o top-k de cada erro")
    args = parser.parse_args()

    indices = abrir_indices()
    print(f"{len(GABARITO)} perguntas, k={args.k}\n")
    configs = {
        "vetorial": dict(modo="vetorial"),
        "bm25": dict(modo="bm25"),
        "hibrido RRF": dict(modo="hibrido", peso_vetorial=None),
        f"hibrido {PESO_VETORIAL_PADRAO}": dict(modo="hibrido", peso_vetorial=PESO_VETORIAL_PADRAO),
    }
    print(f"{'modo':14s} {'hit@1':>6s} {'hit@3':>6s} {'hit@k':>6s} {'MRR':>6s}")
    erros: dict[str, list] = {}
    for modo, kwargs in configs.items():
        retriever = montar_retriever(indices, top_k=args.k, **kwargs)
        posicoes = []
        for pergunta, esperados in GABARITO:
            resultados = retriever.retrieve(pergunta)
            pos = next((i for i, r in enumerate(resultados, 1) if acertou(r.node.metadata, esperados)), None)
            posicoes.append(pos)
            if pos is None:
                erros.setdefault(modo, []).append((pergunta, resultados))
        n = len(posicoes)
        hit = lambda lim: sum(p is not None and p <= lim for p in posicoes) / n  # noqa: E731
        mrr = sum(1 / p for p in posicoes if p) / n
        print(f"{modo:14s} {hit(1):6.2f} {hit(3):6.2f} {hit(args.k):6.2f} {mrr:6.2f}")

    for modo, lista in erros.items():
        print(f"\nErros no modo {modo}:")
        for pergunta, resultados in lista:
            print(f"  - {pergunta}")
            if args.verbose:
                for r in resultados[:5]:
                    print(f"      {r.score:.3f}  {r.node.metadata['norma']} {r.node.metadata['artigo']}")
    indices.client.close()


if __name__ == "__main__":
    main()
