"""Markdown das normas -> nós (chunks) do LlamaIndex.

Esta é a etapa que mais influencia a qualidade do RAG jurídico. Em vez de cortar o
texto a cada N tokens (o que partiria um artigo ao meio e misturaria incisos de
artigos diferentes), cortamos na unidade natural da lei: o ARTIGO.

Cada nó carrega metadados que viajam junto com ele:
- norma, artigo, hierarquia (Livro > Título > Capítulo > Seção): entram no texto
  que vira embedding e no contexto que o LLM lê, para ele saber citar.
- fonte (URL oficial), dominio, arquivo: usados para filtro e para a interface,
  mas ficam fora do embedding (só seriam ruído no vetor).
"""

import json
import re
import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .config import CHUNK_OVERLAP, MAX_CHUNK_TOKENS, NODES_FILE, RAW_DIR
from .corpus import FONTES, SUMULAS, Fonte

# "**Art. 1º**", "**Art. 10.**", "**Art. 1.829.**", "**Art. 4º-A.**", "**Art**. 1º" (sic)
ART_RE = re.compile(
    r"^\*\*Art(?:\.\s*|\*\*\.\s*)(?P<num>\d{1,3}(?:\.\d{3})*)\s*[º°o]?\s*(?:-\s*(?P<suf>[A-Z]{1,2}))?\s*\.?\s*(?:\*\*)?"
)
HEADING_RE = re.compile(r"^(?P<nivel>#{2,6})\s+(?P<texto>.+)")
# Fecho da lei ("Brasília, ... 137º da República"): depois dele vêm assinaturas e anexos.
FECHO_RE = re.compile(r"da Independência e .{0,25}da República", re.IGNORECASE)
REVOGADO_RE = re.compile(r"^[\s*(\[]*(revogad|vetad)", re.IGNORECASE)

# Metadados que não devem poluir o vetor nem o prompt.
SO_FILTRO = ["fonte", "dominio", "arquivo", "tipo", "parte"]

_splitter = SentenceSplitter(chunk_size=MAX_CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP)


@dataclass
class Artigo:
    rotulo: str          # "Art. 1.829"
    numero: int          # 1829
    hierarquia: str      # "LIVRO V ... > TÍTULO II ... > CAPÍTULO I ..."
    linhas: list[str] = field(default_factory=list)

    @property
    def texto(self) -> str:
        return "\n".join(self.linhas).replace("**", "").strip()

    @property
    def revogado(self) -> bool:
        corpo = self.texto.split(" ", 2)[-1]  # tira o "Art. N"
        return len(self.texto) < 300 and bool(REVOGADO_RE.match(corpo))


def ler_markdown(path: Path) -> tuple[dict, str]:
    """Separa o frontmatter YAML do corpo."""
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, fm, corpo = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, corpo
    return {}, raw


def extrair_artigos(corpo: str) -> list[Artigo]:
    """Percorre o texto linha a linha mantendo a pilha de títulos (hierarquia)."""
    pilha: dict[int, str] = {}
    artigos: list[Artigo] = []
    atual: Artigo | None = None

    linhas = corpo.splitlines()
    # Usa o ÚLTIMO fecho: a CLT, por exemplo, começa com o decreto que a aprova,
    # que tem o próprio fecho antes do texto consolidado.
    fechos = [i for i, linha in enumerate(linhas) if FECHO_RE.search(linha)]
    if fechos:
        linhas = linhas[: fechos[-1]]

    for linha in linhas:
        if h := HEADING_RE.match(linha):
            nivel = len(h["nivel"])
            pilha = {n: t for n, t in pilha.items() if n < nivel}
            pilha[nivel] = h["texto"].strip()
            atual = None  # título novo encerra o artigo anterior
            continue
        if m := ART_RE.match(linha):
            numero = int(m["num"].replace(".", ""))
            rotulo = f"Art. {m['num']}" + (f"-{m['suf']}" if m["suf"] else "")
            atual = Artigo(rotulo, numero, " > ".join(pilha[n] for n in sorted(pilha)))
            artigos.append(atual)
        if atual is not None:
            atual.linhas.append(linha)
    return artigos


def _id(*partes: str) -> str:
    # Qdrant exige UUID; uuid5 deixa o id estável entre reindexações.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "#".join(partes)))


def _nos(texto: str, chave: str, metadata: dict) -> list[TextNode]:
    """Um nó por artigo; artigos longos viram várias partes com sobreposição."""
    pedacos = _splitter.split_text(texto)
    nos = []
    for i, pedaco in enumerate(pedacos, 1):
        md = dict(metadata)
        if len(pedacos) > 1:
            md["parte"] = f"{i}/{len(pedacos)}"
        nos.append(
            TextNode(
                id_=_id(chave, str(i)),
                text=pedaco,
                metadata=md,
                excluded_embed_metadata_keys=SO_FILTRO,
                excluded_llm_metadata_keys=SO_FILTRO,
            )
        )
    return nos


def nos_de_leis() -> Iterator[TextNode]:
    por_arquivo: dict[str, list[Fonte]] = defaultdict(list)
    for f in FONTES:
        por_arquivo[f.path].append(f)

    for path, fontes in por_arquivo.items():
        fm, corpo = ler_markdown(RAW_DIR / path)
        artigos = extrair_artigos(corpo)
        for fonte in fontes:
            for art in artigos:
                if fonte.artigos and not (fonte.artigos[0] <= art.numero <= fonte.artigos[1]):
                    continue
                if art.revogado:
                    continue
                md = {
                    "norma": fonte.nome,
                    "artigo": art.rotulo,
                    "hierarquia": art.hierarquia,
                    "fonte": fm.get("fonte", ""),
                    "dominio": fonte.dominio,
                    "arquivo": path,
                    "tipo": "lei",
                }
                yield from _nos(art.texto, f"{path}#{art.rotulo}", md)


def _limpar_pdf(texto: str) -> str:
    """Súmulas vêm de PDF: junta hifenização e quebras de linha no meio da frase."""
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)
    return re.sub(r"[ \t]+", " ", texto).strip()


def classificar_sumula_stj(texto: str) -> str | None:
    if "DIREITO TRIBUTÁRIO" in texto[:400]:
        return "tributario"
    assunto = texto[:400].upper()
    if "CIVIL" in assunto and re.search(
        r"invent[áa]ri|heran[çc]a|sucess[óo]ri|causa mortis|partilha|esp[óo]lio|herdeir", texto, re.IGNORECASE
    ):
        return "inventario"
    return None


def nos_de_sumulas() -> Iterator[TextNode]:
    for diretorio, tribunal in SUMULAS.items():
        for path in sorted((RAW_DIR / diretorio).glob("*.md")):
            fm, corpo = ler_markdown(path)
            if fm.get("revogada"):
                continue
            corpo = corpo.split("\n", 2)[-1]  # remove o "# Súmula nº X"
            if tribunal == "TST":
                if "(cancelada)" in corpo[:300].lower():
                    continue
                # O "Histórico" traz redações antigas, que confundiriam o LLM.
                corpo = corpo.split("Histórico:")[0]
                dominio = "trabalhista"
            else:
                # Fica só assunto + enunciado; os precedentes vêm depois de "Órgão Julgador".
                corpo = corpo.split("Órgão Julgador")[0]
                dominio = classificar_sumula_stj(corpo)
                if dominio is None:
                    continue
            nome = f"Súmula {fm.get('numero')} do {tribunal}"
            md = {
                "norma": nome,
                "artigo": "",
                "hierarquia": f"Jurisprudência > {tribunal}",
                "fonte": fm.get("fonte", ""),
                "dominio": dominio,
                "arquivo": str(path.relative_to(RAW_DIR)),
                "tipo": "sumula",
            }
            yield from _nos(_limpar_pdf(corpo), nome, md)


def carregar_nos() -> list[TextNode]:
    return [*nos_de_leis(), *nos_de_sumulas()]


def salvar_nos(nos: list[TextNode]) -> None:
    """Grava os chunks em JSONL para você abrir e inspecionar o que foi indexado."""
    NODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NODES_FILE.open("w", encoding="utf-8") as f:
        for n in nos:
            f.write(json.dumps({"id": n.node_id, "metadata": n.metadata, "text": n.text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from collections import Counter

    nos = carregar_nos()
    salvar_nos(nos)
    print(f"{len(nos)} nós -> {NODES_FILE}")
    for (norma, dom), qtd in sorted(Counter((n.metadata["norma"].split(" do ")[-1] if n.metadata["tipo"] == "sumula" else n.metadata["norma"], n.metadata["dominio"]) for n in nos).items()):
        print(f"  {qtd:5d}  {dom:12s} {norma}")
