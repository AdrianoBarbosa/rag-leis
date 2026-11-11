# RAG de legislação brasileira

Projeto de estudo de RAG (Retrieval-Augmented Generation) com **LlamaIndex**, usando o acervo
[ctresb/leis](https://github.com/ctresb/leis) e focado em três áreas: **trabalhista**,
**tributária** e **inventário/sucessões**.

```
                       INDEXAÇÃO (uma vez)
 markdown das leis ─► chunk por artigo ─► bge-m3 (Ollama) ─► Qdrant (vetores)
                                     └──────────────────────► BM25 (palavras)

                       CONSULTA (a cada pergunta)
 pergunta ─► busca vetorial ┐
          └► busca BM25    ─┴► fusão ─► top-k trechos ─► prompt ─► LLM ─► resposta com citações
```

## Rodando

Pré-requisitos: `uv`, Ollama rodando localmente.

```bash
uv sync
ollama pull bge-m3                      # modelo de embeddings (1,2 GB)
cp .env.example .env                    # preencha as chaves dos provedores que for usar

uv run python -m rag_leis.download      # baixa só o corpus (~13 MB de ~8 GB do repo)
uv run python -m rag_leis.ingest        # chunks + embeddings + BM25 (~1,5 min com GPU)
uv run python -m rag_leis.avaliar -v    # mede a recuperação, sem LLM
uv run streamlit run app.py             # interface
```

Pelo terminal:

```bash
# só recuperação (ótimo para estudar o retriever isolado)
uv run python -m rag_leis.perguntar "quem herda se não há filhos?"
# com geração
uv run python -m rag_leis.perguntar "quem herda se não há filhos?" -p "Ollama (local)" -m qwen3:8b
```

> O Qdrant em modo local trava a pasta do índice para um processo. Feche o Streamlit antes de
> rodar `ingest`, `avaliar` ou `perguntar`.

Provedores de LLM disponíveis na interface: Ollama (local), Ollama (Cloud), API Claude,
API DeepSeek, API Gemini e OpenRouter. Os embeddings são sempre locais.

## Mapa do código (cada arquivo é um conceito)

| Arquivo | Conceito de RAG |
| --- | --- |
| `rag_leis/corpus.py` | **Curadoria do corpus.** Quais normas entram e quais faixas de artigos. |
| `rag_leis/download.py` | Ingestão de dados brutos. |
| `rag_leis/parsing.py` | **Chunking estrutural** e **metadados**. O ponto que mais afeta a qualidade. |
| `rag_leis/indices.py` | Embeddings, vector store (Qdrant) e índice lexical (BM25). |
| `rag_leis/ingest.py` | Pipeline de indexação. |
| `rag_leis/engine.py` | **Retriever híbrido**, filtros por metadado, multi-query e o prompt de resposta. |
| `rag_leis/llms.py` | Abstração de LLM: seis provedores atrás da mesma interface. |
| `rag_leis/avaliar.py` | **Avaliação da recuperação** (hit@k, MRR) com gabarito. |
| `app.py` | Interface Streamlit com todos os botões de ajuste. |

Depois de rodar o `ingest`, abra `data/storage/nodes.jsonl`: são os chunks exatamente como foram
indexados. Olhar os chunks é o hábito mais útil de quem trabalha com RAG.

## O que este projeto ensina (com achados reais)

### 1. Chunking: corte pela estrutura do documento, não por tamanho

Cortar a cada 512 tokens partiria artigos ao meio e misturaria incisos de artigos vizinhos. Aqui
cada **artigo** vira um chunk, e só artigos longos (acima de 512 tokens) são subdivididos com
sobreposição. Cada chunk carrega `norma`, `artigo` e `hierarquia` (Livro > Título > Capítulo),
que entram no texto do embedding e no contexto do LLM, e é isso que permite a resposta citar
`[Código Civil, Art. 1.829]`. Já `fonte`, `dominio` e `arquivo` ficam fora do embedding
(`excluded_embed_metadata_keys`) porque só seriam ruído no vetor.

### 2. Dados reais são sujos. Valide antes de indexar

Três problemas apareceram só ao conferir a saída do parser:

- **A CLT gerava 2 chunks em vez de ~1000.** O arquivo começa com o decreto que a aprova, que
  tem o próprio fecho ("...da Independência e da República"). O parser parava ali. Correção:
  usar o último fecho do arquivo.
- **A Lei 6.858/1980 não gerava nenhum chunk.** Ela usa `**Art**. 1º` em vez de `**Art. 1º**`.
  A pergunta sobre FGTS de falecido falhava no avaliador por causa disso, não por causa do
  modelo de embeddings.
- **A LC 214/2025 (IBS/CBS) está quebrada no repo**, só tem um trecho de retificação. Foi
  substituída pela LC 227/2026, que traz o IBS/CGIBS e, de bônus, as normas gerais do ITCMD.

Moral: quando a recuperação falha, olhe primeiro os dados, depois o modelo.

### 3. Busca vetorial x BM25 x híbrida

Resultado do `avaliar` (24 perguntas, k=8):

| Modo | hit@1 | hit@k | MRR | MRR só nas referências literais |
| --- | ---: | ---: | ---: | ---: |
| vetorial | 0.88 | 1.00 | 0.92 | 0.73 |
| BM25 | 0.46 | 0.71 | 0.55 | 0.90 |
| híbrida RRF | 0.62 | 1.00 | 0.77 | 0.90 |
| **híbrida ponderada (0.7)** | 0.83 | 1.00 | 0.91 | 0.90 |

- O **vetorial** entende paráfrase: "demissão por justa causa" acha o art. 482 da CLT mesmo
  sem as palavras exatas.
- O **BM25** ganha em referências literais ("Súmula 437 do TST", "art. 1.830").
- A fusão **RRF** (padrão de muitos tutoriais) trata os dois como iguais e, aqui, o BM25
  fraco nas perguntas coloquiais puxa o ranking para baixo.
- A fusão **ponderada** (`relative_score` com peso 0,7 no vetorial) pega o melhor dos dois.
  O peso foi escolhido medindo, não chutando. Faça o mesmo a cada mudança.

### 4. Filtro por metadado

O seletor de domínios vira um `MetadataFilter` aplicado **antes** da busca: no Qdrant como
filtro de payload, no BM25 como máscara de pesos. Pré-filtrar é diferente de buscar tudo e
descartar depois, que pode deixar você com menos de k resultados.

### 5. Recuperação boa não garante resposta boa

Exemplo real: "Meu pai morreu sem testamento, casado em comunhão parcial, com dois filhos. Como
fica a herança dos bens particulares?" O retriever trouxe os arts. 1.829 e 1.791, mas não o
**art. 1.832** (quanto o cônjuge recebe concorrendo com os filhos), que é a resposta exata.
E o `qwen3.5:4b` ainda trocou "sua mãe" por "você". Dois problemas diferentes: recall da
recuperação e capacidade do gerador. Compare o mesmo contexto com um modelo maior na interface.

## Próximos passos sugeridos

1. **Reranker**: recuperar 30 trechos e reordenar com um cross-encoder (por exemplo
   `bge-reranker-v2-m3`) antes de mandar 8 ao LLM. Deve resolver casos como o do art. 1.832.
2. **Chat com memória**: trocar o query engine por `CondensePlusContextChatEngine` para
   perguntas de seguimento ("e se não houver filhos?").
3. **Avaliar a geração**: fidelidade (a resposta se apoia nos trechos?) e se as citações
   existem nos trechos recuperados. Dá para checar as citações com regex, sem LLM.
4. **Limpeza de chunk**: remover notas como "(Redação dada pela Lei...)" do texto do embedding
   e medir se o hit@1 melhora.
5. **ITCMD estadual**: adicionar a lei do seu estado em `corpus.py` (`estaduais/<UF>/...`).
6. **Comparar com LangChain** e depois **refazer do zero** (numpy + sqlite), agora sabendo o
   que cada abstração do framework esconde.

---

Os textos do acervo têm caráter informativo e não substituem a publicação oficial nem a
orientação de um advogado.
