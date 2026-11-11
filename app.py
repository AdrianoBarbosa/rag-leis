"""Interface do RAG de legislação.

Uso: uv run streamlit run app.py
"""

import streamlit as st

from rag_leis.config import DOMINIOS
from rag_leis.engine import PESO_VETORIAL_PADRAO, montar_query_engine, montar_retriever
from rag_leis.indices import abrir_indices
from rag_leis.llms import PROVEDORES, chave_padrao, criar_llm, listar_modelos

st.set_page_config(page_title="RAG Leis", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Abrindo índices...")
def indices():
    # Um único cliente Qdrant por processo: o modo local não aceita dois abertos.
    return abrir_indices()


@st.cache_data(ttl=60, show_spinner=False)
def modelos(provedor: str, api_key: str) -> list[str]:
    return listar_modelos(provedor, api_key)


ROTULO_DOMINIO = {"trabalhista": "Trabalhista", "tributario": "Tributário", "inventario": "Inventário"}

with st.sidebar:
    st.header("Modelo")
    provedor = st.selectbox("Provedor", list(PROVEDORES))
    api_key = ""
    if PROVEDORES[provedor].env_key:
        api_key = st.text_input(
            f"Chave ({PROVEDORES[provedor].env_key})",
            value=chave_padrao(provedor),
            type="password",
            help="Pré-preenchida a partir do .env, se existir.",
        )
    modelo = st.selectbox(
        "Modelo",
        modelos(provedor, api_key),
        accept_new_options=True,
        help="Digite outro nome para usar um modelo fora da lista.",
    )
    temperatura = st.slider("Temperatura", 0.0, 1.0, 0.1, 0.05)

    st.header("Recuperação")
    dominios = st.multiselect(
        "Domínios", DOMINIOS, default=DOMINIOS, format_func=ROTULO_DOMINIO.get
    )
    modo = st.radio("Busca", ["hibrido", "vetorial", "bm25"], horizontal=True,
                    format_func={"hibrido": "Híbrida", "vetorial": "Vetorial", "bm25": "BM25"}.get)
    peso = PESO_VETORIAL_PADRAO
    if modo == "hibrido":
        peso = st.slider("Peso do vetorial na fusão", 0.0, 1.0, PESO_VETORIAL_PADRAO, 0.05,
                         help="1.0 = só vetorial; 0.0 = só BM25.")
    top_k = st.slider("Trechos recuperados (top-k)", 2, 20, 8)
    variacoes = st.slider("Variações da pergunta (multi-query)", 1, 5, 1,
                          help="Acima de 1, o LLM reescreve a pergunta e cada versão é buscada. "
                               "Melhora o recall, custa uma chamada extra ao LLM.")
    so_busca = st.toggle("Só recuperar (sem LLM)", help="Para estudar a recuperação isolada.")

st.title("⚖️ RAG de legislação")
st.caption("Trabalhista · Tributário · Inventário. Fonte: github.com/ctresb/leis. "
           "Respostas informativas, não substituem um advogado.")

if "historico" not in st.session_state:
    st.session_state.historico = []


def mostrar_trechos(trechos: list[dict]) -> None:
    with st.expander(f"📚 {len(trechos)} trechos recuperados"):
        for i, t in enumerate(trechos, 1):
            st.markdown(f"**[{i}] {t['norma']} {t['artigo']}** · score `{t['score']:.3f}` · {t['dominio']}  \n"
                        f"<small>{t['hierarquia']}</small> · [fonte oficial]({t['fonte']})",
                        unsafe_allow_html=True)
            st.text(t["texto"])


for item in st.session_state.historico:
    with st.chat_message(item["papel"]):
        st.markdown(item["texto"])
        if item.get("trechos"):
            mostrar_trechos(item["trechos"])

if pergunta := st.chat_input("Pergunte sobre CLT, CTN, Simples, herança, inventário..."):
    st.session_state.historico.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        try:
            llm = None
            if not so_busca or variacoes > 1:
                llm = criar_llm(provedor, modelo, api_key, temperatura)
            retriever = montar_retriever(indices(), modo, top_k, dominios, llm, variacoes, peso)

            if so_busca:
                with st.spinner("Buscando..."):
                    nos = retriever.retrieve(pergunta)
                texto = "Modo só recuperação: veja os trechos abaixo."
                st.markdown(texto)
            else:
                with st.spinner("Buscando e gerando..."):
                    resposta = montar_query_engine(retriever, llm, streaming=True).query(pergunta)
                texto = st.write_stream(resposta.response_gen)
                nos = resposta.source_nodes

            trechos = [
                {**n.node.metadata, "score": n.score or 0.0, "texto": n.node.text} for n in nos
            ]
            mostrar_trechos(trechos)
            st.session_state.historico.append({"papel": "assistant", "texto": texto, "trechos": trechos})
        except Exception as e:  # erro de chave, modelo inexistente, Ollama desligado...
            st.error(f"{type(e).__name__}: {e}")
