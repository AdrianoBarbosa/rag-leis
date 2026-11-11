"""Fábrica de LLMs. No LlamaIndex todos implementam a mesma interface (LLM), então o
resto do pipeline não sabe nem se importa com qual provedor está gerando a resposta.

Repare que DeepSeek e OpenRouter não precisam de pacote próprio: falam o protocolo da
OpenAI, então basta o OpenAILike apontando para outra URL.
"""

import os
from dataclasses import dataclass

import httpx
from llama_index.core.llms import LLM

from .config import OLLAMA_URL

OLLAMA_CLOUD_URL = "https://ollama.com"


@dataclass(frozen=True)
class Provedor:
    env_key: str | None   # variável de ambiente com a chave de API
    sugestoes: tuple[str, ...]


PROVEDORES: dict[str, Provedor] = {
    "Ollama (local)": Provedor(None, ("qwen3:8b", "qwen3.5:4b")),
    "Ollama (Cloud)": Provedor("OLLAMA_API_KEY", ("gpt-oss:120b", "gpt-oss:20b")),
    "API Claude": Provedor("ANTHROPIC_API_KEY", ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5")),
    "API DeepSeek": Provedor("DEEPSEEK_API_KEY", ("deepseek-chat", "deepseek-reasoner")),
    "API Gemini": Provedor("GEMINI_API_KEY", ("gemini-3.7-flash",)),
    "OpenRouter": Provedor("OPENROUTER_API_KEY", ("openrouter/auto",)),
}


def chave_padrao(provedor: str) -> str:
    env = PROVEDORES[provedor].env_key
    return os.getenv(env, "") if env else ""


def listar_modelos(provedor: str, api_key: str = "") -> list[str]:
    """Para Ollama, pergunta ao servidor quais modelos existem; nos outros, usa as sugestões."""
    sugestoes = list(PROVEDORES[provedor].sugestoes)
    if not provedor.startswith("Ollama"):
        return sugestoes
    url, headers = OLLAMA_URL, {}
    if provedor == "Ollama (Cloud)":
        url = OLLAMA_CLOUD_URL
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(f"{url}/api/tags", headers=headers, timeout=5)
        r.raise_for_status()
        nomes = [m["name"] for m in r.json().get("models", [])]
        # Modelos de embedding não servem para gerar texto.
        nomes = [n for n in nomes if not any(e in n for e in ("bge", "embed", "minilm"))]
        return nomes or sugestoes
    except httpx.HTTPError:
        return sugestoes


def criar_llm(provedor: str, modelo: str, api_key: str = "", temperatura: float = 0.1) -> LLM:
    api_key = api_key or chave_padrao(provedor)
    if PROVEDORES[provedor].env_key and not api_key:
        raise ValueError(f"{provedor} precisa de chave de API ({PROVEDORES[provedor].env_key}).")

    match provedor:
        case "Ollama (local)":
            from llama_index.llms.ollama import Ollama

            # context_window vira o num_ctx do Ollama. O padrão do servidor é pequeno e
            # cortaria em silêncio parte dos trechos recuperados.
            return Ollama(model=modelo, base_url=OLLAMA_URL, temperature=temperatura,
                          context_window=8192, request_timeout=600, thinking=False)
        case "Ollama (Cloud)":
            from llama_index.llms.ollama import Ollama

            return Ollama(model=modelo, base_url=OLLAMA_CLOUD_URL, temperature=temperatura,
                          context_window=32768, request_timeout=300,
                          headers={"Authorization": f"Bearer {api_key}"})
        case "API Claude":
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(model=modelo, api_key=api_key, temperature=temperatura, max_tokens=2048)
        case "API Gemini":
            from llama_index.llms.google_genai import GoogleGenAI

            return GoogleGenAI(model=modelo, api_key=api_key, temperature=temperatura)
        case "API DeepSeek" | "OpenRouter":
            from llama_index.llms.openai_like import OpenAILike

            base = "https://api.deepseek.com/v1" if provedor == "API DeepSeek" else "https://openrouter.ai/api/v1"
            return OpenAILike(model=modelo, api_base=base, api_key=api_key, temperature=temperatura,
                              is_chat_model=True, context_window=64000, max_tokens=2048, timeout=120)
    raise ValueError(f"Provedor desconhecido: {provedor}")
