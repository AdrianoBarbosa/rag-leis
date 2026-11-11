"""Baixa só os arquivos do corpus (o repo inteiro tem ~8 GB).

Uso: uv run python -m rag_leis.download
"""

import asyncio
import os

import httpx

from .config import RAW_BASE, RAW_DIR, REPO
from .corpus import FONTES, SUMULAS


async def listar_diretorio(client: httpx.AsyncClient, path: str) -> list[str]:
    headers = {}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    r = await client.get(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=headers)
    r.raise_for_status()
    return [f"{path}/{item['name']}" for item in r.json() if item["name"].endswith(".md")]


async def baixar(client: httpx.AsyncClient, sem: asyncio.Semaphore, path: str) -> bool:
    destino = RAW_DIR / path
    if destino.exists():
        return False
    async with sem:
        r = await client.get(f"{RAW_BASE}/{path}")
        r.raise_for_status()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    return True


async def main() -> None:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        paths = sorted({f.path for f in FONTES})
        for diretorio in SUMULAS:
            paths += await listar_diretorio(client, diretorio)
        sem = asyncio.Semaphore(16)
        novos = await asyncio.gather(*(baixar(client, sem, p) for p in paths))
    print(f"{len(paths)} arquivos no corpus, {sum(novos)} baixados agora, em {RAW_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
