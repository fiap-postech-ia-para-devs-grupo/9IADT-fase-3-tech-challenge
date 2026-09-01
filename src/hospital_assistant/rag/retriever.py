"""Top-k retrieval com fonte + score, per ESTRATEGIA.md §4.

Score é a similaridade de cosseno (1 - distância cosseno do Chroma), então
"quanto maior, mais relevante" — a Tela 2 exibe esse valor cru para
explicabilidade, então a direção importa: se a normalização mudar aqui, o
significado do número exibido na UI muda junto.

Limitação conhecida (investigada em #7, mitigação tentada em #9):
`all-MiniLM-L6-v2` — modelo fixado em ESTRATEGIA.md §1, "Decisões Fechadas —
não reabrir" — é treinado majoritariamente em inglês e nem sempre separa bem
textos clínicos curtos em português; ex. a query "dor torácica aguda" não traz
`protocolos_sinteticos/dor_toracica_aguda.md` no top-3, mesmo com embeddings
normalizados + distância cosseno. Remover o boilerplate institucional
repetido entre os protocolos sintéticos (feito em `ingest.py`, #9) **não**
resolveu esse caso específico — as outras 4 queries testadas (pneumonia,
exame urgente, crise hipertensiva, qSOFA) rankeiam corretamente. Conclusão:
é limitação de cobertura do modelo em português para esse par
query/documento, não bug de implementação nem de corpus — trocar de modelo
resolveria, mas está fora de escopo (decisão fechada).
"""

from __future__ import annotations

from typing import TypedDict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from hospital_assistant.paths import CHROMA_DIR
from hospital_assistant.rag.ingest import COLLECTION_NAME, EMBEDDING_KWARGS, EMBEDDING_MODEL


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


def _store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, **EMBEDDING_KWARGS)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = 3) -> list[RetrievedChunk]:
    results = _store().similarity_search_with_score(query, k=k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "desconhecida"),
            # store persistido com hnsw:space=cosine, então a "distância" já é
            # 1 - similaridade_de_cosseno; inverter dá de volta a similaridade.
            "score": 1.0 - distance,
        }
        for doc, distance in results
    ]
