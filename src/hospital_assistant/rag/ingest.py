"""Chroma indexing, per ESTRATEGIA.md §4.

Embute com sentence-transformers/all-MiniLM-L6-v2 e persiste um vector store
Chroma em data/chroma/, indexando os protocolos sintéticos e a amostra
MedQuAD em data/raw/.
"""

from __future__ import annotations

import shutil

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hospital_assistant.paths import CHROMA_DIR, RAW_DATA_DIR

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "protocolos_hospital"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# all-MiniLM-L6-v2 é treinado para similaridade de cosseno sobre embeddings
# normalizados — usar distância L2 sobre vetores não normalizados (o default
# do Chroma) produz ranking pior e um "score" sem faixa de valores estável.
EMBEDDING_KWARGS = {"encode_kwargs": {"normalize_embeddings": True}}
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _load_documents() -> list[tuple[str, str]]:
    """Retorna [(texto, fonte_relativa)] para cada .md em data/raw/.

    Descarta o rodapé "--- Fonte: ..." (provenance para leitura humana do
    arquivo bruto) antes de retornar: a fonte já vai no metadata via nome do
    arquivo, e sem esse corte o splitter às vezes isola o rodapé num chunk
    próprio sem conteúdo médico, que ainda assim recebe score não-trivial e
    polui o retrieval.
    """
    documents = []
    for path in sorted(RAW_DATA_DIR.rglob("*.md")):
        source = str(path.relative_to(RAW_DATA_DIR))
        text = path.read_text(encoding="utf-8").split("\n---\nFonte:")[0].strip()
        documents.append((text, source))
    return documents


def ingest() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    documents = _load_documents()
    if not documents:
        raise RuntimeError(f"Nenhum documento encontrado em {RAW_DATA_DIR} para indexar.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts: list[str] = []
    metadatas: list[dict] = []
    for text, source in documents:
        for chunk in splitter.split_text(text):
            texts.append(chunk)
            metadatas.append({"source": source})

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, **EMBEDDING_KWARGS)
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata=COLLECTION_METADATA,
    )


if __name__ == "__main__":
    ingest()
