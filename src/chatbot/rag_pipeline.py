"""RAG pipeline: PDFs in data/pdf -> chunks -> embeddings -> ChromaDB -> retrieval."""

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "pdf"
VECTOR_DIR = ROOT / "data" / "vector_store"
COLLECTION_NAME = "pdf_docs_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGPipeline:
    def __init__(
        self,
        pdf_dir: Path = PDF_DIR,
        persist_dir: Path = VECTOR_DIR,
        collection_name: str = COLLECTION_NAME,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.pdf_dir = pdf_dir
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def ingest(self, reset: bool = False) -> int:
        """Load every PDF, chunk, embed and upsert into Chroma. Returns chunk count."""
        if reset:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_collection()

        docs = []
        for pdf in sorted(self.pdf_dir.glob("*.pdf")):
            docs.extend(PyMuPDFLoader(str(pdf)).load())

        chunks = [c for c in self.splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            return 0

        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunks):
            source = Path(chunk.metadata.get("source", "unknown")).name
            page = int(chunk.metadata.get("page", 0)) + 1
            digest = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:12]
            ids.append(f"{source}:{page}:{i}:{digest}")  # deterministic -> re-ingest is idempotent
            texts.append(chunk.page_content)
            metas.append({"source": source, "page": page})

        self.collection.upsert(
            ids=ids, documents=texts, metadatas=metas, embeddings=self._embed(texts)
        )
        return len(chunks)

    def ensure_ingested(self) -> None:
        if self.collection.count() == 0:
            self.ingest()

    def retrieve(
        self, query: str, top_k: int = 4, min_score: float = 0.3
    ) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = 1 - dist  # cosine distance -> similarity
            if score >= min_score:
                hits.append({"content": text, "metadata": meta, "score": score})
        return hits


def build_context(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{h['metadata']['source']}, p.{h['metadata']['page']}]\n{h['content']}"
        for h in hits
    )
