"""RAG pipeline: PDFs in data/pdf -> chunks -> embeddings -> ChromaDB -> retrieval."""

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()  # before Langfuse import so it sees credentials
from langfuse import get_client, observe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "pdf"
VECTOR_DIR = ROOT / "data" / "vector_store"
COLLECTION_NAME = "pdf_docs_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


_TOKEN_RE = re.compile(r"\w+")
RRF_K = 60  # standard RRF damping constant


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


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
        self._bm25: BM25Okapi | None = None
        self._embeddings = np.empty((0, 0))
        self._bm25_corpus: tuple[list[str], list[str], list[dict]] = ([], [], [])

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
        self._bm25 = None

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
        self._bm25 = None  # corpus changed -> rebuild lazily
        return len(chunks)

    def ensure_ingested(self) -> None:
        if self.collection.count() == 0:
            self.ingest()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))

    def _get_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._bm25_corpus = (data["ids"], data["documents"], data["metadatas"])
            self._embeddings = np.asarray(data["embeddings"], dtype=float)
            self._bm25 = BM25Okapi([_tokenize(d) for d in data["documents"]])
        return self._bm25

    @observe(name="retrieve-context", as_type="retriever", capture_input=False, capture_output=False)
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.3,
        mode: str = "hybrid",
        candidates: int = 20,
    ) -> list[dict[str, Any]]:
        """mode: 'dense' (vector only) or 'hybrid' (vector + BM25 fused with RRF).

        min_score is a cosine-similarity floor applied in both modes, so a keyword-only
        match (e.g. "Hi" matching a greeting in a PDF) can't pull in irrelevant context.
        """
        hits = self._retrieve(query, top_k, min_score, mode, candidates)
        get_client().update_current_span(
            input=query,
            output=[{"content": h["content"], "metadata": h["metadata"], "score": h["score"]} for h in hits],
            metadata={"mode": mode, "top_k": top_k, "min_score": min_score, "num_hits": len(hits)},
        )
        return hits

    def _retrieve(
        self, query: str, top_k: int, min_score: float, mode: str, candidates: int
    ) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        n = top_k if mode == "dense" else max(candidates, top_k)
        query_vec = np.asarray(self._embed([query])[0], dtype=float)
        res = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        dense = [
            {"id": i, "content": t, "metadata": m, "score": 1 - d}
            for i, t, m, d in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        if mode == "dense":
            return [
                {k: h[k] for k in ("content", "metadata", "score")}
                for h in dense
                if h["score"] >= min_score
            ]

        bm25 = self._get_bm25()
        ids, texts, metas = self._bm25_corpus
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        sparse = [(i, float(scores[i])) for i in order[:candidates] if scores[i] > 0]

        by_id = {h["id"]: h for h in dense}
        for idx, bscore in sparse:
            h = by_id.setdefault(
                ids[idx],
                {"id": ids[idx], "content": texts[idx], "metadata": metas[idx],
                 "score": self._cosine(query_vec, self._embeddings[idx])},
            )
            h["bm25"] = bscore
        fused = reciprocal_rank_fusion(
            [[h["id"] for h in dense], [ids[idx] for idx, _ in sparse]]
        )

        hits = []
        for doc_id, rrf in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            h = by_id[doc_id]
            if h["score"] >= min_score:
                hits.append(
                    {
                        "content": h["content"],
                        "metadata": h["metadata"],
                        "score": rrf,
                        "dense_score": h["score"],
                        "bm25_score": h.get("bm25"),
                    }
                )
            if len(hits) == top_k:
                break
        return hits


def build_context(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{h['metadata']['source']}, p.{h['metadata']['page']}]\n{h['content']}"
        for h in hits
    )
