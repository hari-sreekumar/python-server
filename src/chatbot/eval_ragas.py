"""Evaluate retrieval modes (dense vs hybrid) with RAGAS, reporting scores to Langfuse.

Usage:  python src/chatbot/eval_ragas.py [--modes dense hybrid] [--top-k 4]
Dataset: data/eval/questions.json -> [{"question": ..., "reference": "optional ground truth"}]
If every item has a non-empty "reference", ContextRecall is added.
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ragas 0.4.x imports a module that newer langchain_community removed; stub it.
sys.modules.setdefault(
    "langchain_community.chat_models.vertexai",
    types.SimpleNamespace(ChatVertexAI=type("ChatVertexAI", (), {})),
)

from groq import Groq  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langfuse import get_client  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    ContextRecall,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import EMBEDDING_MODEL, ROOT, RAGPipeline, build_context  # noqa: E402

GEN_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", GEN_MODEL)
RAG_PROMPT = (
    "Answer the question using only the document excerpts below. If they don't "
    "contain the answer, say so briefly.\n\nExcerpts:\n{context}\n\nQuestion: {question}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["dense", "hybrid"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--dataset", default=str(ROOT / "data" / "eval" / "questions.json"))
    args = ap.parse_args()

    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    use_reference = all(i.get("reference") for i in items)

    rag = RAGPipeline()
    rag.ensure_ingested()
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # Groq only supports n=1, so ResponseRelevancy runs with strictness=1.
    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0, max_tokens=4096))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=emb, strictness=1),
        LLMContextPrecisionWithoutReference(llm=judge),
    ]
    if use_reference:
        metrics.append(ContextRecall(llm=judge))

    lf = get_client()
    for mode in args.modes:
        rows = []
        for it in items:
            with lf.start_as_current_observation(name=f"eval-{mode}", as_type="span"):
                hits = rag.retrieve(it["question"], top_k=args.top_k, mode=mode)
            answer = groq.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": RAG_PROMPT.format(
                    context=build_context(hits), question=it["question"])}],
            ).choices[0].message.content or ""
            row = {
                "user_input": it["question"],
                "retrieved_contexts": [h["content"] for h in hits],
                "response": answer,
            }
            if use_reference:
                row["reference"] = it["reference"]
            rows.append(row)

        result = evaluate(EvaluationDataset.from_list(rows), metrics=metrics)
        df = result.to_pandas()
        out = ROOT / "data" / "eval" / f"results_{mode}.csv"
        df.to_csv(out, index=False)

        means = df.select_dtypes("number").mean()
        print(f"\n== {mode} ==\n{means.to_string()}\n(saved {out})")

        # One Langfuse trace per mode run, with mean metric scores attached.
        with lf.start_as_current_observation(name=f"ragas-eval-{mode}", as_type="span") as span:
            span.update(input={"mode": mode, "top_k": args.top_k}, output=means.to_dict())
            for metric, value in means.items():
                lf.score_current_trace(name=f"ragas_{metric}", value=float(value))
    lf.flush()


if __name__ == "__main__":
    main()
