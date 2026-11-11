"""Turn measured results into a written report with a Groq-hosted LLM, grounded in the reference PDFs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from .accel import AccelResult
from .fuse import Pothole

MODEL = os.getenv("REPORT_MODEL", "openai/gpt-oss-20b")  # Groq model, same as the chatbot

SYSTEM = (
    "You write road-condition reports from measured data. Use ONLY the numbers provided; never "
    "invent measurements, locations or thresholds. If a field is null (e.g. depth_cm) say it was "
    "not measured and why (depth_note). Roughness values marked uncalibrated are relative, not IRI. "
    "Cite reference excerpts only when you actually rely on them (file/page), and only for "
    "definitions or thresholds - never quote an accuracy/error figure from a reference as if it "
    "applied to this survey. Be concise: a short summary, a table of potholes, roughness per "
    "segment, and 2-4 caveats/recommendations.\n"
    "Field meanings (follow exactly): peak_to_peak_ms2 is a vertical acceleration swing in m/s^2, "
    "NOT a length. status 'confirmed' = camera detection matched to an accelerometer spike; "
    "'video_only' = seen on camera but no spike (avoided or shallow); 'accel_only' = spike with no "
    "matching camera detection - this just means no video evidence (no video may have been "
    "provided); do not call it a bump-vs-pothole classification. severity is a relative band from "
    "spike size, not an engineering grade. rms_az is m/s^2; rms_per_speed is 1/s."
)


def build_payload(accel: AccelResult, potholes: list[Pothole], iri_calibrated: bool) -> dict:
    return {
        "survey": {
            "duration_s": round(accel.duration_s, 1),
            "distance_m": round(accel.distance_m),
            "speed_source": accel.speed_source,
            "roughness_calibrated_to_iri": iri_calibrated,
        },
        "potholes": [asdict(p) for p in potholes],
        "roughness_segments": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(s).items()}
            for s in accel.segments
        ],
    }


def retrieve_references(query: str = "IRI roughness thresholds pothole severity classification") -> str:
    """Pull excerpts from the existing PDF index (data/pdf). Returns '' if unavailable."""
    try:
        from src.chatbot.rag_pipeline import RAGPipeline, build_context

        rag = RAGPipeline()
        rag.ensure_ingested()
        return build_context(rag.retrieve(query))
    except Exception:  # RAG deps/index optional - report still works without references
        return ""


def write_report(payload: dict, references: str = "") -> str:
    from openai import OpenAI  # Groq is OpenAI-compatible

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
    user = "Measured data (JSON):\n" + json.dumps(payload, indent=1)
    if references:
        user += "\n\nReference excerpts:\n" + references
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,  # factual report: keep it low
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""
