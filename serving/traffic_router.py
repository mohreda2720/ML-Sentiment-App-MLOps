"""FastAPI traffic router for A/B testing between model versions.

Routes 80% of traffic to V1 (original) and 20% to V2 (quantized).
Logs all predictions to a JSON lines file.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

app = FastAPI(title="Sentiment A/B Traffic Router", version="2.0.0")

# Model server endpoints (K8s manifests set V1_URL / V2_URL env vars)
V1_URL = os.environ.get("V1_URL", "http://localhost:8001/v1/completions")
V2_URL = os.environ.get("V2_URL", "http://localhost:8002/v1/completions")

# Traffic split: 80% V1, 20% V2
V1_WEIGHT = float(os.environ.get("V1_WEIGHT", "0.80"))

# Prediction log file
LOG_FILE = Path(os.environ.get("LOG_FILE", "predictions.jsonl"))

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following text as positive, negative, or neutral.\n"
    "Text: {text}\n"
    "Sentiment:"
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    model_version: str
    latency_ms: float


def select_model_version() -> tuple[str, str]:
    """Select model version and completions URL based on traffic weights."""
    if random.random() < V1_WEIGHT:
        return "v1", V1_URL
    return "v2", V2_URL


def parse_sentiment(text: str) -> str:
    """Extract sentiment label from completion."""
    text_lower = text.strip().lower()
    for label in ("positive", "negative", "neutral"):
        if label in text_lower:
            return label
    return "neutral"


def log_prediction(
    version: str,
    input_text: str,
    output_label: str,
    latency_ms: float,
) -> None:
    """Append prediction to JSON lines log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "input_text": input_text,
        "predicted_label": output_label,
        "latency_ms": latency_ms,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    # 1. Format the prompt
    prompt = PROMPT_TEMPLATE.format(text=request.text)

    # 2. Select model version
    version, url = select_model_version()

    # 3. Send request and time latency
    start = time.time()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                url,
                json={
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Model server error: {e}")

    latency_ms = (time.time() - start) * 1000

    # 4. Parse sentiment label
    result = response.json()
    generated = result["choices"][0]["text"]
    label = parse_sentiment(generated)

    # 5. Log the prediction
    log_prediction(version, request.text, label, latency_ms)

    # 6. Return response
    return PredictResponse(label=label, model_version=version, latency_ms=latency_ms)


@app.get("/logs", response_class=PlainTextResponse)
def get_logs(n: int | None = None) -> str:
    """Return prediction logs as raw JSONL for drift monitoring."""
    if not LOG_FILE.exists():
        return ""
    lines = LOG_FILE.read_text().splitlines()
    if n is not None:
        lines = lines[-n:]
    return "\n".join(lines) + "\n" if lines else ""


@app.get("/stats")
def stats() -> dict:
    """Return basic A/B testing statistics."""
    if not LOG_FILE.exists():
        return {"total_predictions": 0}

    lines = LOG_FILE.read_text().splitlines()
    entries = [json.loads(l) for l in lines if l.strip()]

    v1 = [e for e in entries if e["model_version"] == "v1"]
    v2 = [e for e in entries if e["model_version"] == "v2"]
    total = len(entries)

    return {
        "total_predictions": total,
        "v1_count": len(v1),
        "v2_count": len(v2),
        "v1_pct": round(len(v1) / total * 100, 1) if total > 0 else 0,
        "v2_pct": round(len(v2) / total * 100, 1) if total > 0 else 0,
        "v1_avg_latency_ms": round(sum(e["latency_ms"] for e in v1) / len(v1), 2) if v1 else 0,
        "v2_avg_latency_ms": round(sum(e["latency_ms"] for e in v2) / len(v2), 2) if v2 else 0,
    }