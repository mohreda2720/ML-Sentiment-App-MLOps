"""Send test predictions to a vLLM completions endpoint and save as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

TEST_TEXTS = [
    "I absolutely love this product, it is amazing!",
    "Terrible service, worst experience ever.",
    "The package arrived on time.",
    "Great quality and fantastic value for money!",
    "Completely broken, waste of money.",
    "It works as expected, nothing special.",
    "Outstanding results, exceeded all expectations!",
    "Poor quality, falls apart quickly.",
    "Standard delivery, standard product.",
    "Incredible improvement over the previous version!",
]

PROMPT_TEMPLATE = (
    "Classify the sentiment of the following text as positive, negative, or neutral.\n"
    "Text: {text}\n"
    "Sentiment:"
)


def send_predictions(url: str, version: str, repeats: int = 3) -> list[dict]:
    entries: list[dict] = []
    for text in TEST_TEXTS * repeats:
        prompt = PROMPT_TEMPLATE.format(text=text)
        start = time.time()
        try:
            r = httpx.post(url, json={"prompt": prompt, "max_tokens": 10, "temperature": 0.0}, timeout=30)
            latency = (time.time() - start) * 1000
            result = r.json()
            generated = result["choices"][0]["text"].strip().lower()
            label = "neutral"
            for candidate in ("positive", "negative", "neutral"):
                if candidate in generated:
                    label = candidate
                    break
            entries.append({
                "model_version": version,
                "input_text": text,
                "predicted_label": label,
                "latency_ms": latency,
            })
        except Exception as e:
            print(f"Error calling {url}: {e}", file=sys.stderr)
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send test predictions to a vLLM endpoint")
    parser.add_argument("--url", required=True, help="vLLM /v1/completions endpoint URL")
    parser.add_argument("--version", required=True, help="Model version label (e.g. v1, v2)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--repeats", type=int, default=3, help="Times to repeat the test set")
    args = parser.parse_args()

    print(f"Sending predictions to {args.url} (version={args.version})...")
    entries = send_predictions(args.url, args.version, args.repeats)
    print(f"  Got {len(entries)} predictions")

    with open(args.output, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
