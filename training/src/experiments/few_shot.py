"""Few-shot prompting experiments for sentiment classification.

The inference logic is implemented for you. Your task is to add MLflow
tracking to each experiment run.

TODO: Add MLflow integration:
- mlflow.start_run() for each shot count
- mlflow.log_param() for experiment parameters
- mlflow.log_metric() for evaluation metrics
- mlflow.log_text() / mlflow.log_dict() for artifacts
"""

from __future__ import annotations

import time

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_sentiment_dataset
from src.evaluate import compute_metrics, get_classification_report, get_confusion_matrix

FEW_SHOT_EXAMPLES = [
    {"text": "I absolutely love this product!", "label": "positive"},
    {"text": "Terrible quality, broke immediately.", "label": "negative"},
    {"text": "It arrived on time, works as described.", "label": "neutral"},
    {"text": "Best purchase I've ever made, highly recommend!", "label": "positive"},
    {"text": "Complete waste of money, very disappointed.", "label": "negative"},
]

VALID_LABELS = {"positive", "negative", "neutral"}


def build_few_shot_prompt(text: str, n_shots: int) -> str:
    """Build a few-shot prompt with n examples."""
    examples = FEW_SHOT_EXAMPLES[:n_shots]
    prompt_parts = [
        "Classify the sentiment of each text as positive, negative, or neutral.\n"
    ]
    for ex in examples:
        prompt_parts.append(f"Text: {ex['text']}\nSentiment: {ex['label']}\n")
    prompt_parts.append(f"Text: {text}\nSentiment:")
    return "\n".join(prompt_parts)


def parse_sentiment(output: str) -> str:
    """Extract sentiment label from model output."""
    output_lower = output.strip().lower()
    for label in VALID_LABELS:
        if label in output_lower:
            return label
    return "neutral"


def run_few_shot(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_path: str,
    model_name: str,
    shot_counts: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Run few-shot experiments with varying example counts.

    Returns:
        Dict mapping shot count to metrics dict.
    """
    if shot_counts is None:
        shot_counts = [1, 3, 5]

    dataset = load_sentiment_dataset(dataset_path)
    y_true = [d["label"] for d in dataset]
    results = {}


    for n_shots in shot_counts:
        with mlflow.start_run(run_name=f"few-shot-{n_shots}", nested=True):
            # mlflow.log_param("experiment_type", "few_shot")
            mlflow.set_tag("experiment_type", "few_shot")
            mlflow.log_param("n_shots", n_shots)
            mlflow.log_param("num_samples", len(dataset))
            mlflow.log_param("vllm_model_path", model_name)

            y_pred = []
            start_time = time.time()

            for sample in dataset:
                prompt = build_few_shot_prompt(sample["text"], n_shots)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                y_pred.append(parse_sentiment(generated))

            elapsed = time.time() - start_time
            metrics = compute_metrics(y_true, y_pred)
            metrics["total_inference_time_s"] = elapsed
            metrics["avg_latency_s"] = elapsed / len(dataset)

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_macro"])
            mlflow.log_metric("f1_macro", metrics["f1_macro"])
            mlflow.log_metric("total_inference_time_s", metrics["total_inference_time_s"])
            mlflow.log_metric("latency_ms", metrics["avg_latency_s"] * 1000)
            mlflow.log_metric("avg_latency_s", metrics["avg_latency_s"])

            report = get_classification_report(y_true, y_pred)
            mlflow.log_text(report, f"classification_report_{n_shots}shot.txt")

            cm = get_confusion_matrix(y_true, y_pred)
            mlflow.log_dict(
                {"confusion_matrix": cm},
                f"confusion_matrix_{n_shots}shot.json"
            )

            results[n_shots] = metrics
            print(
                f"  {n_shots}-shot: accuracy={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1_macro']:.3f}, latency={metrics['avg_latency_s']:.3f}s"
            )
    return results
