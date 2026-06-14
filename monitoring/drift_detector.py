"""Drift detection using Evidently AI.

Monitors prediction logs for data and target drift.
Reports are saved as HTML locally and pushed to the Evidently UI service.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

EVIDENTLY_SERVICE_URL = "http://localhost:8085"
EVIDENTLY_PROJECT_NAME = "Sentiment Drift Monitoring"


def load_predictions(log_file: str = "predictions.jsonl") -> pd.DataFrame:
    """Load prediction logs from JSON lines file."""
    records = []
    with Path(log_file).open() as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return pd.DataFrame(records)


def get_or_create_project(ws):
    """Find existing project by name or create a new one."""
    projects = ws.search_project(EVIDENTLY_PROJECT_NAME)
    if projects:
        return projects[0]
    project = ws.create_project(EVIDENTLY_PROJECT_NAME)
    project.description = "Drift monitoring for the sentiment analysis service"
    ws.update_project(project)
    return project


def push_to_evidently_service(run, service_url: str = EVIDENTLY_SERVICE_URL):
    """Push a report run to the Evidently UI service.

    Args:
        run: The report run result from report.run().
        service_url: URL of the Evidently service.
    """
    try:
        from evidently.ui.workspace import RemoteWorkspace

        ws = RemoteWorkspace(service_url)
        project = get_or_create_project(ws)
        ws.add_run(project.id, run)
        print(f"Report pushed to Evidently UI at {service_url}")
    except Exception as e:
        print(f"Warning: could not push to Evidently service: {e}")
        print("  The HTML report was still saved locally.")


def generate_drift_report(
    reference_path: str,
    current_path: str,
    output_path: str = "drift_report.html",
    push: bool = False,
    evidently_url: str = EVIDENTLY_SERVICE_URL,
) -> dict:
    """Generate a drift report comparing reference and current data.

    Args:
        reference_path: Path to reference prediction logs (baseline period).
        current_path: Path to current prediction logs (monitoring period).
        output_path: Where to save the HTML report.
        push: If True, also push the report to the Evidently UI service.
        evidently_url: URL of the Evidently service.

    Returns:
        Dict with drift detection results.
    """
    reference_df = load_predictions(reference_path)
    current_df = load_predictions(current_path)

    # Map labels to numeric for drift detection
    label_map = {"positive": 1, "negative": -1, "neutral": 0}
    reference_df["prediction_numeric"] = reference_df["predicted_label"].map(label_map)
    current_df["prediction_numeric"] = current_df["predicted_label"].map(label_map)

    # Add text length as a feature
    reference_df["text_length"] = reference_df["input_text"].str.len()
    current_df["text_length"] = current_df["input_text"].str.len()

    data_definition = DataDefinition(
        numerical_columns=["text_length", "latency_ms", "prediction_numeric"],
        categorical_columns=["model_version", "predicted_label"],
    )

    reference_ds = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current_ds = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report([DataDriftPreset()])
    result = report.run(current_ds, reference_ds)

    result.save_html(output_path)
    print(f"Drift report saved to {output_path}")

    if push:
        push_to_evidently_service(result, evidently_url)

    # Extract summary
    result_dict = result.dict()
    metrics = result_dict.get("metrics", [])
    drift_detected = False
    drift_share = 0.0
    for m in metrics:
        r = m.get("result", {})
        if "dataset_drift" in r:
            drift_detected = r["dataset_drift"]
            drift_share = r.get("drift_share", 0.0)
            break

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "report_path": output_path,
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run drift detection")
    parser.add_argument("--reference", required=True, help="Reference predictions file")
    parser.add_argument("--current", required=True, help="Current predictions file")
    parser.add_argument("--output", default="drift_report.html", help="Output report path")
    parser.add_argument(
        "--push", action="store_true",
        help="Push report to Evidently UI service",
    )
    parser.add_argument(
        "--evidently-url", default=EVIDENTLY_SERVICE_URL,
        help="Evidently service URL (default: %(default)s)",
    )
    parser.add_argument(
        "--pushgateway-url", default=None,
        help="Prometheus pushgateway URL; if set, pushes drift_share as sentiment_drift_score",
    )
    args = parser.parse_args()

    result = generate_drift_report(
        args.reference, args.current, args.output,
        push=args.push, evidently_url=args.evidently_url,
    )
    print(f"Drift detected: {result['drift_detected']}")
    print(f"Drift share: {result['drift_share']:.4f}")

    if args.pushgateway_url:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        g = Gauge("sentiment_drift_score", "Drift share from Evidently", registry=registry)
        g.set(result["drift_share"])
        try:
            push_to_gateway(args.pushgateway_url, job="analysis-job", registry=registry)
            print(f"Metrics pushed to {args.pushgateway_url}")
        except Exception as e:
            print(f"Warning: could not push to Prometheus: {e}")
