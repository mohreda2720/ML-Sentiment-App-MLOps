"""Register the best model of a given experiment type in the MLflow Model Registry."""

from __future__ import annotations

import argparse

import mlflow
import yaml
from mlflow import MlflowClient


def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def register_model(
    tracking_uri: str,
    experiment_name: str,
    experiment_type: str,
    alias: str,
) -> None:
    """Register the best run of a given experiment type under the given alias."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Find best run by f1_score for this experiment_type
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.experiment_type = '{experiment_type}'",
        order_by=["metrics.f1_score DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found for experiment_type='{experiment_type}'")

    best_run = runs[0]
    run_id = best_run.info.run_id
    vllm_model_path = best_run.data.params.get("vllm_model_path", "unknown")
    print(f"Best run for '{experiment_type}': {run_id} (f1={best_run.data.metrics.get('f1_score')})")
    print(f"vllm_model_path: {vllm_model_path}")

    # Register model — we store the run ID as the model URI source
    # since we don't upload actual weights due to resource limitations
    model_name = "qwen-sentiment-model"
    try:
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=model_name
        )
    except Exception:
        # If no artifact logged, create model version directly
        client.create_registered_model(model_name)

    # Get latest version and assign alias
    versions = client.get_latest_versions(model_name)
    if not versions:
        # Create a version manually pointing to the run
        mv = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}",
            run_id=run_id,
        )
    else:
        mv = versions[-1]

    # Set vllm_model_path as a version tag for Part B retrieval
    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="vllm_model_path",
        value=vllm_model_path
    )

    # Assign alias (champion or challenger)
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version
    )

    print(f"Registered '{alias}' → version {mv.version} (run {run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model in MLflow Model Registry")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--experiment-type", required=True, help="Value of params.experiment_type to filter on")
    parser.add_argument("--alias", required=True, help="Model alias to assign (e.g. champion, challenger)")
    args = parser.parse_args()

    register_model(args.tracking_uri, args.experiment_name, args.experiment_type, args.alias)