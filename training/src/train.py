"""Training script with MLflow integration.

TODO: Complete this script to:
1. Set up MLflow experiment tracking
2. Run few-shot prompting experiments (with MLflow logging)
3. Run quantization evaluation (with MLflow logging)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.experiments.few_shot import run_few_shot


def load_config(config_path: str = "configs/experiment_config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
#     """Set up MLflow tracking.

#     TODO:
#     - Set the tracking URI (use the course MLflow server)
#     - Create or get the experiment by name
#     - Set the experiment as active

#     Hint: When creating a new experiment, use MlflowClient.create_experiment()
#     with artifact_location="mlflow-artifacts:/" so that artifacts are proxied
#     through the tracking server. mlflow.set_experiment() alone does NOT set
#     the artifact location, which can cause artifact upload failures.
#     """
#     pass

def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    from mlflow import MlflowClient
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location="mlflow-artifacts:/"
        )
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment '{experiment_name}' is ready.")


def load_model_and_tokenizer(
    model_name: str, device: str = "auto"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    return model, tokenizer


def run_quantization_eval(
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    eval_path: str,
    quantization_cfg: dict,
) -> None:
    """Run quantization with evaluation and MLflow logging.

    Calls quantize_model.py with --evaluate and --mlflow-uri flags so that
    post-quantization metrics (accuracy, F1, latency, model size) are logged
    to the same MLflow tracking server.
    """
    output_dir = quantization_cfg.get("output_dir", "./qwen2.5-1.5b-gptq-4bit")
    num_samples = quantization_cfg.get("num_samples", 48)
    bits = quantization_cfg.get("bits", 4)

    quantize_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "quantization", "quantize_model.py",
    )

    cmd = [
        sys.executable, quantize_script,
        "--model", model_name,
        "--output", output_dir,
        "--method", "gptq",
        "--bits", str(bits),
        "--num-samples", str(num_samples),
        "--evaluate",
        "--eval-dataset", eval_path,
        "--mlflow-uri", tracking_uri,
        "--experiment-name", experiment_name
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    

if __name__ == "__main__":
    config = load_config()

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = f"{config['mlflow']['experiment_name']}"

    print("Setting up MLflow...")
    setup_mlflow(experiment_name, tracking_uri)

    model_name = config["model"]["name"]
    eval_path = config["data"]["eval_dataset"]

    model, tokenizer = load_model_and_tokenizer(
        model_name, device=config["model"].get("device", "auto")
    )

    quant_cfg = config.get("quantization", {})
    

    with mlflow.start_run(run_name="experiment-parent"):
        run_few_shot(model, tokenizer, eval_path, model_name)
        # run_quantization_eval(
        #     tracking_uri,
        #     experiment_name,
        #     model_name,
        #     eval_path,
        #     quant_cfg,
        # )

    print("Done!")
