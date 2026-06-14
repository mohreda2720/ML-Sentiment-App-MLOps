# ML Sentiment App — Starter

A sentiment analysis pipeline using Qwen2.5-1.5B, tracked with MLflow.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate a sample evaluation dataset
python -c "from src.data_loader import create_sample_dataset; create_sample_dataset('data/eval_sentiment.csv')"

# Run training (after completing TODOs)
python -m src.train
```

## API Reference

Similar to assignment 1, the starter includes a FastAPI skeleton in
`serving/traffic_router.py`, but this time it will serve as a **starting
point** for the expected inference API contract. In Part B you will
build your own serving layer (`serving/traffic_router.py`) that
exposes the following endpoints:

- `GET /health` — Health check
- `POST /predict` — Sentiment prediction (JSON body: `{"text": "..."}`)
