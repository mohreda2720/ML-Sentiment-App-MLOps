"""Data loading utilities for sentiment evaluation datasets."""

from __future__ import annotations

import csv
from pathlib import Path


def load_sentiment_dataset(path: str) -> list[dict[str, str]]:
    """Load a sentiment dataset from a CSV file.

    Expected CSV format: text,label
    Labels should be: positive, negative, neutral

    Args:
        path: Path to the CSV file.

    Returns:
        List of dicts with 'text' and 'label' keys.
    """
    data = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"text": row["text"], "label": row["label"]})
    return data


def create_sample_dataset(output_path: str, n_samples: int = 100) -> None:
    """Create a sample sentiment dataset for testing.

    This generates a small dataset you can use to verify your pipeline
    before running on larger data.

    Args:
        output_path: Where to write the CSV.
        n_samples: Number of samples to generate.
    """
    samples = [
        ("I love this product, it works perfectly!", "positive"),
        ("Terrible experience, would not recommend.", "negative"),
        ("The package arrived on time.", "neutral"),
        ("Amazing quality and great customer service!", "positive"),
        ("Broke after one day of use, very disappointing.", "negative"),
        ("It is what it is, nothing special.", "neutral"),
        ("Best purchase I have ever made!", "positive"),
        ("Worst product ever, complete waste of money.", "negative"),
        ("Standard shipping, standard product.", "neutral"),
        ("Exceeded all my expectations, highly recommend!", "positive"),
    ]

    with Path(output_path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for i in range(n_samples):
            text, label = samples[i % len(samples)]
            writer.writerow([text, label])
