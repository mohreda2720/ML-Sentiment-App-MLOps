#!/usr/bin/env bash
# Analysis entrypoint for the Argo Rollouts AnalysisRun Job.
#
# 1. Sends test predictions to both BASELINE_URL and CANARY_URL
# 2. Runs drift detection comparing baseline vs canary predictions
# 3. Pushes drift score to Prometheus pushgateway
#
# Environment variables (set by AnalysisTemplate):
#   BASELINE_URL         — V1 vLLM completions endpoint
#   CANARY_URL           — V2 vLLM completions endpoint
#   EVIDENTLY_URL        — Evidently UI service (optional)
#   PROMETHEUS_PUSHGATEWAY_URL — Prometheus pushgateway for metrics
set -euo pipefail

BASELINE_URL="${BASELINE_URL:?BASELINE_URL is required}"
CANARY_URL="${CANARY_URL:?CANARY_URL is required}"
PROMETHEUS_PUSHGATEWAY_URL="${PROMETHEUS_PUSHGATEWAY_URL:-http://prometheus-pushgateway.monitoring:9091}"
EVIDENTLY_URL="${EVIDENTLY_URL:-http://cisc814-evidently:8000}"

echo "=== Analysis Job Starting ==="
echo "  Baseline: $BASELINE_URL"
echo "  Canary:   $CANARY_URL"

python3 send_predictions.py --url "$BASELINE_URL" --version v1 --output baseline_predictions.jsonl
python3 send_predictions.py --url "$CANARY_URL" --version v2 --output canary_predictions.jsonl

echo "=== Running Drift Detection + Pushing Metrics ==="
python3 drift_detector.py \
    --reference baseline_predictions.jsonl \
    --current canary_predictions.jsonl \
    --output drift_report.html \
    --push \
    --evidently-url "$EVIDENTLY_URL" \
    --pushgateway-url "$PROMETHEUS_PUSHGATEWAY_URL"

echo "=== Analysis Job Complete ==="
