FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY traffic_router.py .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "traffic_router:app", "--host", "0.0.0.0", "--port", "8000"]
