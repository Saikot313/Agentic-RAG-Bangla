FROM python:3.11-slim

WORKDIR /app

# faiss-cpu and pypdf need minimal build deps; kept slim on purpose
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# FAISS index persists here — mount a volume in docker-compose to keep data
# across container restarts.
RUN mkdir -p /app/storage/faiss_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
