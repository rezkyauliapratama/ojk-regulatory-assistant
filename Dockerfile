FROM python:3.13-slim

WORKDIR /app

# System deps for sentence-transformers / PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app/ ./app/

RUN pip install --no-cache-dir streamlit openai psycopg2-binary python-dotenv requests pyyaml

# Nyawa BGE embedder (optional memory feature) needs Python + onnxruntime
# + numpy at runtime; python3 is used by nyawa's bge_server.py helper.
RUN pip install --no-cache-dir onnxruntime numpy tokenizers

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
