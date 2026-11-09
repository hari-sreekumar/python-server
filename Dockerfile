FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only PyTorch (the default Linux wheel pulls ~2GB of CUDA libraries we can't use)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# Bake the embedding model into the image so startup doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src ./src
COPY data/pdf ./data/pdf

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "src/chatbot/server.py"]
