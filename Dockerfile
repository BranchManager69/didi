FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CODE_RAG_PATH=/app/didi
ENV CODE_RAG_REPOS_PATH=/app/repos
ENV CODE_RAG_DB_PATH=/app/data/chroma_db
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# Copy application
COPY . /app/didi

# Command
CMD ["python", "-m", "didi.scripts.query_code"]