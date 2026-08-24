# ==============================================================================
# DocuMind AI — Enterprise Document Intelligence System Dockerfile
# Base: Python 3.10 Slim with System OCR, PDF, Audio & GPU Support
# ==============================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies (OCR, PDF processing, Audio/Video, CV & build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and project configuration
COPY requirements.txt pyproject.toml /app/

# Install Python dependencies from reconciled requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and application files
COPY src /app/src
COPY ui /app/ui

# Create persistent storage & cache directories with open permissions for HF Spaces / container execution
RUN mkdir -p /app/data /app/logs /app/embeddings /app/.cache && \
    chmod -R 777 /app/data /app/logs /app/embeddings /app/.cache

# Expose Hugging Face Spaces port (7860) and Streamlit default port (8501)
EXPOSE 7860 8501

# Healthcheck probing Streamlit status on active port
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit Application (defaults to port 7860 for Hugging Face Spaces)
CMD ["streamlit", "run", "ui/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
