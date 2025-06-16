# ParsaCV - Multilingual CV Analyzer
# Dockerfile for containerized deployment

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download ar_core_news_sm

# Copy application code
COPY parsacv.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p temp_cvs outputs samples

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create non-root user for security
RUN useradd -m -u 1000 parsacv && \
    chown -R parsacv:parsacv /app
USER parsacv

# Run the application
CMD ["streamlit", "run", "parsacv.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]

# Metadata
LABEL maintainer="ParsaCV Team" \
      description="Advanced Multilingual CV Analyzer with NLP and Semantic Matching" \
      version="1.0.0"