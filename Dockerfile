# Multi-stage build for production
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
# Upgrade pip first
RUN pip install --upgrade pip
# Install pandas-ta from GitHub (required by openbb, not available on PyPI in required version)
# Clone manually to avoid git credential prompts - set GIT_TERMINAL_PROMPT=0 to prevent prompts
RUN GIT_TERMINAL_PROMPT=0 git clone --depth 1 https://github.com/twopirllc/pandas-ta.git /tmp/pandas-ta && \
    cd /tmp/pandas-ta && \
    pip install --no-cache-dir --user . && \
    cd /app && \
    rm -rf /tmp/pandas-ta
# Then install other requirements (includes openbb[technical])
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY api.py analysis_service.py main.py ./

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create output directory
RUN mkdir -p output

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Make start script executable
RUN chmod +x start.sh 2>/dev/null || true

# Run the application
# Railway sets PORT automatically, defaults to 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

