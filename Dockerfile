FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
# Copy __init__.py and README.md alongside pyproject.toml so hatchling
# can resolve the package version and readme during the build step.
COPY pyproject.toml README.md ./
COPY mnemory/__init__.py mnemory/__init__.py
RUN pip install --no-cache-dir ".[all]"

# Copy application code
COPY mnemory/ mnemory/

# Create data directory for SQLite history and local artifacts
RUN mkdir -p /data

EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8050/health')" || exit 1

CMD ["mnemory"]
