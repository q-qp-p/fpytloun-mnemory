# Deployment

## Installation Methods

### Using uvx (recommended)

```bash
uvx mnemory
```

mnemory starts on `http://localhost:8050/mcp`, stores data in `~/.mnemory/`.

### Using pip

```bash
pip install mnemory
mnemory
```

### Using Docker

```bash
export OPENAI_API_KEY=sk-your-key
docker-compose up -d
```

## Production Setup

For production, use remote Qdrant for vectors and S3/MinIO for artifacts:

```bash
docker run -d \
  -p 8050:8050 \
  -e OPENAI_API_KEY=sk-your-key \
  -e LLM_BASE_URL=https://your-litellm-proxy/v1 \
  -e LLM_MODEL=gpt-5-mini \
  -e QDRANT_HOST=qdrant.example.com \
  -e ARTIFACT_BACKEND=s3 \
  -e S3_ENDPOINT=http://minio.example.com:9000 \
  -e S3_ACCESS_KEY=admin \
  -e S3_SECRET_KEY=secret \
  -e MCP_API_KEYS='{"your-api-key": "your-username"}' \
  -v mnemory-data:/data \
  genunix/mnemory:latest
```

### Docker Build

```bash
# Build for linux/amd64 (for Kubernetes deployment)
docker buildx build --platform linux/amd64 -t genunix/mnemory:latest .

# Push
docker push genunix/mnemory:latest
```

## Authentication

### Single API Key

For simple setups, set a single shared API key:

```bash
MCP_API_KEY=your-secret-key
```

All clients must include `Authorization: Bearer your-secret-key` in requests. No user binding — `user_id` must come from headers or tool parameters.

### Multi-User API Keys (`MCP_API_KEYS`)

Map API keys to user IDs so the LLM doesn't need to pass `user_id` in every tool call:

```bash
MCP_API_KEYS='{"mnm-key-for-filip": "filip", "mnm-shared-service-key": "*"}'
```

- `"key": "username"` — authenticates AND binds `user_id=username` to the session
- `"key": "*"` — authenticates only (wildcard), `user_id` must come from identity headers or tool parameter

See [Configuration](configuration.md#authentication) for full identity resolution details.

## Kubernetes

### Health Probes

Set `MGMT_PORT` to serve `/health` and `/metrics` on a separate port without authentication:

```bash
MGMT_PORT=9090          # /health and /metrics on port 9090, no auth
MGMT_HOST=127.0.0.1     # Optional: bind management to localhost only
```

This is the recommended setup for Kubernetes liveness/readiness probes and Prometheus scraping. See [Configuration](configuration.md#management-port) for details.

### Stateless HTTP

mnemory runs in stateless HTTP mode (`stateless_http=True`) for Kubernetes compatibility. No sticky sessions or WebSocket connections required.

## Monitoring

mnemory exposes Prometheus metrics at `/metrics`. See [Monitoring](monitoring.md) for available metrics, Prometheus scrape config, and a pre-built Grafana dashboard.
