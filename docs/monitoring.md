# Monitoring

mnemory exposes a `/metrics` endpoint in [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/), enabled by default (`ENABLE_METRICS=true`).

## Endpoint Access

By default, `/metrics` and `/health` are served on the main port with standard API key authentication. For production, set `MGMT_PORT` to serve them on a separate port without auth (see [Management Port](configuration.md#management-port)).

## Available Metrics

### Counters (in-memory, reset on restart)

| Metric | Labels | Description |
|---|---|---|
| `mnemory_operations_total` | `operation`, `user_id`, `agent_id` | Total MCP/REST operations. Operations: `add_memory`, `add_memories`, `search_memories`, `find_memories`, `update_memory`, `delete_memory`, `delete_all`, `get_core_memories`, `get_recent_memories`, `list_memories`, `save_artifact`, `get_artifact`, `list_artifacts`, `delete_artifact`, `list_categories`, `recall`, `remember`, `initialize_memory`, `fsck_check`, `fsck_apply` |

### Gauges (from Qdrant, cached)

Refreshed on each scrape with a configurable cache TTL (`METRICS_CACHE_TTL`, default 60s). Aggregated by scrolling all Qdrant points.

| Metric | Labels | Description |
|---|---|---|
| `mnemory_memories_total` | `user_id`, `agent_id`, `memory_type`, `role` | Total memories by all key dimensions |
| `mnemory_memories_decayed_total` | `user_id`, `agent_id` | Decayed (expired) memories |
| `mnemory_memories_pinned_total` | `user_id`, `agent_id` | Pinned memories |
| `mnemory_memories_by_category_total` | `user_id`, `category` | Memories per category |
| `mnemory_memories_with_artifacts_total` | `user_id`, `agent_id` | Memories with artifacts attached |
| `mnemory_active_sessions` | -- | Active memory sessions (recall/remember) |
| `mnemory_info` | `version`, `vector_backend`, `artifact_backend` | Server metadata (always 1) |

## Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: mnemory
    scrape_interval: 60s
    static_configs:
      - targets: ['mnemory:9090']  # MGMT_PORT
```

## Grafana Dashboard

A pre-built Grafana dashboard is available in [`integrations/grafana/`](../integrations/grafana/). Import `dashboard.json` into Grafana for an overview of memories, operations, and breakdowns by type/category/role with user and agent filtering.
