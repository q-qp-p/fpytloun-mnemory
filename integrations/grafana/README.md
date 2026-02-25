# Grafana Dashboard

Pre-built Grafana dashboard for monitoring mnemory.

## What's Included

**Overview row** — stat panels for total memories, pinned, decayed, with artifacts, active sessions, and server version.

**Operations row** — time series showing operation rates by type and by user, plus a stacked bar chart of all operations over time.

**Memory Breakdown row** — donut charts for memories by type (fact, preference, episodic, ...), by category (personal, work, technical, ...), and by role (user vs assistant).

**Details row** — tables showing per-user/agent/type breakdown and per-user/category breakdown, time series for total memories and active sessions over time.

**Template variables** — filter by User (`user_id`) and Agent (`agent_id`), both default to All. Datasource is selectable on import.

## Prerequisites

- Prometheus scraping mnemory's `/metrics` endpoint
- Grafana with a Prometheus datasource configured

See the [Monitoring section](../../README.md#monitoring) in the main README for Prometheus scrape configuration.

## Import

1. Open Grafana
2. Go to **Dashboards > New > Import**
3. Upload `dashboard.json` or paste its contents
4. Select your Prometheus datasource
5. Click **Import**

The dashboard will appear as **mnemory** with tags `mnemory`, `mcp`, `ai`, `memory`.
