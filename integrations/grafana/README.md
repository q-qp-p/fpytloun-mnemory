# Grafana Dashboard

Pre-built Grafana dashboard for monitoring mnemory.

## What's Included

**Overview row** — 8 stat panels: total memories (with sparkline), pinned, decayed, with artifacts, active sessions (with sparkline), total users, total agents, and server version.

**Operations row** — time series showing operation rates by type and by user, horizontal bar gauge of total operations by type, and operations rate by agent.

**Memory Breakdown row** — donut charts for memories by type (fact, preference, episodic, ...), by category (personal, work, technical, ...), and by role (user vs assistant).

**Memory Lifecycle row** — time series of memories by type over time (composition trends) and stacked area chart of active vs decayed memories (memory pool health).

**Agent Breakdown row** — donut chart of memories by agent and time series of memories by agent over time.

**Details row** — tables showing per-user/agent/type breakdown, per-user/category breakdown, and pinned memories per user/agent. Time series for total memories and active sessions over time.

**Template variables** — filter by User (`user_id`), Agent (`agent_id`), and Type (`memory_type`), all default to All. Datasource is selectable on import.

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
