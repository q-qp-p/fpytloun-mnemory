# mnemory Benchmarks

Evaluation benchmarks for measuring mnemory's memory quality against published baselines.

## LoCoMo Benchmark

The [LoCoMo](https://github.com/snap-research/locomo) (Long Conversation Memory) benchmark evaluates long-term conversational memory across 10 multi-session dialogues with 1540 QA questions in 4 categories:

| Category | Questions | Tests |
|---|---|---|
| single_hop | ~841 | Simple fact recall |
| temporal | ~321 | Time-related reasoning |
| multi_hop | ~282 | Connecting multiple facts |
| open_domain | ~96 | Broader knowledge questions |

### Quick Start

```bash
# Ensure OPENAI_API_KEY (or LLM_API_KEY) is set
export OPENAI_API_KEY=sk-your-key

# Download the dataset (~3MB)
python -m benchmarks.locomo download

# Run the full benchmark (all 4 stages)
python -m benchmarks.locomo run
```

By default, ingestion uses the full LLM extraction pipeline (fact extraction, classification, and deduplication). Use `--no-infer` for fast raw storage (embedding only, ~1-2 minutes).

### Pipeline Stages

The benchmark runs 4 sequential stages:

1. **Ingest** — Extract facts from conversation turns via LLM and store in mnemory (or use `--no-infer` for raw embedding-only storage)
2. **Search** — Query mnemory for each question via `search_memories` or `find_memories`
3. **Answer** — Generate answers using an eval LLM with retrieved memories as context
4. **Evaluate** — LLM judge scores answers against ground truth (CORRECT/WRONG)

Each stage saves its state to disk, so you can resume or re-run individual stages.

### Quick Test

For fast iteration (testing models, parameters, etc.):

```bash
# Quick smoke test: 1 conversation, 10 questions per category (~40 total)
python -m benchmarks.locomo run --quick

# Compare models quickly
python -m benchmarks.locomo run --quick --llm-model gpt-5-nano
python -m benchmarks.locomo run --quick --llm-model gpt-5-mini

# Quick test with reduced reasoning effort
python -m benchmarks.locomo run --quick --reasoning-effort low

# Cap ingestion to first 50 turns (composable with --quick)
python -m benchmarks.locomo run --quick --max-turns 50

# Custom quick test: 1 conversation, 20 questions per category
python -m benchmarks.locomo run --conversations 0 --max-questions 20
```

### Configuration

```bash
# Run only specific stages
python -m benchmarks.locomo run --stages ingest
python -m benchmarks.locomo run --stages search,answer,evaluate

# Disable LLM extraction — raw storage with embedding only (fast, cheap)
python -m benchmarks.locomo run --no-infer

# Limit questions per category (useful for quick tests)
python -m benchmarks.locomo run --max-questions 10

# Limit turns per conversation (cap slow infer=True ingestion)
python -m benchmarks.locomo run --max-turns 100

# Set reasoning effort for mnemory's LLM (none/minimal/low/medium/high)
python -m benchmarks.locomo run --reasoning-effort low

# Control parallel workers for ingestion (default: auto — 1 for infer, 4 for --no-infer)
python -m benchmarks.locomo run --workers 8

# Use find_memories (AI-powered multi-query search) instead of search_memories
python -m benchmarks.locomo run --search-method find_memories

# Adjust search result count
python -m benchmarks.locomo run --search-limit 20

# Use a specific model for answering/judging
python -m benchmarks.locomo run --eval-model gpt-4o-mini --judge-model gpt-4o-mini

# Override mnemory's LLM model for extraction
python -m benchmarks.locomo run --llm-model gpt-4o-mini

# Run only specific conversations (0-indexed)
python -m benchmarks.locomo run --conversations 0,1,2

# Resume a previous run
python -m benchmarks.locomo run --resume benchmarks/locomo/results/locomo_20260218_123456/

# View results from a previous run
python -m benchmarks.locomo report benchmarks/locomo/results/locomo_20260218_123456/
```

### Output

Results are saved to `benchmarks/locomo/results/<timestamp>/` with:

- `config.json` — Run configuration and resolved model names
- `ingest_state.json` — Ingestion statistics
- `search_state.json` — Retrieved memories per question
- `answer_state.json` — Generated answers
- `evaluate_state.json` — Judge scores per question
- `report.json` — Final aggregated results

Console output includes a comparison table:

```
LoCoMo Benchmark Results - mnemory v1.0.0
============================================================
Category        Correct  Total  Accuracy
------------------------------------------
single_hop          xxx    841     xx.x%
multi_hop           xxx    282     xx.x%
temporal            xxx    321     xx.x%
open_domain         xxx     96     xx.x%
------------------------------------------
Overall             xxx   1540     xx.x%

Comparison with published results (Memobase convention, gpt-4o-mini):
System           single   multi  temporal    open  Overall
----------------------------------------------------------
Memobase          70.9    52.1      85.0    77.2     75.8
Mem0              67.1    51.2      55.5    72.9     66.9
Zep               61.7    41.4      49.3    76.6     66.0
mnemory            ?.?     ?.?       ?.?     ?.?      ?.?
```

### Cost Estimate

| Stage | Approximate Cost (gpt-5-mini) |
|---|---|
| Ingest (default, LLM extraction) | ~$3-5 |
| Ingest (`--no-infer`, raw) | ~$0.10 |
| Search | ~$0.05 |
| Answer | ~$1-2 |
| Evaluate | ~$1-2 |
| **Total (default)** | **~$5-10** |
| **Total (--no-infer)** | **~$2-4** |
| **Total (--quick)** | **~$1-2** |

### Reference

- Paper: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) (ACL 2024)
- Dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- Published scores from [Memobase evaluation](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)
