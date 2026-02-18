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

### Pipeline Stages

The benchmark runs 4 sequential stages:

1. **Ingest** — Feed conversation turns into mnemory via `add_memory(infer=True)`
2. **Search** — Query mnemory for each question via `search_memories` or `find_memories`
3. **Answer** — Generate answers using an eval LLM with retrieved memories as context
4. **Evaluate** — LLM judge scores answers against ground truth (CORRECT/WRONG)

Each stage saves its state to disk, so you can resume or re-run individual stages.

### Configuration

```bash
# Run only specific stages
python -m benchmarks.locomo run --stages ingest
python -m benchmarks.locomo run --stages search,answer,evaluate

# Use find_memories (AI-powered multi-query search) instead of search_memories
python -m benchmarks.locomo run --search-method find_memories

# Adjust search result count
python -m benchmarks.locomo run --search-limit 20

# Use a specific model for answering/judging
python -m benchmarks.locomo run --eval-model gpt-4o-mini --judge-model gpt-4o-mini

# Override mnemory's LLM model for extraction
python -m benchmarks.locomo run --llm-model gpt-4o-mini

# Skip LLM extraction (infer=False) for faster ingestion
python -m benchmarks.locomo run --no-infer

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

| Stage | Approximate Cost (gpt-4o-mini) |
|---|---|
| Ingest (infer=True) | ~$3-5 |
| Ingest (infer=False) | ~$0.10 |
| Search | ~$0.05 |
| Answer | ~$1-2 |
| Evaluate | ~$1-2 |
| **Total** | **~$5-10** |

### Reference

- Paper: [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753) (ACL 2024)
- Dataset: [snap-research/locomo](https://github.com/snap-research/locomo)
- Published scores from [EverMemOS evaluation](https://github.com/EverMind-AI/EverMemOS/blob/main/evaluation/README.md)
