# Automated Evaluation Dataset Generator from Production Logs (EDG)

A complete, tested ML engineering pipeline that simulates an e-commerce customer-support assistant (**NovaMart**), logs and cleans production-style interactions, discovers semantic categories and edge cases with **UMAP + HDBSCAN**, grades answers with an **evidence-based hybrid LLM judge** calibrated against human annotations, freezes **immutable versioned benchmark releases**, replays multiple model versions with **statistically rigorous regression detection**, and presents all metrics in a **7-page Streamlit dashboard**.

---

## Architecture & Guarantees

1. **Ground truth first**: Store world state (customers, products, orders, payments, shipments, returns, and versioned policies) is generated and frozen into `state_snapshots` before any query is answered. The SUT and judge see the exact same snapshot.
2. **Immutable raw logs & releases**: `interactions_raw` is insert-only; `benchmark_releases` and `release_items` are immutable once frozen.
3. **Full lineage**: Every pipeline step is tracked in `pipeline_runs` with `run_id`, `config_hash`, `git_commit`, timestamps, and counts.
4. **No leakage**: Calibration splits (`judge_dev` vs `judge_test`) and benchmark splits (`dev` vs `heldout`) are strictly cluster-disjoint. Prompt tuning is blocked from reading `judge_test`.
5. **No fake numbers**: Every metric in reports and the dashboard is computed directly from database rows, with 95% bootstrap CIs and separate reporting of `mined` vs unbiased `random_control` slices.

---

## Quick Start (Local Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,dashboard]"
pip install scikit-learn umap-learn hdbscan scipy statsmodels

# Run the complete pipeline end-to-end on the --small profile (~500 queries)
edg run-all --small --provider fake --no-hf --no-st --no-spacy

# Launch the 7-page Streamlit dashboard
streamlit run dashboard/app.py
```

### Running with Docker Compose (PostgreSQL 16 + pgvector + Ollama)

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec app edg run-all --small
# Open http://localhost:8501
```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `edg db init [--small]` | Run Alembic migrations to create all 29 EDG tables |
| `edg db status` | Show existing database tables |
| `edg config show / hash [--small]` | Inspect resolved config and deterministic SHA-256 hash |
| `edg world build [--small]` | Generate policies, catalogue, store state, hard states, and snapshots |
| `edg sut run [--small] [--version v1]` | Bind queries, run tool-calling SUT, log `interactions_raw`, inject 10% controlled failures |
| `edg clean run [--small]` | Incrementally validate, clean text, pseudonymise PII, and exact/near dedupe |
| `edg embed run / compare [--small]` | Generate cached query/response embeddings and compare models on recall@5 |
| `edg cluster run [--small]` | Run 10D UMAP + HDBSCAN grid search, 2D plot UMAP, 3-seed ARI, labels, intent mapping |
| `edg priority run [--small]` | Compute 6 normalized `[0, 1]` signals and build weighted 3-part judge sample |
| `edg judge run [--small]` | Run hybrid evidence-based judge (claim extraction, code verifier, LLM, code rules, cache) |
| `edg calibrate export / import / dev-eval / freeze / test-eval` | Cluster-disjoint annotation export/import, dev tuning, judge freeze, and held-out test evaluation |
| `edg label run [--small]` | Map judgements to failure labels, taxonomy severities, and human review queue |
| `edg testcases generate [--small]` | Draft & snapshot-validate checklists, dedupe within cluster, create critical variants |
| `edg release create <v1.0> [--small]` | Freeze approved gold/silver test cases into an immutable release |
| `edg eval replay <v1.0> --model <v1\|v2>` | Replay model over bound snapshots and grade with the release's frozen judge |
| `edg eval compare <v1.0> --baseline v1 --candidate v2` | Paired bootstrap 95% CIs, exact McNemar test, Holm correction, and regression flags |
| `edg run-all [--small]` | Run the complete end-to-end pipeline and write `data/exports/results_summary.md` |

---

## Data Sources, Licences, and Synthetic Provenance

Every record carries an explicit `source_tag` and is never presented as real Amazon production data:

- **Bitext Customer Support Dataset** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`, Hugging Face; CDLA-Sharing-1.0 licence): Used for realistic customer-support queries and intent taxonomy (`source_tag = "public_bitext"`). When offline or `--no-hf` is passed, template-generated fallback queries are tagged `source_tag = "synthetic_natural"`.
- **Amazon Reviews 2023 Metadata** (`McAuley-Lab/Amazon-Reviews-2023`, Hugging Face; research use): Optional product title/category/price sample (`source = "amazon_catalogue_sample"`). Falls back cleanly to deterministic synthetic NovaMart products (`source = "synthetic"`).
- **Synthetic Data**:
  - `synthetic_natural`: Store world records, hard-state scenarios, and generated multi-intent/emotional/contradictory/conditional/mixed-language hard queries.
  - `synthetic_controlled`: Stratified 10% single-failure rewrites used exclusively for judge recall validation and strictly excluded from benchmark releases.
  - `synthetic_variant`: Paraphrase variants of critical test cases requiring human approval.

---

## Known Limitations

- **Single-turn scope**: Conversations are currently single-turn (`turn_index = 0`); `conversation_id` and `turn_index` are preserved in the schema for future multi-turn extension.
- **Offline fallback embeddings & LLM**: When running without live Ollama models (`--provider fake`) or without `sentence-transformers` (`--no-st`), deterministic hash-projection embeddings and deterministic tool/evidence-grounded fake SUT/judge behaviors are used for reproducibility and fast CI testing. For real semantic quality, run with Ollama + `sentence-transformers` + PostgreSQL pgvector.
- **Simulated calibration in unattended `run-all`**: `edg run-all` populates clearly labelled `sim_annotator_1`/`sim_annotator_2` annotations so the full pipeline runs without manual pauses; real deployments should import human-reviewed CSVs via `edg calibrate import` or the Page 5 Streamlit annotation form.
