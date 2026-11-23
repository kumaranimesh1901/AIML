# Automated Evaluation Dataset Generator from Production Logs (EDG)

A complete, tested ML engineering pipeline that simulates an e-commerce customer-support assistant (**NovaMart**), logs and cleans production-style interactions, discovers semantic categories and edge cases with **UMAP + HDBSCAN**, grades answers with an **evidence-based hybrid LLM judge** calibrated against human annotations, freezes **immutable versioned benchmark releases**, replays multiple model versions with **statistically rigorous regression detection**, and presents all metrics in a **interactive Streamlit dashboard** (headline results, a live Chat Assistant, and one page per pipeline stage).

---

## Architecture & Guarantees

1. **Ground truth first**: Store world state (customers, products, orders, payments, shipments, returns, and versioned policies) is generated and frozen into `state_snapshots` before any query is answered. The SUT and judge see the exact same snapshot.
2. **Immutable raw logs & releases**: `interactions_raw` is insert-only; `benchmark_releases` and `release_items` are immutable once frozen.
3. **Full lineage**: Every pipeline step is tracked in `pipeline_runs` with `run_id`, `config_hash`, `git_commit`, timestamps, and counts.
4. **No leakage**: Calibration splits (`judge_dev` vs `judge_test`) and benchmark splits (`dev` vs `heldout`) are strictly cluster-disjoint. Prompt tuning is blocked from reading `judge_test`.
5. **No fake numbers**: Every metric in reports and the dashboard is computed directly from database rows, with 95% bootstrap CIs and separate reporting of `mined` vs unbiased `random_control` slices.

---

## Quick start: open the dashboard with real results (about 5 minutes)

The repository ships the results of a real run (`data/demo/edg_demo.db`): every answer was
written by local `qwen2.5` models on Ollama and graded by the frozen `qwen2.5:14b` judge, nothing
in fake mode. So you can explore every page without running the pipeline yourself.

You need macOS (or Linux), `git` and **Python 3.11 or newer** (`brew install python@3.11`).

```zsh
git clone https://github.com/kumaranimesh1901/aiml.git
cd aiml
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dashboard]"
edg db init
python -m streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501. `edg db init` copies the bundled results into
`data/edg.db` (your own working copy, ignored by git) the first time, and never replaces a
database that already holds data. Next time, only `cd aiml`, `source .venv/bin/activate` and
`python -m streamlit run dashboard/app.py` are needed. Use `python -m streamlit`, not plain
`streamlit`: it makes sure the project's own Python is used, even if another Streamlit is
installed on your Mac (for example through Homebrew).

| Page | What it shows | Needs Ollama? |
|---|---|---|
| Home, Conversations, Topics & Clusters, Edge Cases, AI Judge, Test Dataset, v1 vs v2 | The pipeline's results, read live from the database | No |
| Chat Assistant | Ask NovaMart's assistant (v1 or v2) about a real order; each answer is checked by the AI judge | **Yes** |
| Support Desk | Tickets from unhappy customers or serious AI errors; send a corrected answer and re-evaluate it | Only for re-evaluation |

### Turn on the live chat (optional)

Install [Ollama](https://ollama.com/download), open it once, then pull the three models
(about 15 GB in total; the 14b judge needs a Mac with 16 GB of memory or more):

```zsh
ollama pull qwen2.5:1.5b   # v1 assistant, about 1 GB
ollama pull qwen2.5:7b     # v2 assistant and helper, about 4.7 GB
ollama pull qwen2.5:14b    # frozen AI judge, about 9 GB
```

Or let the project do it: the **Local models** box in the dashboard's sidebar shows whether
Ollama is running and whether the v1 and v2 assistants and the judge are there, with a
**Download** button (and a progress bar) for each one that is missing. From the terminal,
`edg doctor` runs the same check and asks before downloading each missing model
(`edg doctor --pull` downloads without asking).

Answers take a few seconds each on an M-series Mac, plus a few more when the AI quality check is on.

### Customer feedback and support escalation

The **Chat Assistant** asks *"Are you satisfied with the solution provided?"* after every
answer, and an AI quality check (the frozen judge) grades each answer separately. A turn escalates
to a support ticket when the customer says no **or** the judge/verification finds a serious error
(high/critical severity or a claim that contradicts the order record), even if the customer was
satisfied. On the **Support Desk** page an executive sends a corrected answer to the customer; the
judge re-evaluates it, and a validated correction becomes a gold *candidate* test case (approve it
on Test Dataset to include it in the next release). Batch re-evaluation: `edg support reevaluate`;
counts: `edg support stats`. Tables: `chat_turns`, `support_tickets` (migration 0003).

---

## Run the whole pipeline yourself (optional, hours)

This regenerates every number from scratch with your own local models. Use real Ollama models
for real results: `--provider fake` exists only so the test suite can run without models, and
its numbers mean nothing.

```zsh
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:3b && ollama pull qwen2.5:7b && ollama pull qwen2.5:14b
pip install -e ".[dev,dashboard,ml]"
python -m spacy download en_core_web_sm

rm -f data/edg.db               # start from an empty database (the bundled copy stays in data/demo)
edg db init --empty
edg run-all --small --hf --st --spacy
python -m streamlit run dashboard/app.py
```

| Profile | Command | Time on an M-series MacBook Pro |
|---|---|---|
| Smoke (tiny, checks the setup) | `edg run-all --config configs/smoke.yaml --hf --st --spacy --annotations-dir data/annotations_smoke --summary-out data/exports/smoke_summary.md` | about 20 to 30 minutes |
| Small (the bundled results) | `edg run-all --small --hf --st --spacy` | about 5 hours (judge about 2 h 45 min, replays about 1 h 40 min) |
| Final (full size) | `edg run-all --config configs/final.yaml --hf --st --spacy` | about 15 to 18 hours |

`run-all` rewrites `data/exports/results_summary.md`; run `git checkout -- data/exports/results_summary.md`
before `git pull` to avoid a conflict. To share a new run's results, `edg db snapshot` writes
them to `data/demo/edg_demo.db` (without the LLM cache or chat test sessions) for you to commit.

Tests: `pytest` (uses fake mode internally, no models needed).

### Running with Docker Compose (PostgreSQL 16 + pgvector + Ollama)

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:14b
docker compose exec app edg run-all --small
# Open http://localhost:8501
```

---

## CLI Command Reference

| Command | Description |
|---|---|
| `edg db init [--small] [--empty]` | Run Alembic migrations to create all EDG tables; on a fresh clone, first load the bundled results unless `--empty` |
| `edg doctor [--pull]` | Check that Ollama runs and has the v1/v2 assistants and the judge; offer to download any that is missing |
| `edg db snapshot` | Save the current run's results to `data/demo/edg_demo.db` so they can be committed and shared |
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

- **Single-turn pipeline**: Logged pipeline conversations are single-turn (`turn_index = 0`). The live Chat Assistant keeps the earlier turns of a chat as context, but only the pipeline's single-turn logs feed the benchmark.
- **Offline fallback embeddings & LLM**: When running without live Ollama models (`--provider fake`) or without `sentence-transformers` (`--no-st`), deterministic hash-projection embeddings and deterministic tool/evidence-grounded fake SUT/judge behaviors are used for reproducibility and fast CI testing. For real semantic quality, run with Ollama + `sentence-transformers` + PostgreSQL pgvector.
- **Simulated calibration in unattended `run-all`**: `edg run-all` populates clearly labelled `sim_annotator_1`/`sim_annotator_2` annotations so the full pipeline runs without manual pauses; real deployments should import human-reviewed CSVs via `edg calibrate import`. The bundled results also include a real two-annotator check of the judge (`edg calibrate human-import`), shown on the AI Judge page.
