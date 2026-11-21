# Automated Evaluation Dataset Generator — Results Summary

> Every number in this report is computed directly from the SQLAlchemy database.

## Lineage & Run Metadata

- **Profile**: `small` (seed `42`)
- **Config Hash**: `73090618cc55e65e765b382ba79c3b5139d609bc238112d1f9b888197743da98`
- **Benchmark Release**: `v1.0` (content hash `696b48ffd66a6728bd5957d0e66dcd1be3535665adc14a971bc6236b7a6f1b31`)
- **Frozen Judge Version**: `j1.3` (`llama3.1:8b`)
- **Models Compared**: `v1` (`qwen2.5:1.5b`) vs `v2` (`qwen2.5:7b`)
- **Recorded Pipeline Runs**: 14

## 1. Data Processing

| Run ID | Step | Status | Key Counts |
|---|---|---|---|
| `run_60e2784fd50e` | `world_build` | `completed` | policy_chunks=17, products=200, catalogue_source=synthetic, customers=150, orders=300, payments=312, shipments=280, refunds_returns=69, snapshots=300 |
| `run_ab01f103b9e5` | `sut_run_v1` | `completed` | model_version=v1, bitext_queries=400, hard_queries=100, natural_inserted=500, errors=0, controlled_injected=50 |
| `run_131809874697` | `clean_run` | `completed` | records_in=550, records_out=550, valid_count=550, rejected_count=0, pii_replacements=0, dup_groups=507, dup_share=0.0782, throughput_per_min=491272.5 |
| `run_7f6365e0e754` | `embed_run` | `completed` | model_name=all-MiniLM-L6-v2, queries_processed=550, responses_processed=550, rows_inserted=1100, cache_hits=0, cache_misses=1100, cache_hit_rate=0.0, seconds_pe |
| `run_498ccae2f24b` | `cluster_run` | `completed` | clustering_run_id=clrun_3682af22c035, representatives_clustered=458, duplicates_propagated=42, best_min_cluster_size=5, best_min_samples=3, n_clusters=40, noise |
| `run_5fde93a4bc40` | `priority_run` | `completed` | clustering_run_id=clrun_3682af22c035, scored_count=500, min_score=0.096, mean_score=0.275, max_score=0.7191, population_size=500, sampled_total=300, random_cont |
| `run_ebcca8ca701f` | `judge_run` | `completed` | total_targets=350, judged_new=350, cache_hits=0, judge_errors=0, abstentions=0, failures_detected=126, judge_model=llama3.1:8b, judge_version=j1.3, prompt_hash= |
| `run_816edc2e2641` | `calibrate_import` | `completed` | dev_imported=40, test_imported=60, double_annotated=50 |
| `run_85d9dca6b1cd` | `label_run` | `completed` | labeled_count=350, failures_count=126, needs_human_review_count=60 |
| `run_397e8c3761e3` | `testcases_generate` | `completed` | candidates_considered=220, duplicates_merged=0, test_cases_created=220, approved_silver=176, pending_candidate=44, variants_created=10 |
| `run_914907efd6be` | `release_create_v1.0` | `completed` | release_id=v1.0, content_hash=696b48ffd66a6728bd5957d0e66dcd1be3535665adc14a971bc6236b7a6f1b31, judge_version=j1.3, total_cases=176 |
| `run_f7a86effa1fb` | `eval_replay_v1.0_v1` | `completed` | release_id=v1.0, model_id=v1, judge_version=j1.3, total_release_items=176, evaluated_new=176, passed_new=105, pass_rate_new=0.5966 |
| `run_bd6cd3de0d4e` | `eval_replay_v1.0_v2` | `completed` | release_id=v1.0, model_id=v2, judge_version=j1.3, total_release_items=176, evaluated_new=176, passed_new=157, pass_rate_new=0.892 |
| `run_b5e5dfb03d43` | `eval_compare_v1.0_v1_vs_v2` | `completed` | n_shared_cases=176, flagged_regressions_count=0 |

## 2. Embedding Model Comparison

Evaluated on `458` representative queries:

| Model | Within-Intent Recall@5 | Seconds / 1k Texts |
|---|---|---|
| `all-MiniLM-L6-v2` | 0.9760 | 0.023 |
| `BAAI/bge-small-en-v1.5` | 0.9760 | 0.025 |

## 3. Clustering Quality

- **Clustering Run ID**: `clrun_3682af22c035`
- **Clusters Discovered**: 40
- **Noise Share**: 8.30%
- **DBCV (relative_validity)**: 0.5354
- **3-Seed Stability (Pairwise Non-Noise ARI)**: 0.9831
- **Mean Intent Purity**: 0.8827
- **Candidate New Intents**: 17
- **Silhouette (Non-Noise Points Only)**: 0.7755

## 4. Priority Scoring & Lift vs. Random Control

| Group | n | Failure Rate (Precision@k) | Lift vs Random Control |
|---|---|---|---|
| `random_control` | 60 | 15.00% | 1.0x |
| `top_50` | 50 | 38.00% | 2.53x |
| `top_100` | 100 | 34.00% | 2.27x |
| `top_500` | 300 | 25.33% | 1.69x |

## 5. Judge Validation (`judge_test`) & Reliability

- **Paired Items (`judge_test`)**: 60
- **Double-Annotated Human-Human Ceiling Subset**: 50 items

| Dimension | Assessed (Excluded N/A) | Weighted Kappa | Spearman | MAE | Pass/Fail F1 | Human-Human Kappa |
|---|---|---|---|---|---|---|
| `correctness` | 60 (0) | 0.9500 | 0.8586 | 0.0500 | 1.0000 | 0.8707 |
| `relevance` | 60 (0) | 0.0000 | N/A | 0.0667 | 1.0000 | -0.0309 |
| `helpfulness` | 60 (0) | 0.8941 | 0.7972 | 0.1000 | 1.0000 | 0.8798 |
| `completeness` | 60 (0) | 0.0000 | N/A | 0.0667 | 1.0000 | -0.0274 |
| `safety` | 60 (0) | 0.0000 | N/A | 0.1000 | 1.0000 | -0.0606 |

- **Controlled-Tier Overall Recall**: 100.00% (`n=50`)
- **Abstention Rate**: 0.00%
- **Judge Error Rate**: 0.00%
- **Length-vs-Score Spearman (Bias Check)**: -0.0684
- **Judge Repeat Flip Rate**: 0.00% (n=9)

## 6. Benchmark Dataset Quality (`v1.0`)

- **Total Frozen Cases**: 176
- **By Slice**: `mined=123`, `random_control=53`
- **By Split (Cluster-Disjoint)**: `dev=89`, `heldout=87`
- **Coverage**: 26 intents, 35 clusters
- **By Severity**: {'low': 144, 'high': 32}
- **Pool Approval Rate**: 76.52% (230 total pool cases)

## 7. Model Comparison (`v1` vs `v2` on `v1.0`)

Mined and `random_control` slices are reported separately alongside overall metrics:

| Scope | Metric | Baseline | Candidate | Paired Diff | 95% CI | McNemar p | Holm p_adj | Flagged |
|---|---|---|---|---|---|---|---|---|
| `overall` | `correctness_pass_rate` | 81.82% | 100.00% | 0.1818 | [0.1307, 0.2386] | 0.000000 | 0.000000 | `False` |
| `overall` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `overall` | `failure_rate` | 18.18% | 0.00% | -0.1818 | [-0.2386, -0.1307] | 0.000000 | 0.000000 | `False` |
| `overall` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `overall` | `pass_rate` | 59.66% | 89.20% | 0.2955 | [0.2330, 0.3580] | 0.000000 | 0.000000 | `False` |
| `overall` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `correctness_pass_rate` | 75.61% | 100.00% | 0.2439 | [0.1789, 0.3295] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `failure_rate` | 24.39% | 0.00% | -0.2439 | [-0.3295, -0.1789] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `pass_rate` | 52.85% | 89.43% | 0.3659 | [0.2884, 0.4514] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `correctness_pass_rate` | 96.23% | 100.00% | 0.0377 | [0.0000, 0.1132] | 0.500000 | 1.000000 | `False` |
| `slice:random_control` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `failure_rate` | 3.77% | 0.00% | -0.0377 | [-0.1132, 0.0000] | 0.500000 | 1.000000 | `False` |
| `slice:random_control` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `pass_rate` | 75.47% | 88.68% | 0.1321 | [0.0377, 0.2264] | 0.015625 | 0.015625 | `False` |
| `slice:random_control` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `correctness_pass_rate` | 85.39% | 100.00% | 0.1461 | [0.0787, 0.2135] | 0.000244 | 0.000976 | `False` |
| `split:dev` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `failure_rate` | 14.61% | 0.00% | -0.1461 | [-0.2135, -0.0787] | 0.000244 | 0.000976 | `False` |
| `split:dev` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `pass_rate` | 68.54% | 96.63% | 0.2809 | [0.1963, 0.3708] | 0.000000 | 0.000000 | `False` |
| `split:dev` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `correctness_pass_rate` | 78.16% | 100.00% | 0.2184 | [0.1434, 0.3103] | 0.000004 | 0.000020 | `False` |
| `split:heldout` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `failure_rate` | 21.84% | 0.00% | -0.2184 | [-0.3103, -0.1434] | 0.000004 | 0.000020 | `False` |
| `split:heldout` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `pass_rate` | 50.57% | 81.61% | 0.3103 | [0.2184, 0.4138] | 0.000000 | 0.000000 | `False` |
| `split:heldout` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |

- **Total Scope × Metric Comparisons**: 612
- **Categories Marked Insufficient (`n < 30`)**: 540
- **Flagged Regressions**: 0
