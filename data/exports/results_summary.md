# Automated Evaluation Dataset Generator — Results Summary

> Every number in this report is computed directly from the SQLAlchemy database.

## Lineage & Run Metadata

- **Profile**: `default` (seed `42`)
- **Config Hash**: `b9794534fd1f2f166d819f913b0567dd6504273f1d023f21433edbc880c96f7d`
- **Benchmark Release**: `v1.0` (content hash `e5869f566b5c0a4a51839cab79a55879ce866ccd646efbaacf4752840a3205ca`)
- **Frozen Judge Version**: `j1.0` (`llama3.1:8b`)
- **Models Compared**: `v1` (`qwen2.5:1.5b`) vs `v2` (`qwen2.5:7b`)
- **Recorded Pipeline Runs**: 14

## 1. Data Processing

| Run ID | Step | Status | Key Counts |
|---|---|---|---|
| `run_a1b8243d44f2` | `world_build` | `completed` | policy_chunks=17, products=3000, catalogue_source=synthetic, customers=2000, orders=5000, payments=5200, shipments=4530, refunds_returns=1184, snapshots=5000 |
| `run_862b60eb04f6` | `sut_run_v1` | `completed` | model_version=v1, bitext_queries=8000, hard_queries=1500, natural_inserted=9500, errors=0, controlled_injected=950 |
| `run_6ee2b8a6f642` | `clean_run` | `completed` | records_in=10450, records_out=10450, valid_count=10450, rejected_count=0, pii_replacements=0, dup_groups=7526, dup_share=0.2798, throughput_per_min=560940.3 |
| `run_875028d6985e` | `embed_run` | `completed` | model_name=all-MiniLM-L6-v2, queries_processed=10450, responses_processed=10450, rows_inserted=20900, cache_hits=0, cache_misses=20900, cache_hit_rate=0.0, seco |
| `run_f6259ca16434` | `cluster_run` | `completed` | clustering_run_id=clrun_ee70b4f63d42, representatives_clustered=6797, duplicates_propagated=2703, best_min_cluster_size=15, best_min_samples=10, n_clusters=192, |
| `run_18b0ac68c1d6` | `priority_run` | `completed` | clustering_run_id=clrun_ee70b4f63d42, scored_count=9500, min_score=0.0795, mean_score=0.301, max_score=0.7748, population_size=9500, sampled_total=6000, random_ |
| `run_417ee66d7a45` | `judge_run` | `completed` | total_targets=6950, judged_new=6950, cache_hits=0, judge_errors=0, abstentions=0, failures_detected=2346, judge_model=llama3.1:8b, judge_version=j1.0, prompt_ha |
| `run_26fa278cdd30` | `calibrate_import` | `completed` | dev_imported=200, test_imported=400, double_annotated=50 |
| `run_e3b1a5f1845b` | `label_run` | `completed` | labeled_count=6950, failures_count=2346, needs_human_review_count=993 |
| `run_442134224f8e` | `testcases_generate` | `completed` | candidates_considered=4111, duplicates_merged=8, test_cases_created=4103, approved_silver=3411, pending_candidate=692, variants_created=10 |
| `run_36d66115473c` | `release_create_v1.0` | `completed` | release_id=v1.0, content_hash=e5869f566b5c0a4a51839cab79a55879ce866ccd646efbaacf4752840a3205ca, judge_version=j1.0, total_cases=3411 |
| `run_149f04cb886f` | `eval_replay_v1.0_v1` | `completed` | release_id=v1.0, model_id=v1, judge_version=j1.0, total_release_items=3411, evaluated_new=3411, passed_new=2208, pass_rate_new=0.6473 |
| `run_52396a55764e` | `eval_replay_v1.0_v2` | `completed` | release_id=v1.0, model_id=v2, judge_version=j1.0, total_release_items=3411, evaluated_new=3411, passed_new=3147, pass_rate_new=0.9226 |
| `run_857a4efd046c` | `eval_compare_v1.0_v1_vs_v2` | `completed` | n_shared_cases=3411, flagged_regressions_count=0 |

## 2. Embedding Model Comparison

Evaluated on `7526` representative queries:

| Model | Within-Intent Recall@5 | Seconds / 1k Texts |
|---|---|---|
| `all-MiniLM-L6-v2` | 0.9993 | 0.011 |
| `BAAI/bge-small-en-v1.5` | 0.9995 | 0.012 |

## 3. Clustering Quality

- **Clustering Run ID**: `clrun_ee70b4f63d42`
- **Clusters Discovered**: 192
- **Noise Share**: 19.74%
- **DBCV (relative_validity)**: 0.4517
- **3-Seed Stability (Pairwise Non-Noise ARI)**: 0.8951
- **Mean Intent Purity**: 0.9227
- **Candidate New Intents**: 54
- **Silhouette (Non-Noise Points Only)**: 0.7729

## 4. Priority Scoring & Lift vs. Random Control

| Group | n | Failure Rate (Precision@k) | Lift vs Random Control |
|---|---|---|---|
| `random_control` | 1200 | 19.17% | 1.0x |
| `top_50` | 50 | 72.00% | 3.76x |
| `top_100` | 100 | 67.00% | 3.5x |
| `top_500` | 500 | 75.40% | 3.93x |

## 5. Judge Validation (`judge_test`) & Reliability

- **Paired Items (`judge_test`)**: 400
- **Double-Annotated Human-Human Ceiling Subset**: 50 items

| Dimension | Assessed (Excluded N/A) | Weighted Kappa | Spearman | MAE | Pass/Fail F1 | Human-Human Kappa |
|---|---|---|---|---|---|---|
| `correctness` | 400 (0) | 0.5808 | 0.6286 | 0.5125 | 0.8970 | 0.9275 |
| `relevance` | 400 (0) | 0.0000 | N/A | 0.0500 | 1.0000 | -0.0309 |
| `helpfulness` | 400 (0) | 0.5807 | 0.6388 | 0.3500 | 0.8970 | 0.9291 |
| `completeness` | 400 (0) | 0.0000 | N/A | 0.0350 | 1.0000 | -0.0274 |
| `safety` | 400 (0) | 0.0000 | N/A | 0.0500 | 1.0000 | -0.0606 |

- **Controlled-Tier Overall Recall**: 99.79% (`n=950`)
- **Abstention Rate**: 0.00%
- **Judge Error Rate**: 0.00%
- **Length-vs-Score Spearman (Bias Check)**: -0.1201
- **Judge Repeat Flip Rate**: 0.00% (n=171)

## 6. Benchmark Dataset Quality (`v1.0`)

- **Total Frozen Cases**: 3411
- **By Slice**: `mined=2334`, `random_control=1077`
- **By Split (Cluster-Disjoint)**: `dev=1478`, `heldout=1933`
- **Coverage**: 27 intents, 184 clusters
- **By Severity**: {'high': 700, 'low': 2711}
- **Pool Approval Rate**: 82.93% (4113 total pool cases)

## 7. Model Comparison (`v1` vs `v2` on `v1.0`)

Mined and `random_control` slices are reported separately alongside overall metrics:

| Scope | Metric | Baseline | Candidate | Paired Diff | 95% CI | McNemar p | Holm p_adj | Flagged |
|---|---|---|---|---|---|---|---|---|
| `overall` | `correctness_pass_rate` | 79.48% | 100.00% | 0.2052 | [0.1914, 0.2196] | 0.000000 | 0.000000 | `False` |
| `overall` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `overall` | `failure_rate` | 20.52% | 0.00% | -0.2052 | [-0.2196, -0.1914] | 0.000000 | 0.000000 | `False` |
| `overall` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `overall` | `pass_rate` | 64.73% | 92.26% | 0.2753 | [0.2606, 0.2902] | 0.000000 | 0.000000 | `False` |
| `overall` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `correctness_pass_rate` | 74.59% | 100.00% | 0.2541 | [0.2365, 0.2716] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `failure_rate` | 25.41% | 0.00% | -0.2541 | [-0.2716, -0.2365] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:mined` | `pass_rate` | 59.73% | 91.39% | 0.3166 | [0.2978, 0.3359] | 0.000000 | 0.000000 | `False` |
| `slice:mined` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `correctness_pass_rate` | 90.06% | 100.00% | 0.0994 | [0.0817, 0.1170] | 0.000000 | 0.000000 | `False` |
| `slice:random_control` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `failure_rate` | 9.94% | 0.00% | -0.0994 | [-0.1170, -0.0817] | 0.000000 | 0.000000 | `False` |
| `slice:random_control` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `slice:random_control` | `pass_rate` | 75.58% | 94.15% | 0.1857 | [0.1625, 0.2089] | 0.000000 | 0.000000 | `False` |
| `slice:random_control` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `correctness_pass_rate` | 69.76% | 100.00% | 0.3024 | [0.2801, 0.3248] | 0.000000 | 0.000000 | `False` |
| `split:dev` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `failure_rate` | 30.24% | 0.00% | -0.3024 | [-0.3248, -0.2801] | 0.000000 | 0.000000 | `False` |
| `split:dev` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:dev` | `pass_rate` | 52.57% | 89.78% | 0.3721 | [0.3491, 0.3958] | 0.000000 | 0.000000 | `False` |
| `split:dev` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `correctness_pass_rate` | 86.91% | 100.00% | 0.1309 | [0.1164, 0.1459] | 0.000000 | 0.000000 | `False` |
| `split:heldout` | `critical_failure_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `failure_rate` | 13.09% | 0.00% | -0.1309 | [-0.1459, -0.1164] | 0.000000 | 0.000000 | `False` |
| `split:heldout` | `hallucination_rate` | 0.00% | 0.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |
| `split:heldout` | `pass_rate` | 74.03% | 94.15% | 0.2012 | [0.1837, 0.2193] | 0.000000 | 0.000000 | `False` |
| `split:heldout` | `relevance_pass_rate` | 100.00% | 100.00% | 0.0000 | [0.0000, 0.0000] | 1.000000 | 1.000000 | `False` |

- **Total Scope × Metric Comparisons**: 1962
- **Categories Marked Insufficient (`n < 30`)**: 1548
- **Flagged Regressions**: 0
