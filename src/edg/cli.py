"""Typer CLI entrypoint for EDG (`edg ...`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sqlalchemy import inspect

from edg import __version__
from edg.annotation.agreement import (
    DEFAULT_FREEZE_REGISTRY,
    JudgeFreezeError,
    SplitLeakageError,
    evaluate_frozen_judge,
    evaluate_prompt_tuning,
    freeze_judge_version,
)
from edg.annotation.export import export_annotation_sheets
from edg.annotation.import_ import import_annotations_csv
from edg.benchmark.releases import ReleaseImmutabilityError, create_benchmark_release
from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.clustering.pipeline import run_clustering_pipeline
from edg.config import load_config
from edg.db.repositories import track_run
from edg.db.session import create_db_engine, get_session, init_db
from edg.embeddings.embedder import compare_embedding_models, embed_interactions
from edg.evaluation.regression import compare_models_on_release
from edg.evaluation.replay import replay_and_grade_release
from edg.judge.llm_judge import run_judge_pipeline
from edg.judge.prompts import build_judge_system_prompt, compute_prompt_hash, load_judge_rubric
from edg.labeling.rules import run_labeling_pipeline
from edg.llm.cache import LLMCache
from edg.llm.client import create_llm_client
from edg.logging import get_logger, setup_logging
from edg.pipeline.orchestrator import run_full_pipeline
from edg.priority.sampler import run_priority_and_sampling
from edg.sut.runner import run_sut_pipeline
from edg.testcases.generator import run_testcase_generation
from edg.world.store_generator import WorldAlreadyBuiltError, generate_store_world

app = typer.Typer(
    name="edg",
    help="Automated Evaluation Dataset Generator from Production Logs.",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Database schema and inspection commands.")
config_app = typer.Typer(help="Configuration inspection commands.")
world_app = typer.Typer(help="World, policies, catalogue, and snapshot generation commands.")
sut_app = typer.Typer(help="Query binding, SUT execution, and controlled failure injection.")
clean_app = typer.Typer(help="Ingestion validation, cleaning, PII pseudonymisation, and dedupe.")
embed_app = typer.Typer(help="Semantic embedding generation and model comparison.")
cluster_app = typer.Typer(help="UMAP reduction, HDBSCAN clustering, stability, and intent mapping.")
priority_app = typer.Typer(help="Edge-case priority scoring and weighted judge sampling.")
judge_app = typer.Typer(help="Evidence-based hybrid LLM judge commands.")
calibrate_app = typer.Typer(help="Human annotation export/import, judge calibration, and freezing.")
label_app = typer.Typer(help="Failure labelling, severity resolution, and human review queue.")
testcases_app = typer.Typer(help="Replayable test-case generation, checklists, dedupe, and variants.")
release_app = typer.Typer(help="Immutable versioned benchmark release creation.")
eval_app = typer.Typer(help="Model replay, frozen-judge grading, and regression detection.")

app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")
app.add_typer(world_app, name="world")
app.add_typer(sut_app, name="sut")
app.add_typer(clean_app, name="clean")
app.add_typer(embed_app, name="embed")
app.add_typer(cluster_app, name="cluster")
app.add_typer(priority_app, name="priority")
app.add_typer(judge_app, name="judge")
app.add_typer(calibrate_app, name="calibrate")
app.add_typer(label_app, name="label")
app.add_typer(testcases_app, name="testcases")
app.add_typer(release_app, name="release")
app.add_typer(eval_app, name="eval")

logger = get_logger(__name__)


@app.command("version")
def version_cmd() -> None:
    """Print the EDG package version."""
    typer.echo(f"edg {__version__}")


@db_app.command("init")
def db_init(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
) -> None:
    """Initialize all database tables via Alembic migrations."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    tables = sorted(inspect(engine).get_table_names())
    logger.info("db_initialized", extra={"database_url": cfg.database.url, "table_count": len(tables)})
    typer.echo(f"Initialized database ({len(tables)} tables): {', '.join(tables)}")


@db_app.command("status")
def db_status(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
) -> None:
    """List existing tables in the configured database."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    engine = create_db_engine(cfg.database.url, echo=cfg.database.echo)
    tables = sorted(inspect(engine).get_table_names())
    payload = {"database_url": cfg.database.url, "table_count": len(tables), "tables": tables}
    typer.echo(json.dumps(payload, indent=2))


@config_app.command("show")
def config_show(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
) -> None:
    """Display the resolved configuration as JSON."""
    cfg = load_config(config_path, small=small)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2))


@config_app.command("hash")
def config_hash(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
) -> None:
    """Print the deterministic SHA-256 hash of the resolved configuration."""
    cfg = load_config(config_path, small=small)
    typer.echo(cfg.config_hash())


@world_app.command("build")
def world_build(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    synthetic_catalogue: bool = typer.Option(
        False,
        "--synthetic-catalogue",
        help="Skip Hugging Face download and generate synthetic catalogue directly.",
    ),
    use_st: bool = typer.Option(
        True,
        "--st/--no-st",
        help="Use sentence-transformers for policy embeddings when available.",
    ),
) -> None:
    """Build store world: policies, catalogue, customers, orders, hard states, and snapshots."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    try:
        with get_session(engine) as session:
            with track_run(session, step="world_build", config=cfg) as (run, counts):
                result = generate_store_world(
                    session,
                    cfg,
                    run_id=run.run_id,
                    prefer_hf_catalogue=not synthetic_catalogue,
                    use_sentence_transformers=use_st,
                )
                counts.update(result)
    except WorldAlreadyBuiltError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(counts, indent=2))


@sut_app.command("run")
def sut_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    version: str = typer.Option("v1", "--version", "-v", help="SUT model version name (e.g. v1, v2)."),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM provider (ollama, openai_compatible, fake)."
    ),
    prefer_hf: bool = typer.Option(
        True, "--hf/--no-hf", help="Attempt to load Bitext dataset from Hugging Face."
    ),
    inject_failures: bool = typer.Option(
        True, "--inject-failures/--no-inject-failures", help="Run controlled failure injector."
    ),
) -> None:
    """Bind queries, run SUT assistant, write interactions_raw, and inject controlled tier."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step=f"sut_run_{version}", config=cfg) as (run, counts):
            result = run_sut_pipeline(
                session,
                cfg,
                run_id=run.run_id,
                version_name=version,
                prefer_hf_bitext=prefer_hf,
                inject_failures=inject_failures,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@clean_app.command("run")
def clean_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    use_spacy: bool = typer.Option(
        True, "--spacy/--no-spacy", help="Use spaCy en_core_web_sm NER for person names when installed."
    ),
) -> None:
    """Incrementally validate, clean, pseudonymise PII, and deduplicate new raw interactions."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="clean_run", config=cfg) as (run, counts):
            result = run_cleaning_pipeline(
                session,
                cfg,
                run_id=run.run_id,
                use_spacy=use_spacy,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@embed_app.command("run")
def embed_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    model_name: str | None = typer.Option(None, "--model", help="Override embedding model name."),
    use_st: bool = typer.Option(
        True, "--st/--no-st", help="Use sentence-transformers when available."
    ),
) -> None:
    """Embed clean queries and responses separately with DB caching."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="embed_run", config=cfg) as (run, counts):
            result = embed_interactions(
                session,
                cfg,
                run_id=run.run_id,
                model_name=model_name,
                use_sentence_transformers=use_st,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@embed_app.command("compare")
def embed_compare(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    use_st: bool = typer.Option(
        True, "--st/--no-st", help="Use sentence-transformers when available."
    ),
) -> None:
    """Compare the primary and alternative embedding models on within-intent recall@5 and speed."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = create_db_engine(cfg.database.url, echo=cfg.database.echo)
    with get_session(engine) as session:
        report = compare_embedding_models(session, cfg, use_sentence_transformers=use_st)
    typer.echo(json.dumps(report, indent=2))


@cluster_app.command("run")
def cluster_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM provider for cluster labeling."
    ),
) -> None:
    """Run 10D UMAP + HDBSCAN grid search, 3-seed stability, descriptions, and intent mapping."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        cache = LLMCache(engine=engine) if cfg.llm.cache_enabled else None
        client = create_llm_client(cfg.llm, cache=cache)
        with track_run(session, step="cluster_run", config=cfg) as (run, counts):
            result = run_clustering_pipeline(
                session,
                cfg,
                run_id=run.run_id,
                llm_client=client,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@priority_app.command("run")
def priority_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
) -> None:
    """Compute priority signals, rank interactions, and build weighted three-part judge sample."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="priority_run", config=cfg) as (run, counts):
            result = run_priority_and_sampling(session, cfg, run_id=run.run_id)
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@judge_app.command("run")
def judge_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM provider (ollama, openai_compatible, fake)."
    ),
    include_controlled: bool = typer.Option(
        True,
        "--include-controlled/--no-controlled",
        help="Also judge synthetic_controlled tier for recall validation.",
    ),
) -> None:
    """Run the evidence-based hybrid LLM judge on sampled interactions."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="judge_run", config=cfg) as (run, counts):
            result = run_judge_pipeline(
                session,
                cfg,
                run_id=run.run_id,
                include_controlled=include_controlled,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@calibrate_app.command("export")
def calibrate_export(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    output_dir: Path = typer.Option(
        Path("data/annotations"), "--out-dir", help="Directory for judge_dev.csv and judge_test.csv."
    ),
) -> None:
    """Export cluster-disjoint judge_dev and judge_test CSV annotation sheets."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = create_db_engine(cfg.database.url, echo=cfg.database.echo)
    with get_session(engine) as session:
        stats = export_annotation_sheets(session, cfg, output_dir=output_dir)
    typer.echo(json.dumps(stats, indent=2))


@calibrate_app.command("import")
def calibrate_import(
    csv_path: Path = typer.Argument(..., help="Path to filled human annotations CSV."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
) -> None:
    """Import human annotations from CSV into human_annotations."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="calibrate_import", config=cfg) as (run, counts):
            stats = import_annotations_csv(session, csv_path, run_id=run.run_id)
            counts.update(stats)
    typer.echo(json.dumps(stats, indent=2))


@calibrate_app.command("dev-eval")
def calibrate_dev_eval(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    split: str = typer.Option("judge_dev", "--split", help="Must be judge_dev (guard rejects judge_test)."),
) -> None:
    """Evaluate prompt-tuning agreement strictly on judge_dev."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    engine = create_db_engine(cfg.database.url, echo=cfg.database.echo)
    try:
        with get_session(engine) as session:
            report = evaluate_prompt_tuning(
                session, split=split, judge_version=cfg.judge.judge_version
            )
    except SplitLeakageError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


@calibrate_app.command("freeze")
def calibrate_freeze(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    registry_path: Path = typer.Option(
        DEFAULT_FREEZE_REGISTRY, "--registry", help="Path to frozen judge registry JSON."
    ),
) -> None:
    """Freeze the configured judge version and prompt hash before test evaluation."""
    cfg = load_config(config_path, small=small)
    rubric = load_judge_rubric()
    sys_prompt = build_judge_system_prompt(rubric)
    p_hash = compute_prompt_hash(sys_prompt, rubric)
    try:
        entry = freeze_judge_version(
            judge_version=cfg.judge.judge_version,
            judge_model=cfg.judge.model_name,
            prompt_hash=p_hash,
            registry_path=registry_path,
        )
    except JudgeFreezeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(entry, indent=2))


@calibrate_app.command("test-eval")
def calibrate_test_eval(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    registry_path: Path = typer.Option(
        DEFAULT_FREEZE_REGISTRY, "--registry", help="Path to frozen judge registry JSON."
    ),
) -> None:
    """Evaluate a frozen judge version on the held-out judge_test split."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    engine = create_db_engine(cfg.database.url, echo=cfg.database.echo)
    try:
        with get_session(engine) as session:
            report = evaluate_frozen_judge(
                session,
                judge_version=cfg.judge.judge_version,
                registry_path=registry_path,
            )
    except JudgeFreezeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(report, indent=2))


@label_app.command("run")
def label_run(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
) -> None:
    """Turn judgements into failure labels, severities, and human review queue flags."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step="label_run", config=cfg) as (run, counts):
            result = run_labeling_pipeline(session, cfg, run_id=run.run_id)
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@testcases_app.command("generate")
def testcases_generate(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM provider for checklist/variant drafting."
    ),
    variants: bool = typer.Option(
        True, "--variants/--no-variants", help="Generate paraphrase variants of critical cases."
    ),
) -> None:
    """Generate replayable test cases with snapshot-validated checklists and within-cluster dedupe."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        cache = LLMCache(engine=engine) if cfg.llm.cache_enabled else None
        client = create_llm_client(cfg.llm, cache=cache)
        with track_run(session, step="testcases_generate", config=cfg) as (run, counts):
            result = run_testcase_generation(
                session,
                cfg,
                run_id=run.run_id,
                llm_client=client,
                generate_variants=variants,
            )
            counts.update(result)
    typer.echo(json.dumps(counts, indent=2))


@release_app.command("create")
def release_create(
    release_id: str = typer.Argument(..., help="Immutable release identifier (e.g. v1.0)."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    description: str | None = typer.Option(None, "--description", "-d", help="Release notes."),
) -> None:
    """Freeze approved gold/silver test cases into an immutable benchmark release."""
    overrides = {"database": {"url": database_url}} if database_url else None
    cfg = load_config(config_path, small=small, overrides=overrides)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    try:
        with get_session(engine) as session:
            with track_run(session, step=f"release_create_{release_id}", config=cfg) as (run, counts):
                res = create_benchmark_release(
                    session,
                    cfg,
                    release_id=release_id,
                    run_id=run.run_id,
                    description=description,
                )
                counts.update(res)
    except ReleaseImmutabilityError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(counts, indent=2))


@eval_app.command("replay")
def eval_replay(
    release_id: str = typer.Argument(..., help="Frozen release ID (e.g. v1.0)."),
    model: str = typer.Option(..., "--model", "-m", help="SUT model version (e.g. v1, v2)."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(None, "--provider", help="Override LLM provider."),
) -> None:
    """Replay a model version on a frozen release and grade with its frozen judge."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(session, step=f"eval_replay_{release_id}_{model}", config=cfg) as (run, counts):
            res = replay_and_grade_release(
                session,
                cfg,
                release_id=release_id,
                model_version_name=model,
                run_id=run.run_id,
            )
            counts.update(res)
    typer.echo(json.dumps(counts, indent=2))


@eval_app.command("compare")
def eval_compare(
    release_id: str = typer.Argument(..., help="Frozen release ID (e.g. v1.0)."),
    baseline: str = typer.Option("v1", "--baseline", "-b", help="Baseline model version."),
    candidate: str = typer.Option("v2", "--candidate", "-n", help="Candidate model version."),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(None, "--provider", help="Override LLM provider for flip rate."),
) -> None:
    """Compare baseline vs candidate models on a frozen release with CIs, McNemar, and Holm correction."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)
    engine = init_db(cfg.database.url, use_alembic=True, echo=cfg.database.echo)
    with get_session(engine) as session:
        with track_run(
            session, step=f"eval_compare_{release_id}_{baseline}_vs_{candidate}", config=cfg
        ) as (run, counts):
            res = compare_models_on_release(
                session,
                cfg,
                release_id=release_id,
                baseline_model_id=baseline,
                candidate_model_id=candidate,
                run_id=run.run_id,
            )
            counts.update(
                {
                    "release_id": release_id,
                    "baseline": baseline,
                    "candidate": candidate,
                    "n_shared_cases": res["n_shared_cases"],
                    "flagged_regressions_count": res["flagged_regressions_count"],
                }
            )
    typer.echo(json.dumps(res, indent=2))


@app.command("run-all")
def run_all(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    small: bool = typer.Option(False, "--small", help="Use the small development config profile."),
    database_url: str | None = typer.Option(None, "--db-url", help="Override database URL."),
    provider: str | None = typer.Option(
        None, "--provider", help="Override LLM provider (ollama, openai_compatible, fake)."
    ),
    release_id: str = typer.Option("v1.0", "--release-id", help="Benchmark release ID to create."),
    prefer_hf: bool = typer.Option(False, "--hf/--no-hf", help="Download HF datasets when enabled."),
    use_st: bool = typer.Option(False, "--st/--no-st", help="Use sentence-transformers when enabled."),
    use_spacy: bool = typer.Option(False, "--spacy/--no-spacy", help="Use spaCy NER when enabled."),
    annotations_dir: Path = typer.Option(
        Path("data/annotations"), "--annotations-dir", help="Directory for exported/imported CSVs."
    ),
    summary_out: Path = typer.Option(
        Path("data/exports/results_summary.md"),
        "--summary-out",
        help="Path to write DB-backed results_summary.md.",
    ),
) -> None:
    """Run the entire EDG pipeline end-to-end from empty DB to frozen release, V1 vs V2 report, and summary."""
    overrides: dict[str, object] = {}
    if database_url:
        overrides["database"] = {"url": database_url}
    if provider:
        overrides["llm"] = {"provider": provider}
    cfg = load_config(config_path, small=small, overrides=overrides or None)
    setup_logging(cfg.logging)

    try:
        res = run_full_pipeline(
            cfg,
            release_id=release_id,
            prefer_hf=prefer_hf,
            use_sentence_transformers=use_st,
            use_spacy=use_spacy,
            annotations_dir=annotations_dir,
            freeze_registry_path=annotations_dir / "frozen_judges.json",
            summary_output_path=summary_out,
        )
    except WorldAlreadyBuiltError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    app()
