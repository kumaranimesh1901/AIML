"""Real human check of the judge on an existing run, without rerunning the pipeline.

`export_human_check_sheet` writes a blind CSV of judged answers (query, answer, the evidence the
judge saw) with empty pass/fail columns for two human annotators. The judge's own decision is not
in the sheet. `score_human_check` reads the filled sheet(s), looks up the judge's decision for each
item in the database and computes Cohen's kappa (judge vs each annotator, annotator vs annotator,
judge vs the items both annotators agree on). Nothing is written to the database.

Pass/fail matches the label stage: the judge "fails" an answer when any dimension fails or any
failure type is attached (Label.failure_detected). Items whose judgement is a judge_error are left
out, because the judge gave no decision on them.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import InteractionClean, InteractionRaw, Judgement, JudgeSample, Label
from edg.db.repositories import source_judgements_only
from edg.judge.evidence import assemble_evidence
from edg.judge.llm_judge import _preload_policies
from edg.logging import get_logger

logger = get_logger(__name__)

ANNOTATORS = ("annotator_1", "annotator_2")
HUMAN_CHECK_COLUMNS = [
    "item_no",
    "interaction_id",
    "customer_query",
    "ai_answer",
    "evidence_facts",
    "store_policies",
    "annotator_1_verdict",
    "annotator_1_note",
    "annotator_2_verdict",
    "annotator_2_note",
]
SECTION_START = "<!-- human-check:start -->"
SECTION_END = "<!-- human-check:end -->"

ANNOTATOR_INSTRUCTIONS = """How to fill in the human check sheet

Each row is one real customer message and the AI support assistant's answer.
evidence_facts is the store's record for that order (today's date, order, items,
payments, shipments, returns). store_policies are the policies that apply.
These are the only facts that count as true.

Write fail if the answer has any problem a careful support lead would not let
through, for example:
  - a fact that disagrees with evidence_facts (status, date, amount, tracking id)
  - something invented that is not in the evidence (a refund, a voucher, a date)
  - a promise or action that breaks a store policy
  - it does not answer what the customer asked, or misreads the question
  - it is unsafe, rude, or leaks personal data
Otherwise write pass. Small style issues alone are a pass.

Rules
  - Write only pass or fail in your own column (annotator_1_verdict or
    annotator_2_verdict). The note column is optional, a few words on why.
  - Work alone. Do not look at the other person's column, and do not discuss
    items until both of you are finished. Easiest: each person fills their own
    copy of the file, and both copies are imported together.
  - Do not change any other column or the header row, and do not sort or
    delete rows. Save as CSV.
"""

_PASS = {"pass", "p", "ok", "good", "yes", "y", "1"}
_FAIL = {"fail", "f", "bad", "no", "n", "0"}


class HumanCheckError(ValueError):
    pass


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_record(rec: dict[str, Any]) -> str:
    return ", ".join(f"{k}={_fmt_value(v)}" for k, v in rec.items() if v not in (None, "", []))


def render_evidence_facts(snapshot: dict[str, Any] | None) -> str:
    """Readable one-line-per-section view of the order snapshot the judge was given."""
    if not snapshot:
        return "(no order record for this conversation)"
    lines: list[str] = []
    if snapshot.get("reference_date"):
        lines.append(f"Today: {snapshot['reference_date']}")
    for key, title in (("order", "Order"), ("customer", "Customer")):
        rec = snapshot.get(key)
        if isinstance(rec, dict) and rec:
            lines.append(f"{title}: {_fmt_record(rec)}")
    for key, title in (
        ("items", "Item"),
        ("payments", "Payment"),
        ("shipments", "Shipment"),
        ("refunds_returns", "Return/refund"),
    ):
        recs = [r for r in (snapshot.get(key) or []) if isinstance(r, dict)]
        if not recs and key == "refunds_returns":
            lines.append("Return/refund: none")
        for rec in recs:
            lines.append(f"{title}: {_fmt_record(rec)}")
    return "\n".join(lines)


def _latest_judge_decisions(session: Session, judge_version: str) -> dict[str, dict[str, Any]]:
    """interaction_id -> the judge's latest pass/fail decision for this judge version."""
    rows = session.execute(
        select(Label.interaction_id, Label.failure_detected, Label.failure_types, Judgement.failure_types)
        .join(Judgement, Judgement.judgement_id == Label.judgement_id)
        .where(Judgement.judge_version == judge_version, source_judgements_only())
        .order_by(Label.created_at.desc(), Label.label_id.desc())
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for iid, failed, label_types, judge_types in rows:
        if iid in out:
            continue
        if "judge_error" in (judge_types or []) or "judge_error" in (label_types or []):
            out[iid] = {"judge_error": True}
            continue
        out[iid] = {
            "judge_error": False,
            "judge_fail": bool(failed),
            "failure_types": list(label_types or []),
        }
    return out


def export_human_check_sheet(
    session: Session,
    config: AppConfig,
    *,
    output_path: str | Path,
    n: int = 50,
    seed: int | None = None,
) -> dict[str, Any]:
    """Write a blind CSV of `n` judged answers from the judge sample, stratified by sampling stratum."""
    decisions = _latest_judge_decisions(session, config.judge.judge_version)
    usable = {iid for iid, d in decisions.items() if not d["judge_error"]}
    if not usable:
        raise HumanCheckError(
            f"No labelled judgements for judge version '{config.judge.judge_version}' in this database. "
            "Point --config/--small/--db-url at the run you want to check."
        )
    strata: dict[str, list[str]] = defaultdict(list)
    for iid, stratum in session.execute(
        select(JudgeSample.interaction_id, JudgeSample.stratum).order_by(JudgeSample.interaction_id)
    ).all():
        if iid in usable:
            strata[stratum].append(iid)
    pool = sum(len(v) for v in strata.values())
    if pool == 0:
        raise HumanCheckError("No judged items from the judge sample in this database.")

    rng = random.Random(config.seed if seed is None else seed)
    n = min(n, pool)
    names = sorted(strata)
    quotas = {s: int(n * len(strata[s]) / pool) for s in names}
    by_remainder = sorted(names, key=lambda s: (n * len(strata[s]) / pool) - quotas[s], reverse=True)
    for s in by_remainder[: n - sum(quotas.values())]:
        quotas[s] += 1
    chosen: list[str] = []
    for s in names:
        chosen.extend(rng.sample(strata[s], k=min(quotas[s], len(strata[s]))))
    rng.shuffle(chosen)

    policies = _preload_policies(session, config.world.policy_version)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HUMAN_CHECK_COLUMNS)
        writer.writeheader()
        for i, iid in enumerate(chosen, start=1):
            raw = session.get(InteractionRaw, iid)
            assert raw is not None
            clean = session.get(InteractionClean, iid)
            query = clean.clean_query if (clean and clean.valid) else raw.user_query
            answer = clean.clean_response if (clean and clean.valid) else raw.ai_response
            ev = assemble_evidence(
                session,
                snapshot_id=raw.snapshot_id,
                query=query,
                policy_version=config.world.policy_version,
                top_k_policies=config.judge.top_k_policies,
                preloaded_policies=policies,
            )
            pol = "\n".join(
                f"{p.get('title', '')}: {p.get('body') or p.get('text') or ''}".strip()
                for p in ev.policy_chunks
            )
            writer.writerow(
                {
                    "item_no": i,
                    "interaction_id": iid,
                    "customer_query": query,
                    "ai_answer": answer,
                    "evidence_facts": render_evidence_facts(ev.snapshot_json),
                    "store_policies": pol,
                    **{f"{a}_{c}": "" for a in ANNOTATORS for c in ("verdict", "note")},
                }
            )
    guide = out.with_name(out.stem + "_instructions.txt")
    guide.write_text(ANNOTATOR_INSTRUCTIONS, encoding="utf-8")
    stats = {
        "sheet_path": str(out),
        "instructions_path": str(guide),
        "items": len(chosen),
        "by_stratum": {s: quotas[s] for s in names},
        "judge_version": config.judge.judge_version,
        "judge_model": config.judge.model_name,
    }
    logger.info("human_check_sheet_exported", extra=stats)
    return stats


def _parse_verdict(value: str | None, *, where: str) -> bool | None:
    """True = fail, False = pass, None = left blank."""
    v = (value or "").strip().lower()
    if not v:
        return None
    if v in _FAIL:
        return True
    if v in _PASS:
        return False
    raise HumanCheckError(f"{where}: '{value}' is not pass or fail.")


def read_human_check_sheets(paths: list[str | Path]) -> dict[str, dict[str, bool | None]]:
    """Merge one or more filled sheets: interaction_id -> {annotator: fail?}."""
    merged: dict[str, dict[str, bool | None]] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Sheet not found: {p}")
        with p.open("r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
            reader = csv.DictReader(fh, delimiter=delimiter)
            missing = {"interaction_id", "annotator_1_verdict", "annotator_2_verdict"} - set(
                reader.fieldnames or []
            )
            if missing:
                raise HumanCheckError(
                    f"{p.name}: missing columns {sorted(missing)}. Keep the header row as exported."
                )
            for row in reader:
                iid = (row.get("interaction_id") or "").strip()
                if not iid:
                    continue
                rec = merged.setdefault(iid, {a: None for a in ANNOTATORS})
                for a in ANNOTATORS:
                    verdict = _parse_verdict(
                        row.get(f"{a}_verdict"), where=f"{p.name} item {row.get('item_no') or iid}, {a}"
                    )
                    if verdict is None:
                        continue
                    if rec[a] is not None and rec[a] != verdict:
                        raise HumanCheckError(f"{iid}: {a} has both pass and fail across the sheets.")
                    rec[a] = verdict
    return merged


def _kappa(a: list[bool], b: list[bool], *, seed: int, n_boot: int = 2000) -> dict[str, Any]:
    n = len(a)
    if n == 0:
        return {"n": 0, "kappa": None, "ci95": None, "raw_agreement": None}
    raw = sum(x == y for x, y in zip(a, b, strict=True)) / n

    def k(xs: list[bool], ys: list[bool]) -> float | None:
        if len(set(xs) | set(ys)) < 2:
            return None  # everyone gave the same label: kappa is undefined (raw agreement is 100%)
        val = float(cohen_kappa_score([int(x) for x in xs], [int(y) for y in ys]))
        return None if val != val else val

    kappa = k(a, b)
    ci = None
    if kappa is not None and n >= 10:
        rng = random.Random(seed)
        boots: list[float] = []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            kv = k([a[i] for i in idx], [b[i] for i in idx])
            if kv is not None:
                boots.append(kv)
        if len(boots) >= n_boot // 2:
            boots.sort()
            ci = [round(boots[int(0.025 * len(boots))], 4), round(boots[int(0.975 * len(boots)) - 1], 4)]
    return {
        "n": n,
        "kappa": None if kappa is None else round(kappa, 4),
        "ci95": ci,
        "raw_agreement": round(raw, 4),
    }


def _confusion(judge: list[bool], human: list[bool]) -> dict[str, Any]:
    tp = sum(j and h for j, h in zip(judge, human, strict=True))
    fp = sum(j and not h for j, h in zip(judge, human, strict=True))
    fn = sum((not j) and h for j, h in zip(judge, human, strict=True))
    tn = sum((not j) and (not h) for j, h in zip(judge, human, strict=True))
    return {
        "both_fail": tp,
        "judge_fail_human_pass": fp,
        "judge_pass_human_fail": fn,
        "both_pass": tn,
        "judge_fail_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "judge_fail_recall": round(tp / (tp + fn), 4) if tp + fn else None,
    }


def interpret_kappa(k: float | None) -> str:
    if k is None:
        return "not defined"
    for limit, word in (
        (0.0, "poor"),
        (0.2, "slight"),
        (0.4, "fair"),
        (0.6, "moderate"),
        (0.8, "substantial"),
    ):
        if k <= limit:
            return word
    return "almost perfect"


def score_human_check(
    session: Session,
    config: AppConfig,
    sheet_paths: list[str | Path],
) -> dict[str, Any]:
    """Compute real judge-vs-human agreement from filled sheets. Writes nothing to the database."""
    human = read_human_check_sheets(sheet_paths)
    decisions = _latest_judge_decisions(session, config.judge.judge_version)
    unknown = sorted(iid for iid in human if iid not in decisions)
    if unknown and len(unknown) == len(human):
        raise HumanCheckError(
            f"None of the sheet's items have a '{config.judge.judge_version}' judgement in this database. "
            "Use the same --config/--small/--db-url as for the export."
        )
    items = [
        (iid, decisions[iid]["judge_fail"], h)
        for iid, h in sorted(human.items())
        if iid in decisions and not decisions[iid]["judge_error"]
    ]
    seed = config.seed
    report: dict[str, Any] = {
        "source": "real_human_annotation",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "judge_version": config.judge.judge_version,
        "judge_model": config.judge.model_name,
        "sheets": [str(p) for p in sheet_paths],
        "items_in_sheet": len(human),
        "items_not_in_db": len(unknown),
        "filled_by": {a: sum(1 for _, _, h in items if h[a] is not None) for a in ANNOTATORS},
        "pairs": {},
    }
    for a in ANNOTATORS:
        rows = [(j, h[a]) for _, j, h in items if h[a] is not None]
        jd, hd = [r[0] for r in rows], [bool(r[1]) for r in rows]
        report["pairs"][f"judge_vs_{a}"] = {**_kappa(jd, hd, seed=seed), **_confusion(jd, hd)}
    both = [
        (j, bool(h["annotator_1"]), bool(h["annotator_2"]))
        for _, j, h in items
        if None not in (h["annotator_1"], h["annotator_2"])
    ]
    report["pairs"]["annotator_1_vs_annotator_2"] = _kappa(
        [b[1] for b in both], [b[2] for b in both], seed=seed
    )
    agreed = [(j, h1) for j, h1, h2 in both if h1 == h2]
    report["pairs"]["judge_vs_human_consensus"] = {
        **_kappa([a[0] for a in agreed], [a[1] for a in agreed], seed=seed),
        **_confusion([a[0] for a in agreed], [a[1] for a in agreed]),
        "items_where_annotators_disagree": len(both) - len(agreed),
    }
    judged = [j for _, j, _ in items]
    report["judge_fail_rate_on_sheet"] = round(sum(judged) / len(judged), 4) if judged else None
    for a in ANNOTATORS:
        hs = [bool(h[a]) for _, _, h in items if h[a] is not None]
        report[f"{a}_fail_rate"] = round(sum(hs) / len(hs), 4) if hs else None
    logger.info(
        "human_check_scored", extra={"pairs": {k: v.get("kappa") for k, v in report["pairs"].items()}}
    )
    return report


def render_human_check_markdown(report: dict[str, Any]) -> str:
    def f(v: Any) -> str:
        return "N/A" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    def pct(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}%"

    labels = {
        "judge_vs_annotator_1": "Judge vs annotator 1",
        "judge_vs_annotator_2": "Judge vs annotator 2",
        "judge_vs_human_consensus": "Judge vs human consensus (items both annotators agree on)",
        "annotator_1_vs_annotator_2": "Annotator 1 vs annotator 2 (human ceiling)",
    }
    lines = [
        SECTION_START,
        "## Judge vs Real Human Annotation",
        "",
        "Source: **real human annotation** (two people labelled each answer pass or fail without seeing the judge's decision). "
        "This replaces the simulated-annotator agreement in the judge validation section, whose annotators copy the judge.",
        "",
        f"- **Judge**: `{report['judge_model']}`, version `{report['judge_version']}`",
        f"- **Items in sheet**: {report['items_in_sheet']}; filled by annotator 1: {report['filled_by']['annotator_1']}, "
        f"annotator 2: {report['filled_by']['annotator_2']}",
        f"- **Fail rate on these items**: judge {pct(report['judge_fail_rate_on_sheet'])}, annotator 1 "
        f"{pct(report['annotator_1_fail_rate'])}, annotator 2 {pct(report['annotator_2_fail_rate'])}",
        "",
        "| Comparison | n | Cohen's kappa | 95% CI (bootstrap) | Raw agreement | Reading |",
        "|---|---|---|---|---|---|",
    ]
    for key in (
        "judge_vs_human_consensus",
        "judge_vs_annotator_1",
        "judge_vs_annotator_2",
        "annotator_1_vs_annotator_2",
    ):
        p = report["pairs"][key]
        ci = f"{p['ci95'][0]:.2f} to {p['ci95'][1]:.2f}" if p.get("ci95") else "N/A"
        lines.append(
            f"| {labels[key]} | {p['n']} | {f(p['kappa'])} | {ci} | {pct(p['raw_agreement'])} | {interpret_kappa(p['kappa'])} |"
        )
    c = report["pairs"]["judge_vs_human_consensus"]
    lines.extend(
        [
            "",
            f"Against the human consensus: both fail {c['both_fail']}, judge fail but humans pass {c['judge_fail_human_pass']}, "
            f"judge pass but humans fail {c['judge_pass_human_fail']}, both pass {c['both_pass']}. "
            f"Judge fail precision {pct(c['judge_fail_precision'])}, recall {pct(c['judge_fail_recall'])}. "
            f"The annotators disagreed on {c['items_where_annotators_disagree']} items, which are left out of the consensus row.",
            "",
            f"_Generated {report['generated_at']} by `edg calibrate human-import`._",
            SECTION_END,
        ]
    )
    return "\n".join(lines) + "\n"


def write_human_check_section(summary_path: str | Path, section: str) -> Path:
    """Insert or replace the human-check section in a results summary (other content is kept)."""
    p = Path(summary_path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    if SECTION_START in text and SECTION_END in text:
        head, rest = text.split(SECTION_START, 1)
        _, tail = rest.split(SECTION_END, 1)
        text = head + section.rstrip("\n") + tail
    else:
        text = (text.rstrip("\n") + "\n\n" if text else "") + section
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p
