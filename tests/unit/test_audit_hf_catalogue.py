"""Regression test: the HF Amazon-Reviews-2023 catalogue could never load on current `datasets`.

`McAuley-Lab/Amazon-Reviews-2023` is a script-based dataset and `datasets>=4` no longer runs
dataset scripts (and rejects `trust_remote_code`), so the loader always failed and silently fell
back to the synthetic catalogue. It must read the raw metadata JSONL file directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from edg.world import catalogue


def test_amazon_catalogue_reads_raw_jsonl_without_dataset_script(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    rows = [
        {"title": f"USB-C Charger {i}", "price": "19.99", "main_category": "Electronics", "details": {"Brand": "X"}}
        for i in range(5)
    ]

    def fake_load_dataset(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        calls.append((args, kwargs))
        if args and args[0] != "json":
            raise RuntimeError("Dataset scripts are no longer supported")
        assert "trust_remote_code" not in kwargs
        return iter(rows)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    products = catalogue.load_catalogue(size=5, seed=42, prefer_hf=True)
    assert [p.source for p in products] == ["amazon_catalogue_sample"] * 5
    assert products[0].title == "USB-C Charger 0"
    json_calls = [c for c in calls if c[0] and c[0][0] == "json"]
    assert json_calls, "must load the raw JSONL via the generic json builder"
    data_files = str(json_calls[0][1].get("data_files"))
    assert "McAuley-Lab/Amazon-Reviews-2023" in data_files and "meta_Electronics.jsonl" in data_files
    assert json_calls[0][1].get("streaming") is True
