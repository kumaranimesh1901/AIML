"""Ollama check shown in the sidebar of every page and on the Chat Assistant page: is the server
running, are the v1 and v2 assistants and the judge pulled, and a Download button for any that
is missing (it pulls through Ollama's own /api/pull, so the models are the real ones)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from edg.config import AppConfig
from edg.sut.chat import MODEL_SIZES, check_models, pull_model


@st.cache_data(ttl=15, show_spinner=False)
def _cached_check(_config: AppConfig, key: str) -> dict[str, Any]:  # noqa: ARG001 - key busts the cache
    return check_models(_config)


def model_status(config: AppConfig) -> dict[str, Any]:
    return _cached_check(config, f"{config.llm.ollama_base_url}|{config.sut.versions}|{config.judge.model_name}")


def _download(config: AppConfig, model: str) -> None:
    bar = st.progress(0.0, text=f"Downloading {model}…")
    try:
        for msg in pull_model(config.llm.ollama_base_url, model):
            total, done = msg.get("total"), msg.get("completed")
            status = str(msg.get("status", ""))
            if total and done is not None:
                bar.progress(min(done / total, 1.0), text=f"{model}: {done / 1e9:.2f} of {total / 1e9:.2f} GB")
            else:
                bar.progress(1.0 if status == "success" else 0.0, text=f"{model}: {status}")
    except Exception as exc:  # noqa: BLE001 - show any network or Ollama error to the user
        bar.empty()
        st.error(f"Couldn't download {model}: {exc}. You can also run `ollama pull {model}` in a terminal.")
        return
    bar.empty()
    st.toast(f"{model} is ready", icon=":material/check_circle:")
    _cached_check.clear()
    st.rerun()


def render_model_status(config: AppConfig, *, compact: bool) -> bool:
    """Show the check; returns True when Ollama runs and every required model is there.

    `compact` is the sidebar version (an expander that opens itself when something is missing)."""
    if config.llm.provider == "fake":
        st.caption("LLM provider is `fake` (tests only): the live chat is off.")
        return False
    status = model_status(config)
    missing = [m for m in status["models"] if not m["installed"]]
    ready = status["reachable"] and not missing
    label = "Local models: ready" if ready else "Local models: action needed"
    box = st.expander(label, expanded=not ready, icon=":material/memory:") if compact else st.container(border=True)
    with box:
        if not status["reachable"]:
            st.markdown(f":red[**Ollama is not running**] at `{status['base_url']}`.")
            st.markdown(
                "Open the Ollama app (or run `ollama serve` in a terminal), then press *Check again*. "
                "Install it from [ollama.com/download](https://ollama.com/download) if you don't have it."
            )
        else:
            st.markdown(":green[**Ollama is running.**]")
        for m in status["models"]:
            mark = ":green[:material/check_circle:]" if m["installed"] else ":red[:material/cancel:]"
            line = f"{mark} `{m['model']}` · {m['role']}"
            if not m["installed"] and status["reachable"]:
                size = MODEL_SIZES.get(m["model"])
                cols = st.columns([3, 2]) if not compact else None
                if cols:
                    cols[0].markdown(line + " · **missing**")
                    target = cols[1]
                else:
                    st.markdown(line + " · **missing**")
                    target = st
                if target.button(
                    f"Download{f' ({size})' if size else ''}",
                    key=f"pull_{'side' if compact else 'page'}_{m['model']}",
                    icon=":material/download:",
                    use_container_width=True,
                ):
                    _download(config, m["model"])
            else:
                st.markdown(line)
        if not ready and st.button(
            "Check again", key=f"recheck_{'side' if compact else 'page'}", icon=":material/refresh:"
        ):
            _cached_check.clear()
            st.rerun()
    return ready
