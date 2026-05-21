from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelinePaths
from .constants import PROMPT_MODE_SHORT, SYSTEM_MESSAGE, Direction


@dataclass
class BatchManifestEntry:
    lang: str
    direction: Direction
    prompt_mode: str
    batch_jsonl: str
    test_csv: str
    num_requests: int


def parse_custom_id(cid: str) -> tuple[str, str, int] | None:
    parts = cid.split("::")
    if len(parts) != 4:
        return None
    lang, direction, _ptag, idx_s = parts
    return lang, direction, int(idx_s)


def write_batch_jsonl_from_prompts(
    path: Path,
    *,
    lang_name: str,
    direction: Direction,
    prompt_mode: str,
    prompts: list[str],
    model: str,
    temperature: float,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = PROMPT_MODE_SHORT.get(prompt_mode, prompt_mode)
    n = len(prompts)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            entry = {
                "custom_id": f"{lang_name}::{direction}::{pm}::{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompts[i]},
                    ],
                    "temperature": temperature,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return n


def save_test_table(
    path: Path, split: dict[str, list[str]], *, max_rows: int | None = None
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    src = split["source"]
    ref = split["target"]
    if max_rows is not None:
        src = src[:max_rows]
        ref = ref[:max_rows]
    n = len(src)
    pd.DataFrame({"Index": range(n), "Source": src, "Reference": ref}).to_csv(
        path, index=False
    )
    return n


def load_manifest_entries(
    man_path: Path,
    *,
    prompt_modes_filter: frozenset[str] | None = None,
) -> list[BatchManifestEntry]:
    data = json.loads(man_path.read_text(encoding="utf-8"))
    entries_raw = data.get("entries") or []
    manifest: list[BatchManifestEntry] = []
    for e in entries_raw:
        try:
            pm = e.get("prompt_mode", "zeroshot")
            if prompt_modes_filter is not None and pm not in prompt_modes_filter:
                continue
            manifest.append(
                BatchManifestEntry(
                    lang=e["lang"],
                    direction=e["direction"],
                    prompt_mode=pm,
                    batch_jsonl=e["batch_jsonl"],
                    test_csv=e["test_csv"],
                    num_requests=int(e["num_requests"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return manifest


def write_manifest(
    paths: PipelinePaths,
    *,
    entries: list[BatchManifestEntry],
    summary_meta: dict[str, Any],
) -> Path:
    man_path = paths.manifest
    summary = {**summary_meta, "entries": [m.__dict__ for m in entries]}
    man_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return man_path
