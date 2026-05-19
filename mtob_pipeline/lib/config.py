from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_LANGUAGES_YAML = PACKAGE_DIR / "config" / "languages.yaml"
DEFAULT_OUTPUT_DIR = Path("mtob_run")

_PATH_KEYS = frozenset({"output_dir", "typology_json", "languages_config", "test_splits_dir"})
_LIST_KEYS = frozenset({"prompt_modes", "only_langs", "exclude_langs", "tatoeba_splits"})
_KNOWN_CORPUS_TYPES = frozenset({"flores", "tatoeba_mt", "csv"})


@dataclass
class LanguageSpec:
    name: str
    grambank_id: str
    corpus: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelCorpus:
    english: list[str]
    target: list[str]
    source: str
    lang_name: str

    def __len__(self) -> int:
        return len(self.english)

    def __post_init__(self) -> None:
        if len(self.english) != len(self.target):
            raise ValueError(
                f"{self.lang_name}: english/target length mismatch "
                f"({len(self.english)} vs {len(self.target)})"
            )


class PipelinePaths:
    def __init__(self, output_dir: Path | str) -> None:
        self.root = Path(output_dir).expanduser().resolve()
        self.batches = self.root / "batches"
        self.test_splits = self.root / "test_splits"
        self.batch_results = self.root / "batch_results"
        self.metrics = self.root / "metrics"
        self.manifest = self.root / "manifest.json"
        self.submitted_batch_ids = self.root / "submitted_batch_ids.json"

    def setup_directories(self) -> None:
        for directory in (
            self.batches,
            self.test_splits,
            self.batch_results,
            self.metrics,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _coerce_run_value(key: str, value: Any) -> Any:
    if key == "truncate_max_input_tokens":
        return None if value is None else int(value)
    if key in _PATH_KEYS:
        return Path(value) if value else None
    if key in _LIST_KEYS:
        if not isinstance(value, list):
            raise ValueError(f"run.yaml {key} must be a list")
        return [str(x).strip() for x in value if str(x).strip()]
    return value


def normalize_run_dict(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = {k: _coerce_run_value(k, v) for k, v in raw.items()}
    if env_model := os.environ.get("OPENAI_BATCH_MODEL"):
        cfg["model"] = env_model
    if env_smol := os.environ.get("TATOEBA_SMOL_CONFIG"):
        cfg["smol_config"] = env_smol
    return cfg


def load_run_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("run.yaml must be a mapping")
    return normalize_run_dict(raw)


def validate_language_spec(spec: LanguageSpec) -> None:
    if not str(spec.name).strip():
        raise ValueError("language spec: name is required")

    corpus = spec.corpus
    ctype = corpus.get("type")
    if not ctype:
        raise ValueError(f"{spec.name}: corpus.type is required")
    if ctype not in _KNOWN_CORPUS_TYPES:
        raise ValueError(f"{spec.name}: unknown corpus.type {ctype!r}")

    if ctype == "flores":
        for key in ("iso_639_3", "script"):
            if not corpus.get(key):
                raise ValueError(f"{spec.name}: corpus.{key} is required for flores")
    elif ctype == "tatoeba_mt":
        if not corpus.get("pair"):
            raise ValueError(f"{spec.name}: corpus.pair is required for tatoeba_mt")
    elif ctype == "csv":
        if not corpus.get("path"):
            raise ValueError(f"{spec.name}: corpus.path is required for csv")

    if not str(spec.grambank_id).strip():
        warnings.warn(
            f"{spec.name}: empty grambank_id (typology/paratyp prompts may be empty)",
            stacklevel=2,
        )
