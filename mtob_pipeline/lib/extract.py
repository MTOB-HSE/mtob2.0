"""Extract parallel examples from grammar PDFs via OpenAI Responses API."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm
import fitz

LEAK = re.compile(
    r"\b(AGR|PROG|COMP|UNM|SG|PL|PST|PRF|ERG|ABS|DAT|GEN|LOC|INSTR|NMLZ)\b"
)

DEFAULT_MAX_CHARS = 36000
DEFAULT_OVERLAP_PAGES = 1
DEFAULT_EXTRACT_MODEL = "gpt-5.2"
DEFAULT_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parent / "prompts" / "extract_template.txt"
)


@dataclass(frozen=True)
class GrammarSpec:
    id: str
    language: str
    book_title: str
    pdf: Path
    output_csv: str
    system_prompt_file: Path | None = None
    output_jsonl: str | None = None

    @property
    def csv_path(self) -> Path:
        return Path(self.output_csv)

    @property
    def jsonl_path(self) -> Path:
        if self.output_jsonl:
            return Path(self.output_jsonl)
        return self.csv_path.with_suffix(".jsonl")


@dataclass
class ExtractConfig:
    grammars: list[GrammarSpec]
    output_dir: Path
    prompt_template: Path = DEFAULT_PROMPT_TEMPLATE
    model: str = DEFAULT_EXTRACT_MODEL
    max_chars: int = DEFAULT_MAX_CHARS
    overlap_pages: int = DEFAULT_OVERLAP_PAGES
    only_grammars: frozenset[str] | None = None


def chunk_pages(
    doc: Any,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_pages: int = DEFAULT_OVERLAP_PAGES,
) -> Iterator[tuple[int, int, str]]:
    """Yield (page_start, page_end, chunk_text) with 1-indexed page labels."""
    pages_text = [doc.load_page(i).get_text("text") or "" for i in range(doc.page_count)]
    i = 0
    while i < doc.page_count:
        start = i
        end = i
        buff = ""
        while end < doc.page_count and len(buff) + len(pages_text[end]) < max_chars:
            buff += f"\n\n=== PAGE {end + 1} ===\n" + pages_text[end]
            end += 1

        ov_start = max(0, start - overlap_pages)
        ov_buff = ""
        for p in range(ov_start, start):
            ov_buff += f"\n\n=== PAGE {p + 1} ===\n" + pages_text[p]

        yield ov_start + 1, end, ov_buff + buff
        i = end


def parse_json_list(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    a = text.find("[")
    b = text.rfind("]")
    if a != -1 and b != -1 and b > a:
        return json.loads(text[a : b + 1])
    raise ValueError("Model did not return a JSON list.")


def should_skip(csv_path: Path, *, force: bool) -> bool:
    return not force and csv_path.is_file()


def _resolve_path(path: Path, base: Path) -> Path:
    p = path.expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def load_prompt_template(path: Path, base: Path) -> str:
    full = _resolve_path(path, base)
    return full.read_text(encoding="utf-8")


def render_extract_prompt(
    template: str,
    *,
    book_title: str,
    language: str,
) -> str:
    """Fill {book_title} and {language} placeholders in the extract prompt template."""
    return template.format(book_title=book_title, language=language).strip()


def system_prompt_for_grammar(
    spec: GrammarSpec,
    *,
    template: str,
    base: Path,
) -> str:
    if spec.system_prompt_file is not None:
        return load_prompt_template(spec.system_prompt_file, base).strip()
    return render_extract_prompt(
        template, book_title=spec.book_title, language=spec.language
    )


def load_extract_config(config_path: Path, base: Path | None = None) -> ExtractConfig:
    base = base or config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    grammars_cfg = raw.get("grammars_config")
    if grammars_cfg:
        grammars_path = _resolve_path(Path(grammars_cfg), base)
    else:
        grammars_path = config_path.parent / "grammars.yaml"

    gdata = yaml.safe_load(grammars_path.read_text(encoding="utf-8")) or {}
    output_dir = _resolve_path(Path(raw.get("output_dir", ".")), base)
    prompt_template = _resolve_path(
        Path(raw.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)), base
    )

    only_raw = raw.get("only_grammars")
    only_grammars = frozenset(str(x) for x in only_raw) if only_raw else None

    grammars: list[GrammarSpec] = []
    for entry in gdata.get("grammars", []):
        gid = str(entry["id"])
        pdf = _resolve_path(Path(entry["pdf"]), base)
        out_csv = str(entry["output_csv"])
        out_jsonl = entry.get("output_jsonl")
        language = str(entry.get("language", gid))
        book_title = str(entry.get("book_title", language))
        prompt_override = entry.get("system_prompt_file")
        grammars.append(
            GrammarSpec(
                id=gid,
                language=language,
                book_title=book_title,
                pdf=pdf,
                output_csv=out_csv,
                system_prompt_file=(
                    _resolve_path(Path(prompt_override), base)
                    if prompt_override
                    else None
                ),
                output_jsonl=str(out_jsonl) if out_jsonl else None,
            )
        )

    return ExtractConfig(
        grammars=grammars,
        output_dir=output_dir,
        prompt_template=prompt_template,
        model=str(raw.get("model", DEFAULT_EXTRACT_MODEL)),
        max_chars=int(raw.get("max_chars", DEFAULT_MAX_CHARS)),
        overlap_pages=int(raw.get("overlap_pages", DEFAULT_OVERLAP_PAGES)),
        only_grammars=only_grammars,
    )


def _require_fitz() -> Any:
    if fitz is None:
        raise ImportError("pip install pymupdf (required for PDF extraction)")
    return fitz


def count_chunks(pdf_path: Path, *, max_chars: int, overlap_pages: int) -> int:
    doc = _require_fitz().open(pdf_path)
    try:
        return sum(1 for _ in chunk_pages(doc, max_chars=max_chars, overlap_pages=overlap_pages))
    finally:
        doc.close()


def extract_grammar(
    spec: GrammarSpec,
    *,
    output_dir: Path,
    model: str,
    max_chars: int,
    overlap_pages: int,
    base: Path,
    prompt_template: str,
    client: Any,
    force: bool = False,
    dry_run: bool = False,
) -> Path | None:
    csv_path = _resolve_path(Path(spec.output_csv), output_dir)
    jsonl_path = _resolve_path(spec.jsonl_path, output_dir)

    if not spec.pdf.is_file():
        print(f"[{spec.id}] PDF not found: {spec.pdf}", file=sys.stderr)
        return None

    if dry_run:
        n = count_chunks(spec.pdf, max_chars=max_chars, overlap_pages=overlap_pages)
        skip_note = " (would skip: CSV exists)" if should_skip(csv_path, force=force) else ""
        print(f"[{spec.id}] dry-run: pdf={spec.pdf} chunks={n} -> {csv_path}{skip_note}")
        return None

    if should_skip(csv_path, force=force):
        print(f"[{spec.id}] Skip (exists): {csv_path}")
        return csv_path

    system_prompt = system_prompt_for_grammar(
        spec, template=prompt_template, base=base
    )
    doc = _require_fitz().open(spec.pdf)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, str, str]] = set()

    try:
        chunks = list(
            chunk_pages(doc, max_chars=max_chars, overlap_pages=overlap_pages)
        )
        for page_start, page_end, chunk_text in tqdm(
            chunks,
            desc=f"Extract {spec.id}",
            unit="chunk",
        ):
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text},
                ],
            )
            try:
                items = parse_json_list(resp.output_text)
            except (ValueError, json.JSONDecodeError) as e:
                print(
                    f"[{spec.id}] chunk {page_start}-{page_end}: parse error: {e}",
                    file=sys.stderr,
                )
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue
                obj = (it.get("object_language") or "").strip()
                trn = (it.get("translation") or "").strip()
                if not obj or not trn:
                    continue
                if LEAK.search(obj):
                    continue
                key = (it.get("type"), obj, trn)
                if key in seen:
                    continue
                seen.add(key)
                if not it.get("page_hint"):
                    it["page_hint"] = f"{page_start}-{page_end}"
                rows.append(it)
    finally:
        doc.close()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[{spec.id}] Extracted {len(df)} rows -> {csv_path}, {jsonl_path}")
    return csv_path


def run_extract(
    cfg: ExtractConfig,
    *,
    config_path: Path,
    only: frozenset[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    base = config_path.parent.resolve()
    filter_ids = only or cfg.only_grammars
    specs = cfg.grammars
    if filter_ids:
        specs = [g for g in specs if g.id in filter_ids]
        missing = filter_ids - {g.id for g in specs}
        if missing:
            print(f"Unknown grammar id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    if not specs:
        print("No grammars to process.", file=sys.stderr)
        return 2

    template_text = load_prompt_template(cfg.prompt_template, base)
    client = None if dry_run else OpenAI()
    errors = 0
    for spec in specs:
        result = extract_grammar(
            spec,
            output_dir=cfg.output_dir,
            model=cfg.model,
            max_chars=cfg.max_chars,
            overlap_pages=cfg.overlap_pages,
            base=base,
            prompt_template=template_text,
            client=client,
            force=force,
            dry_run=dry_run,
        )
        if result is None and not dry_run:
            errors += 1
    return 1 if errors else 0
