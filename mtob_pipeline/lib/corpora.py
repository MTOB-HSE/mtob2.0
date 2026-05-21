from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import load_dataset

from .config import (
    DEFAULT_LANGUAGES_YAML,
    LanguageSpec,
    ParallelCorpus,
    validate_language_spec,
)
from .constants import ENG_SPEC

CorpusLoadFn = Callable[[LanguageSpec, dict[str, Any]], ParallelCorpus]


def _sort_extract_flores_text(ds, iso_639_3: str, iso_15924: str) -> list[str]:
    sub = ds.filter(lambda x: x["iso_639_3"] == iso_639_3 and x["iso_15924"] == iso_15924)
    sub = sub.sort("id")
    return list(sub["text"])


def load_flores_aligned(
    *,
    split: str = "devtest",
    lang_specs: dict[str, tuple[str, str]],
) -> tuple[list[str], dict[str, list[str]]]:
    ds = load_dataset("openlanguagedata/flores_plus", split=split)
    english = _sort_extract_flores_text(ds, *ENG_SPEC)
    per_lang: dict[str, list[str]] = {}
    for lang_name, spec in lang_specs.items():
        per_lang[lang_name] = _sort_extract_flores_text(ds, *spec)
        if len(per_lang[lang_name]) != len(english):
            raise ValueError(
                f"FLORES+ length mismatch: EN={len(english)}, "
                f"{lang_name}={len(per_lang[lang_name])}"
            )
    return english, per_lang


def load_book_csv_parallel(
    path: Path,
    *,
    en_column: str = "translation",
    lang_column: str = "object_language",
) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    if en_column not in df.columns or lang_column not in df.columns:
        raise ValueError(
            f"CSV {path} must contain columns {en_column!r} and {lang_column!r}; "
            f"got: {list(df.columns)}"
        )
    en_list = [str(x).strip() for x in df[en_column].tolist()]
    lang_list = [str(x).strip() for x in df[lang_column].tolist()]
    if len(en_list) != len(lang_list):
        raise ValueError(f"CSV {path}: column length mismatch")
    return en_list, lang_list


def load_smol_pairs(config_name: str) -> list[tuple[str, str]] | None:
    try:
        ds = load_dataset(
            "google/smol",
            config_name,
            split="train",
            trust_remote_code=True,
        )
    except Exception:
        return None
    out: list[tuple[str, str]] = []
    for i in range(len(ds)):
        ex = ds[i]
        src = ex.get("src")
        trg = ex.get("trg")
        if trg is None and ex.get("trgs") is not None:
            t = ex["trgs"]
            trg = t[0] if isinstance(t, (list, tuple)) else t
        if not src or not trg:
            continue
        sl, tl = ex.get("sl", ""), ex.get("tl", "")
        if sl == "en" and tl not in ("", "en"):
            out.append((src, trg))
        elif tl == "en" and sl not in ("", "en"):
            out.append((trg, src))
        else:
            out.append((src, trg))
    return out


def _tatoeba_split_to_lists(ds) -> tuple[list[str], list[str]]:
    en_list, tgt_list = [], []
    for i in range(len(ds)):
        ex = ds[i]
        s = str(ex["sourceString"]).strip()
        t = str(ex["targetString"]).strip()
        sl = str(ex.get("sourceLang", "")).lower()
        tl = str(ex.get("targetlang", "")).lower()
        if "eng" in sl and "cha" in tl:
            en_list.append(s)
            tgt_list.append(t)
        elif "cha" in sl and "eng" in tl:
            tgt_list.append(s)
            en_list.append(t)
        else:
            print(
                f"  skipping pair with unexpected languages: {sl!r} -> {tl!r}",
                file=sys.stderr,
            )
    return en_list, tgt_list


def _extend_pairs_from_mt_train(
    ens: list[str],
    tgts: list[str],
    *,
    pair: str = "cha-eng",
) -> None:
    try:
        ds = load_dataset(
            "Helsinki-NLP/tatoeba_mt_train",
            pair,
            split="train",
            trust_remote_code=True,
        )
    except Exception as ex:
        print(f"  tatoeba_mt_train {pair} train unavailable: {ex}", file=sys.stderr)
        return
    n0 = len(ens)
    for i in range(len(ds)):
        ex = ds[i]
        s = str(ex["source_text"]).strip()
        t = str(ex["target_text"]).strip()
        sl = str(ex.get("source_lang", "")).lower()
        tl = str(ex.get("target_lang", "")).lower()
        if sl == "cha" and tl == "eng":
            ens.append(t)
            tgts.append(s)
        elif sl == "eng" and tl == "cha":
            ens.append(s)
            tgts.append(t)
        else:
            print(f"  skipping mt_train pair: {sl!r} -> {tl!r}", file=sys.stderr)
    print(f"  tatoeba_mt_train {pair} train: +{len(ens) - n0} pairs")


def load_tatoeba_mt_splits(
    *,
    language_pair: str,
    splits: list[str],
    include_mt_train: bool,
) -> tuple[list[str], list[str]]:
    ens: list[str] = []
    tgts: list[str] = []
    for sp in splits:
        sp = sp.strip()
        if not sp:
            continue
        try:
            ds = load_dataset(
                "Helsinki-NLP/tatoeba_mt",
                language_pair=language_pair,
                split=sp,
                trust_remote_code=True,
            )
        except Exception as ex:
            print(
                f"  Tatoeba {language_pair} split={sp!r} unavailable: {ex}",
                file=sys.stderr,
            )
            continue
        e, t = _tatoeba_split_to_lists(ds)
        print(f"  Tatoeba {language_pair} {sp}: +{len(e)} pairs")
        ens.extend(e)
        tgts.extend(t)

    if include_mt_train:
        _extend_pairs_from_mt_train(ens, tgts, pair=language_pair)

    seen: set[tuple[str, str]] = set()
    out_e: list[str] = []
    out_t: list[str] = []
    for e, tgt in zip(ens, tgts):
        key = (e, tgt)
        if key in seen:
            continue
        seen.add(key)
        out_e.append(e)
        out_t.append(tgt)
    return out_e, out_t


def load_tatoeba_mt_parallel(
    smol_config: str | None,
    *,
    language_pair: str,
    tatoeba_splits: list[str],
    include_mt_train: bool,
) -> tuple[list[str], list[str]]:
    if smol_config:
        pairs = load_smol_pairs(smol_config)
        if pairs:
            en_list, tgt_list = zip(*pairs)
            return list(en_list), list(tgt_list)

    return load_tatoeba_mt_splits(
        language_pair=language_pair,
        splits=tatoeba_splits,
        include_mt_train=include_mt_train,
    )


def _load_tatoeba_mt(spec: LanguageSpec, overrides: dict[str, Any]) -> ParallelCorpus:
    pair = str(spec.corpus.get("pair", "cha-eng"))
    splits = overrides.get("tatoeba_splits") or spec.corpus.get("splits") or ["test"]
    include_mt_train = overrides.get(
        "include_mt_train", spec.corpus.get("mt_train", True)
    )
    smol_cfg = overrides.get("smol_config")
    if smol_cfg is None:
        env_key = spec.corpus.get("smol_env")
        if env_key:
            smol_cfg = os.environ.get(str(env_key))
    if smol_cfg is None:
        smol_cfg = spec.corpus.get("smol_config")
    english, target = load_tatoeba_mt_parallel(
        smol_cfg,
        language_pair=pair,
        tatoeba_splits=[str(s) for s in splits],
        include_mt_train=bool(include_mt_train),
    )
    return ParallelCorpus(
        english=english,
        target=target,
        source=f"tatoeba_mt:{pair}",
        lang_name=spec.name,
    )


def _load_csv(spec: LanguageSpec, _overrides: dict[str, Any]) -> ParallelCorpus:
    path = Path(str(spec.corpus["path"])).expanduser()
    en_col = str(spec.corpus.get("en_column", "translation"))
    lang_col = str(spec.corpus.get("lang_column", "object_language"))
    english, target = load_book_csv_parallel(path, en_column=en_col, lang_column=lang_col)
    return ParallelCorpus(
        english=english,
        target=target,
        source=f"csv:{path.name}",
        lang_name=spec.name,
    )


CORPUS_LOADERS: dict[str, CorpusLoadFn] = {
    "tatoeba_mt": _load_tatoeba_mt,
    "csv": _load_csv,
}


def load_flores_batch(
    specs: list[LanguageSpec],
    *,
    flores_split: str,
) -> dict[str, ParallelCorpus]:
    lang_specs = {s.name: (s.corpus["iso_639_3"], s.corpus["script"]) for s in specs}
    split = str(specs[0].corpus.get("split", flores_split)) if specs else flores_split
    print("Loading FLORES+ ...")
    english, per_lang = load_flores_aligned(split=split, lang_specs=lang_specs)
    out: dict[str, ParallelCorpus] = {}
    for name, lines in per_lang.items():
        out[name] = ParallelCorpus(
            english=english,
            target=lines,
            source=f"flores:{split}",
            lang_name=name,
        )
    return out


def load_corpus_for_spec(spec: LanguageSpec, overrides: dict[str, Any]) -> ParallelCorpus:
    ctype = str(spec.corpus.get("type", ""))
    loader = CORPUS_LOADERS.get(ctype)
    if loader is None:
        raise ValueError(f"unknown corpus type: {ctype!r}")
    return loader(spec, overrides)


def load_language_specs(config_path: Path | None = None) -> list[LanguageSpec]:
    path = config_path or DEFAULT_LANGUAGES_YAML
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("languages") or []
    specs = [
        LanguageSpec(
            name=str(item["name"]),
            grambank_id=str(item.get("grambank_id", "")),
            corpus=dict(item.get("corpus") or {}),
        )
        for item in items
    ]
    for spec in specs:
        validate_language_spec(spec)
    return specs


def load_aligned_corpora(
    specs: list[LanguageSpec],
    *,
    only_langs: set[str] | None = None,
    flores_split: str = "devtest",
    tatoeba_overrides: dict | None = None,
) -> dict[str, ParallelCorpus]:
    overrides = tatoeba_overrides or {}
    aligned: dict[str, ParallelCorpus] = {}
    active = [s for s in specs if only_langs is None or s.name in only_langs]
    if not active:
        return aligned

    flores_specs = [s for s in active if s.corpus.get("type") == "flores"]
    if flores_specs:
        aligned.update(load_flores_batch(flores_specs, flores_split=flores_split))
        for spec in flores_specs:
            if spec.name in aligned:
                print(f"  {aligned[spec.name].lang_name} (FLORES+): N = {len(aligned[spec.name])} pairs")

    for spec in active:
        ctype = spec.corpus.get("type")
        if ctype == "flores":
            continue
        if ctype not in CORPUS_LOADERS:
            print(f"  Skipping {spec.name}: unknown corpus.type={ctype!r}")
            continue
        print(f"Loading {spec.name} ({ctype}) ...")
        try:
            corpus = load_corpus_for_spec(spec, overrides)
            aligned[spec.name] = corpus
            label = {"tatoeba_mt": "Tatoeba/SMOL", "csv": "CSV"}.get(ctype, ctype)
            print(f"  {corpus.lang_name} ({label}): N = {len(corpus)} pairs")
        except Exception as exc:
            print(f"  {spec.name} unavailable: {exc}", file=sys.stderr)

    return aligned
