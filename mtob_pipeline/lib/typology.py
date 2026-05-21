from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import GRAMBANK_ENGLISH_ID, GRAMBANK_LANG_IDS_ORDERED


def load_typology_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    p = path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Typology file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError('typology-json must be a JSON object { "Igbo": "...", ... }')
    return {str(k): str(v) for k, v in data.items()}


def _grambank_feature_prompt_block(lang: str, row: pd.Series) -> str:
    desc_mid = (
        "Below is a short summary of the grammatical feature, an explanation of the process "
        "for assigning the feature's\n"
        "code, and examples of the feature from other languages including interlinear glossed text."
    )
    pid = row["Parameter_ID"]
    name = row["Name"]
    desc = row["Description"]
    return f"""
Feature ID: {pid}: {name}
{lang} Value: Code {row[lang]}

English Value: Code {row["English"]}

---
{desc_mid}
---
{desc}
This is the end of the summary for feature {pid}: "{name}".
"""


def _grambank_typology_lists(master_df: pd.DataFrame) -> dict[str, list[str]]:
    cols = master_df.columns.tolist()
    start_idx = cols.index("Parameter_ID") + 1
    end_idx = cols.index("Name")
    languages = cols[start_idx:end_idx]
    out: dict[str, list[str]] = {}
    for lang in languages:
        if lang == "English":
            continue
        blocks: list[str] = []
        for _, row in master_df.iterrows():
            blocks.append(_grambank_feature_prompt_block(lang, row))
        out[lang] = blocks
    return out


def _combine_grambank_typology_strings(prompt_lists: dict[str, list[str]]) -> dict[str, str]:
    separator = "\n___\n"
    final: dict[str, str] = {}
    for lang, prompts_list in prompt_lists.items():
        intro = (
            f"The following typological features describe the grammatical features of {lang} "
            "and English including word order, verbal tense, nominal case, and other language universals. "
            "Each feature is assigned a value that indicates the extent to which the language tends "
            "to exhibit that feature.\n"
        )
        outro = f"---\nThis is the end of the typological feature summary for {lang} and English."
        body = separator.join(prompts_list)
        final[lang] = f"{intro}\n{body}\n\n{outro}"
    return final


def build_typology_map_from_grambank_csv(repo_root: Path) -> dict[str, str]:
    para_path = repo_root / "parameters.csv"
    val_path = repo_root / "values.csv"
    if not para_path.is_file():
        raise FileNotFoundError(f"Missing {para_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing {val_path}")

    para_df = pd.read_csv(para_path)
    val_df = pd.read_csv(val_path)

    first_lang, first_id = GRAMBANK_LANG_IDS_ORDERED[0]
    master_df = (
        val_df[val_df["Language_ID"] == first_id][["Parameter_ID", "Value"]]
        .rename(columns={"Value": first_lang})
        .copy()
    )
    for lang_name, lid in GRAMBANK_LANG_IDS_ORDERED[1:]:
        sub = val_df[val_df["Language_ID"] == lid][["Parameter_ID", "Value"]].rename(
            columns={"Value": lang_name}
        )
        master_df = master_df.merge(sub, on="Parameter_ID", how="left")

    eng_sub = val_df[val_df["Language_ID"] == GRAMBANK_ENGLISH_ID][
        ["Parameter_ID", "Value"]
    ].rename(columns={"Value": "English"})
    master_df = master_df.merge(eng_sub, on="Parameter_ID", how="left")

    master_df = master_df.merge(
        para_df[["ID", "Name", "Description"]].rename(columns={"ID": "Parameter_ID"}),
        on="Parameter_ID",
        how="left",
    )
    hyperlink_pattern = r"\[([^\]]+)\]\([^\)]+\)"
    master_df["Description"] = master_df["Description"].fillna("").astype(str)
    master_df["Description"] = master_df["Description"].str.replace(
        hyperlink_pattern, r"\1", regex=True
    )

    lists_map = _grambank_typology_lists(master_df)
    return _combine_grambank_typology_strings(lists_map)


def resolve_typology_map(cli_path: Path | None, repo_root: Path) -> tuple[dict[str, str], str]:
    if cli_path is not None:
        p = cli_path.expanduser().resolve()
        return load_typology_map(p), str(p)
    for name in ("typology.json", "grambank_typology.json"):
        cand = repo_root / name
        if cand.is_file():
            return load_typology_map(cand), str(cand.resolve())
    built = build_typology_map_from_grambank_csv(repo_root)
    label = f"Grambank CSV: {repo_root / 'parameters.csv'} + {repo_root / 'values.csv'}"
    return built, label
