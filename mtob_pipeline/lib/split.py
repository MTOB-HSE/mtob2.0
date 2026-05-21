from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import Direction


def split_pairs_train_test(
    en_texts: list[str],
    lang_texts: list[str],
    *,
    test_size: float,
    random_state: int,
    full_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame({"object_language": lang_texts, "translation": en_texts})
    if full_test:
        empty = df.iloc[0:0].copy()
        return empty, df.copy()

    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    is_single_word = test["object_language"].str.strip().str.contains(r"^\S+$")
    words_to_move = test[is_single_word]
    test = test[~is_single_word].copy()
    train = pd.concat([train, words_to_move], ignore_index=True)
    return train, test


def train_df_excluding_test(
    en_texts: list[str],
    lang_texts: list[str],
    test_source: list[str],
    test_target: list[str],
) -> pd.DataFrame:
    """Train rows = full parallel corpus minus fixed test pairs (both directions)."""
    test_keys = {(str(s).strip(), str(t).strip()) for s, t in zip(test_source, test_target)}
    rows: list[dict[str, str]] = []
    for en, lang in zip(en_texts, lang_texts):
        en_s, lang_s = str(en).strip(), str(lang).strip()
        if (en_s, lang_s) in test_keys or (lang_s, en_s) in test_keys:
            continue
        rows.append({"object_language": lang_s, "translation": en_s})
    return pd.DataFrame(rows)


def load_frozen_test_split(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    for col in ("Source", "Reference"):
        if col not in df.columns:
            raise ValueError(
                f"{path}: expected columns Source, Reference; got {list(df.columns)}"
            )
    return {
        "source": [str(x) for x in df["Source"].tolist()],
        "target": [str(x) for x in df["Reference"].tolist()],
    }


def corpus_split_for_direction(
    train: pd.DataFrame, test: pd.DataFrame, direction: Direction
) -> dict[str, dict[str, list[str]]]:
    if direction == "en_to_L":
        return {
            "train": {
                "source": train["translation"].tolist(),
                "target": train["object_language"].tolist(),
            },
            "test": {
                "source": test["translation"].tolist(),
                "target": test["object_language"].tolist(),
            },
        }
    return {
        "train": {
            "source": train["object_language"].tolist(),
            "target": train["translation"].tolist(),
        },
        "test": {
            "source": test["object_language"].tolist(),
            "target": test["translation"].tolist(),
        },
    }
