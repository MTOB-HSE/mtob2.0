from __future__ import annotations

from typing import Literal

Direction = Literal["en_to_L", "L_to_en"]

VALID_PROMPT_MODES: frozenset[str] = frozenset({"zeroshot", "typology", "paratyp"})

GRAMBANK_LANG_IDS_ORDERED: list[tuple[str, str]] = [
    ("Igbo", "nucl1417"),
    ("Basque", "basq1248"),
    ("Georgian", "nucl1302"),
    ("Chamorro", "cham1312"),
    ("Korean", "kore1280"),
]
GRAMBANK_ENGLISH_ID = "stan1293"

ENG_SPEC = ("eng", "Latn")

PROMPT_MODE_SHORT: dict[str, str] = {
    "zeroshot": "zs",
    "typology": "typ",
    "paratyp": "paratyp",
}

SYSTEM_MESSAGE = "You are a professional translator and only output the result."

COMMON_PROMPT_TAIL = """Now write the translation. If you are not sure what the translation should be, then give your best guess. Do not say that you do not speak the source language. Do not say you do not have enough information, you must make a guess. If your translation is wrong, that is fine, but you have to provide a translation.
Your translation must be on the first line of your response, with no other text before the translation. Only explain your reasoning after providing the translation.
It is crucial that you only give the translation on the first line of your response, otherwise you will fail."""

_TRUNCATE_CHAT_OVERHEAD_TOKENS = 512
