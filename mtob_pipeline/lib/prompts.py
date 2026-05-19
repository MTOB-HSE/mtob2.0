from __future__ import annotations

from typing import Any

from .constants import (
    COMMON_PROMPT_TAIL,
    SYSTEM_MESSAGE,
    VALID_PROMPT_MODES,
    Direction,
    _TRUNCATE_CHAT_OVERHEAD_TOKENS,
)

_TYPOLOGY_INTRO = "{lang} is a language that exhibits specific typological characteristics. "


def translation_instruction(lang_name: str, sentence: str, direction: Direction) -> str:
    if direction == "en_to_L":
        return f"Translate the following sentence from English to {lang_name}: {sentence}"
    return f"Translate the following sentence from {lang_name} to English: {sentence}"


def translation_answer_block(lang_name: str, sentence: str, direction: Direction) -> str:
    if direction == "en_to_L":
        return f"Now write the translation:\nEnglish: {sentence}\n{lang_name}:"
    return f"Now write the translation:\n{lang_name}: {sentence}\nEnglish:"


def base_translation_prompt(lang_name: str, sentence: str, direction: Direction) -> str:
    return (
        f"{translation_instruction(lang_name, sentence, direction)}\n\n"
        f"{COMMON_PROMPT_TAIL}\n\n"
        f"{translation_answer_block(lang_name, sentence, direction)}"
    )


def generate_zeroshot_prompt(lang_name: str, sentence: str, direction: Direction) -> str:
    return base_translation_prompt(lang_name, sentence, direction)


def generate_typology_prompt(
    lang_name: str,
    sentence: str,
    typological_info: str,
    direction: Direction,
) -> str:
    body = _TYPOLOGY_INTRO.format(lang=lang_name) + base_translation_prompt(
        lang_name, sentence, direction
    )
    return f"""{body}

To help with the translation, here is typological information for both languages:

{typological_info}
"""


def get_parallel_examples_block(
    source_list: list[str], target_list: list[str], lang_name: str
) -> str:
    if not source_list:
        return ""
    block_header = (
        f"To help with the translation, here are some example {lang_name}-English "
        "parallel sentences and words:\n"
    )
    lines = [f"{lang_name}: {src}\nEnglish translation: {tgt}" for src, tgt in zip(source_list, target_list)]
    return block_header + "\n".join(lines)


def generate_paratyp_prompt(
    lang_name: str,
    sentence: str,
    typological_info: str,
    parallel_examples: str,
    direction: Direction,
) -> str:
    return (
        f"{_TYPOLOGY_INTRO.format(lang=lang_name)}"
        f"{translation_instruction(lang_name, sentence, direction)}\n\n"
        f"{COMMON_PROMPT_TAIL}\n\n"
        f"{parallel_examples}\n\n"
        "To help with the translation, here is typological information "
        f"for both languages:\n\n{typological_info}\n\n"
        f"{translation_answer_block(lang_name, sentence, direction)}\n"
    )


def build_prompts_for_mode(
    mode: str,
    lang_name: str,
    direction: Direction,
    test_sentences: list[str],
    train_split: dict[str, list[str]],
    typology_text: str,
) -> list[str]:
    if mode not in VALID_PROMPT_MODES:
        raise ValueError(mode)
    if mode == "zeroshot":
        return [generate_zeroshot_prompt(lang_name, s, direction) for s in test_sentences]
    if mode == "typology":
        return [
            generate_typology_prompt(lang_name, s, typology_text, direction)
            for s in test_sentences
        ]
    if mode == "paratyp":
        para = get_parallel_examples_block(
            train_split["source"],
            train_split["target"],
            lang_name,
        )
        return [
            generate_paratyp_prompt(lang_name, s, typology_text, para, direction)
            for s in test_sentences
        ]
    raise ValueError(mode)


def tiktoken_encoding_for_model(model_name: str) -> Any:
    try:
        import tiktoken
    except ImportError:
        raise SystemExit(
            "--truncate-tokens requires tiktoken: pip install tiktoken"
        ) from None

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def truncate_user_prompt_to_input_budget(
    user_prompt: str,
    *,
    model_name: str,
    max_input_tokens: int,
) -> tuple[str, int, int]:
    enc = tiktoken_encoding_for_model(model_name)
    sys_n = len(enc.encode(SYSTEM_MESSAGE))
    cap = max_input_tokens - sys_n - _TRUNCATE_CHAT_OVERHEAD_TOKENS
    if cap < 4096:
        cap = 4096
    raw_ids = enc.encode(user_prompt)
    before = len(raw_ids)
    if before <= cap:
        return user_prompt, before, before
    return enc.decode(raw_ids[:cap]), before, cap


def truncate_typology_paratyp_prompts(
    prompts: list[str],
    prompt_mode: str,
    *,
    enabled: bool,
    model_name: str,
    max_input_tokens: int,
) -> tuple[list[str], int]:
    if not enabled or prompt_mode == "zeroshot":
        return prompts, 0
    if prompt_mode not in ("typology", "paratyp"):
        return prompts, 0
    out: list[str] = []
    n_cut = 0
    for p in prompts:
        t, before, after = truncate_user_prompt_to_input_budget(
            p, model_name=model_name, max_input_tokens=max_input_tokens
        )
        if after < before:
            n_cut += 1
        out.append(t)
    return out, n_cut
