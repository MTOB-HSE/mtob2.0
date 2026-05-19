from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .batch import (
    BatchManifestEntry,
    save_test_table,
    write_batch_jsonl_from_prompts,
    write_manifest,
)
from .config import DEFAULT_LANGUAGES_YAML, REPO_ROOT, PipelinePaths, load_run_config
from .constants import VALID_PROMPT_MODES, Direction
from .corpora import load_aligned_corpora, load_language_specs
from .prompts import build_prompts_for_mode, truncate_typology_paratyp_prompts
from .split import (
    corpus_split_for_direction,
    load_frozen_test_split,
    split_pairs_train_test,
    train_df_excluding_test,
)
from .typology import resolve_typology_map


def load_config(run_config: Path, output_dir: Path | None) -> dict[str, Any]:
    cfg = load_run_config(run_config)
    if output_dir is not None:
        cfg["output_dir"] = Path(output_dir).expanduser().resolve()
    return cfg


def prepare(cfg: dict[str, Any], dry_run: bool) -> int:
    prompt_modes_list = cfg["prompt_modes"]
    for m in prompt_modes_list:
        if m not in VALID_PROMPT_MODES:
            print(f"Unknown mode in prompt_modes: {m}", file=sys.stderr)
            return 2

    typology_map: dict[str, str] = {}
    typology_source_label: str | None = None
    if any(x in ("typology", "paratyp") for x in prompt_modes_list):
        try:
            typology_map, typology_source_label = resolve_typology_map(
                cfg.get("typology_json"), REPO_ROOT
            )
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"Typology: {typology_source_label}")

    req_limit: int | None = None if cfg.get("no_limit") else cfg["limit"]
    only_langs = cfg.get("only_langs") or []
    only_set: set[str] | None = set(only_langs) if only_langs else None

    specs = load_language_specs(cfg.get("languages_config") or DEFAULT_LANGUAGES_YAML)
    exclude_langs = cfg.get("exclude_langs") or []
    if exclude_langs:
        excluded = set(exclude_langs)
        specs = [s for s in specs if s.name not in excluded]

    tatoeba_overrides: dict = {}
    if cfg.get("tatoeba_test_only"):
        tatoeba_overrides["tatoeba_splits"] = ["test"]
        tatoeba_overrides["include_mt_train"] = False
    else:
        tatoeba_splits = cfg.get("tatoeba_splits") or []
        if tatoeba_splits:
            tatoeba_overrides["tatoeba_splits"] = list(tatoeba_splits)
        tatoeba_overrides["include_mt_train"] = not cfg.get("no_tatoeba_mt_train", False)
    if cfg.get("smol_config"):
        tatoeba_overrides["smol_config"] = cfg["smol_config"]

    aligned = load_aligned_corpora(
        specs,
        only_langs=only_set,
        flores_split=cfg["flores_split"],
        tatoeba_overrides=tatoeba_overrides,
    )
    if not aligned:
        print("No corpora loaded; check only_langs and languages config.", file=sys.stderr)
        return 1

    paths = PipelinePaths(cfg["output_dir"])
    paths.setup_directories()

    directions: list[Direction]
    if cfg["directions"] == "both":
        directions = ["en_to_L", "L_to_en"]
    else:
        directions = [cfg["directions"]]  # type: ignore[list-item]

    manifest: list[BatchManifestEntry] = []
    test_splits_dir = cfg.get("test_splits_dir")
    frozen_dir = (
        test_splits_dir.expanduser().resolve() if test_splits_dir is not None else None
    )

    for lang_name, corpus in aligned.items():
        for direction in directions:
            stem = f"{lang_name}_{direction}"
            frozen_path = (
                frozen_dir / f"{stem}_test.csv" if frozen_dir is not None else None
            )
            if frozen_path is not None and frozen_path.is_file():
                test_split = load_frozen_test_split(frozen_path)
                train_df = train_df_excluding_test(
                    corpus.english,
                    corpus.target,
                    test_split["source"],
                    test_split["target"],
                )
                train_split = corpus_split_for_direction(
                    train_df, train_df.iloc[0:0], direction
                )["train"]
                print(f"  {stem}: frozen test from {frozen_path.name} (n={len(test_split['source'])})")
            else:
                if frozen_dir is not None:
                    print(f"Skipping {stem}: missing {frozen_path}", file=sys.stderr)
                    continue
                train_df, test_df = split_pairs_train_test(
                    corpus.english,
                    corpus.target,
                    test_size=cfg["test_size"],
                    random_state=cfg["seed"],
                    full_test=cfg.get("full_test", False),
                )
                if len(test_df) == 0:
                    print(f"Skipping {lang_name}: empty test after split.")
                    continue
                corp = corpus_split_for_direction(train_df, test_df, direction)
                test_split = corp["test"]
                train_split = corp["train"]

            if not test_split["source"]:
                print(f"Skipping {stem}: empty test.")
                continue

            csv_path = paths.test_splits / f"{stem}_test.csv"
            n_cap = (
                len(test_split["source"])
                if req_limit is None
                else min(len(test_split["source"]), req_limit)
            )
            if not dry_run:
                save_test_table(csv_path, test_split, max_rows=n_cap)
            test_sents = test_split["source"][:n_cap]

            for mode in prompt_modes_list:
                typo_txt = typology_map.get(lang_name, "") if mode != "zeroshot" else ""
                if mode in ("typology", "paratyp") and not typo_txt.strip():
                    print(
                        f"Warning: {lang_name} / {mode}: empty typology text",
                        file=sys.stderr,
                    )
                prompts = build_prompts_for_mode(
                    mode, lang_name, direction, test_sents, train_split, typo_txt
                )
                trunc_cap = cfg.get("truncate_max_input_tokens")
                prompts, trunc_i = truncate_typology_paratyp_prompts(
                    prompts,
                    mode,
                    enabled=trunc_cap is not None,
                    model_name=cfg["model"],
                    max_input_tokens=trunc_cap or 0,
                )
                if trunc_cap is not None and trunc_i:
                    print(
                        f"  truncate [{stem} / {mode}]: trimmed {trunc_i}/{len(prompts)} rows"
                    )
                if dry_run:
                    k = min(2, len(prompts))
                    if k:
                        print(f"\n{'=' * 72}\n{stem} [{mode}]: first {k} prompts\n{'=' * 72}")
                        for j in range(k):
                            print(f"\n--- #{j} ---\n{prompts[j]}\n")
                    print(f"[dry-run] {stem} [{mode}]: {len(prompts)} prompts")
                    continue

                batch_path = paths.batches / f"batch_{stem}_{mode}.jsonl"
                n = write_batch_jsonl_from_prompts(
                    batch_path,
                    lang_name=lang_name,
                    direction=direction,
                    prompt_mode=mode,
                    prompts=prompts,
                    model=cfg["model"],
                    temperature=cfg["temperature"],
                )
                manifest.append(
                    BatchManifestEntry(
                        lang=lang_name,
                        direction=direction,
                        prompt_mode=mode,
                        batch_jsonl=str(batch_path.relative_to(paths.root)),
                        test_csv=str(csv_path.relative_to(paths.root)),
                        num_requests=n,
                    )
                )
                print(f"OK {stem} [{mode}]: {n} rows -> {batch_path.name}")

    if dry_run:
        print("\n[dry-run] Done. Re-run without --dry-run to write files.")
        return 0

    typology_json = cfg.get("typology_json")
    man = write_manifest(
        paths,
        entries=manifest,
        summary_meta={
            "flores_split": cfg["flores_split"],
            "full_test": cfg.get("full_test", False),
            "model": cfg["model"],
            "temperature": cfg["temperature"],
            "limit": req_limit,
            "prompt_modes": prompt_modes_list,
            "truncate_max_input_tokens": cfg.get("truncate_max_input_tokens"),
            "typology_json": (
                str(typology_json.resolve())
                if typology_json is not None
                else typology_source_label
            ),
        },
    )
    print(f"Manifest: {man}")
    return 0
