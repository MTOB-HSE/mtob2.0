#!/usr/bin/env python3
"""MTOB CLI: python mtob_pipeline/mtob.py prepare -c run.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.config import DEFAULT_OUTPUT_DIR, PipelinePaths
from lib.constants import VALID_PROMPT_MODES
from lib.metrics import build_metrics_summary, parse_batch_output_file
from lib.openai import batch_status, collect_results, submit_manifest
from lib.prepare import load_config, prepare


def prompt_modes_filter(raw: str | None) -> frozenset[str] | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip().lower()
    modes = (
        ["zeroshot", "typology", "paratyp"]
        if text == "all"
        else [x.strip().lower() for x in text.split(",") if x.strip()]
    )
    bad = [m for m in modes if m not in VALID_PROMPT_MODES]
    if bad:
        raise ValueError(f"Unknown prompt mode(s): {', '.join(bad)}")
    return frozenset(modes)


def main() -> int:
    p = argparse.ArgumentParser(prog="mtob")
    sub = p.add_subparsers(dest="command", required=True)
    o = argparse.ArgumentParser(add_help=False)
    o.add_argument("--output-dir", type=Path)

    a = sub.add_parser("prepare", parents=[o])
    a.add_argument("-c", "--run-config", type=Path, required=True)
    a.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("submit", parents=[o])
    a.add_argument("-o", "--ids-json", type=Path)
    a.add_argument("--prompt-modes")
    a.add_argument("--no-skip-existing", action="store_true")

    a = sub.add_parser("collect", parents=[o])
    a.add_argument("-i", "--ids-json", type=Path)

    a = sub.add_parser("status")
    a.add_argument("ids_json", type=Path, nargs="?", default="active_batch_jobs.json")

    a = sub.add_parser("score")
    a.add_argument("batch_output_jsonl", type=Path)
    a.add_argument("test_csv", type=Path)
    a.add_argument("-o", "--output-csv", type=Path)

    sub.add_parser("summarize", parents=[o])

    args = p.parse_args()
    out = getattr(args, "output_dir", None) or DEFAULT_OUTPUT_DIR

    if args.command == "prepare":
        return prepare(load_config(args.run_config, args.output_dir), args.dry_run)

    if args.command == "submit":
        try:
            modes = prompt_modes_filter(args.prompt_modes)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        return submit_manifest(
            out, args.ids_json, prompt_modes_filter=modes, skip_existing=not args.no_skip_existing
        )

    if args.command == "collect":
        return collect_results(out, args.ids_json)

    if args.command == "status":
        return batch_status(Path(args.ids_json))

    if args.command == "score":
        df = parse_batch_output_file(args.batch_output_jsonl, args.test_csv)
        path = args.output_csv or args.batch_output_jsonl.with_suffix(".scored.csv")
        df.to_csv(path, index=False)
        print(f"Metrics saved: {path} ({len(df)} rows)")
        return 0

    if args.command == "summarize":
        sm = build_metrics_summary(PipelinePaths(out).metrics)
        print(f"Summary: {sm}")
        return 0

    return 2


if __name__ == "__main__":
    main()