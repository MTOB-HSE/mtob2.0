from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import BinaryIO

import httpx
from openai import APIConnectionError, BadRequestError, OpenAI

from .batch import BatchManifestEntry, load_manifest_entries
from .config import PipelinePaths
from .metrics import build_metrics_summary, parse_batch_output_file

_BATCH_SKIP_IF_STATUS = frozenset(
    {"validating", "in_progress", "finalizing", "completed", "cancelling"}
)
_BATCH_RESUBMIT_IF_STATUS = frozenset({"failed", "expired", "cancelled", "canceled"})


def _load_batch_ids_json(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _save_batch_ids_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class _UploadProgressFile(io.IOBase):
    def __init__(self, fh: BinaryIO, total_size: int, *, step_pct: int = 5) -> None:
        super().__init__()
        self._fh = fh
        self._total = max(int(total_size), 1)
        self._read = 0
        self._step = max(1, min(step_pct, 50))
        self._last_print_pct = -self._step

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if size is None:
            size = -1
        buf = self._fh.read(size)
        self._read += len(buf)
        pct = min(100, int(100.0 * self._read / self._total))
        while pct >= self._last_print_pct + self._step:
            self._last_print_pct += self._step
            msg_pct = min(self._last_print_pct, 100)
            print(
                f"      uploaded ~{msg_pct}% of file "
                f"({self._read / (1024 * 1024):.1f} / {self._total / (1024 * 1024):.1f} MiB)",
                flush=True,
            )
        return buf

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._fh.close()
        finally:
            super().close()


def _short_exc_summary(exc: BaseException) -> str:
    tail = exc.__cause__ or getattr(exc, "__context__", None)
    base = f"{type(exc).__qualname__}: {exc}"
    if tail is not None:
        base += f" | caused by: {type(tail).__qualname__}: {tail}"
    return base


def _print_exception_chain_stderr(exc: BaseException, *, indent: str = "      ") -> None:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__, chain=True)
    for block in lines:
        for sub in block.rstrip("\n").split("\n"):
            print(f"{indent}{sub}", file=sys.stderr, flush=True)


def submit_batches_openai(
    output_dir: Path,
    manifest: list[BatchManifestEntry],
    api_key: str | None,
    *,
    ids_storage_path: Path,
    skip_existing: bool = True,
) -> dict[str, str]:
    connect_s = float(os.environ.get("OPENAI_BATCH_HTTP_CONNECT", "120"))
    write_s = float(os.environ.get("OPENAI_BATCH_HTTP_WRITE", "7200"))
    read_s = float(os.environ.get("OPENAI_BATCH_HTTP_READ", "600"))
    max_retries_sdk = int(os.environ.get("OPENAI_MAX_RETRIES", "8"))
    upload_attempts = max(1, int(os.environ.get("OPENAI_BATCH_UPLOAD_ATTEMPTS", "5")))

    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        timeout=httpx.Timeout(
            connect=connect_s,
            read=read_s,
            write=write_s,
            pool=connect_s,
        ),
        max_retries=max_retries_sdk,
    )

    merged = _load_batch_ids_json(ids_storage_path)
    session_ids: dict[str, str] = {}
    n = len(manifest)

    for i, ent in enumerate(manifest, start=1):
        bpath = (output_dir / ent.batch_jsonl).resolve()
        key = f"{ent.lang}_{ent.direction}_{ent.prompt_mode}"
        total_sz = bpath.stat().st_size
        sz_mb = total_sz / (1024 * 1024)

        if skip_existing and key in merged:
            bid = merged[key]
            try:
                prev = client.batches.retrieve(bid)
                st = str(prev.status)
                if st in _BATCH_SKIP_IF_STATUS:
                    print(
                        f"[{i}/{n}] Skipping {key}: already submitted ({bid}), status {st}",
                        flush=True,
                    )
                    session_ids[key] = bid
                    continue
                if st in _BATCH_RESUBMIT_IF_STATUS:
                    print(
                        f"[{i}/{n}] Resubmitting {key}: previous job {bid} status {st}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{i}/{n}] Unknown status {st} for {key} ({bid}); retrying submit",
                        flush=True,
                    )
            except Exception as ex:
                print(
                    f"[{i}/{n}] Could not retrieve {key} ({bid}): {ex}; resubmitting",
                    flush=True,
                )

        print(
            f"[{i}/{n}] Uploading to OpenAI: {key} ({bpath.name}, {sz_mb:.1f} MiB) ...",
            flush=True,
        )
        up = None
        for attempt in range(1, upload_attempts + 1):
            try:
                fh = bpath.open("rb")
                prog = _UploadProgressFile(fh, total_sz)
                try:
                    up = client.files.create(file=(bpath.name, prog), purpose="batch")
                except BadRequestError as bre:
                    low = str(bre).lower()
                    if (
                        "jsonl" in low
                        or "invalid file format" in low
                        or "utf-8" in low
                        or "utf8" in low
                    ):
                        print(
                            "      [upload] BadRequest on multipart wrapper; "
                            "retrying with Path (no progress) ...",
                            flush=True,
                        )
                        up = client.files.create(file=bpath, purpose="batch")
                    else:
                        raise
                finally:
                    prog.close()
                break
            except BadRequestError:
                raise
            except APIConnectionError as err:
                print(f"      [network] {_short_exc_summary(err)}", flush=True)
                print(f"            attempt {attempt}/{upload_attempts}", flush=True)
                if attempt >= upload_attempts:
                    print(
                        "[network] full exception chain and traceback:",
                        file=sys.stderr,
                        flush=True,
                    )
                    _print_exception_chain_stderr(err)
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        assert up is not None
        print(f"      file body read completely ({total_sz / (1024 * 1024):.1f} MiB)", flush=True)
        print(f"[{i}/{n}] Creating batch job ...", flush=True)
        job = None
        for attempt in range(1, upload_attempts + 1):
            try:
                job = client.batches.create(
                    input_file_id=up.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                break
            except APIConnectionError as err:
                print(f"      [network] batches.create: {_short_exc_summary(err)}", flush=True)
                print(f"            attempt {attempt}/{upload_attempts}", flush=True)
                if attempt >= upload_attempts:
                    print(
                        "[network] full exception chain and traceback:",
                        file=sys.stderr,
                        flush=True,
                    )
                    _print_exception_chain_stderr(err)
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        assert job is not None
        merged[key] = job.id
        session_ids[key] = job.id
        _save_batch_ids_json(ids_storage_path, merged)
        print(f"[{i}/{n}] Done {key}: batch id = {job.id} (ids JSON saved)", flush=True)
    return session_ids


def submit_manifest(
    output_dir: Path,
    ids_out: Path | None,
    *,
    prompt_modes_filter: frozenset[str] | None = None,
    skip_existing: bool = True,
) -> int:
    paths = PipelinePaths(output_dir)
    if not paths.manifest.is_file():
        print(f"Missing {paths.manifest}; run prepare first.", file=sys.stderr)
        return 1
    manifest = load_manifest_entries(paths.manifest, prompt_modes_filter=prompt_modes_filter)
    if not manifest:
        filt = f" (mode filter: {sorted(prompt_modes_filter)})" if prompt_modes_filter else ""
        print(f"Manifest is empty or entries could not be read{filt}.", file=sys.stderr)
        return 1
    if prompt_modes_filter:
        print(
            f"Submitting modes only: {', '.join(sorted(prompt_modes_filter))} "
            f"({len(manifest)} batches)"
        )
    for ent in manifest:
        if not (paths.root / ent.batch_jsonl).is_file():
            print(f"Missing batch file: {paths.root / ent.batch_jsonl}", file=sys.stderr)
            return 1
    outp = ids_out or paths.submitted_batch_ids
    submit_batches_openai(
        paths.root,
        manifest,
        os.environ.get("OPENAI_API_KEY"),
        ids_storage_path=outp.resolve(),
        skip_existing=skip_existing,
    )
    print(f"Batch ids saved in {outp}")
    return 0


def collect_results(output_dir: Path, ids_json: Path | None) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    paths = PipelinePaths(output_dir)
    if not paths.manifest.is_file():
        print(f"Missing {paths.manifest}", file=sys.stderr)
        return 1
    ids_path = ids_json if ids_json is not None else paths.submitted_batch_ids
    if not ids_path.is_file():
        print(f"Missing {ids_path}; pass -i with batch ids JSON", file=sys.stderr)
        return 1

    manifest_data = json.loads(paths.manifest.read_text(encoding="utf-8"))
    ids_map = json.loads(ids_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    paths.setup_directories()
    processed = 0
    for ent in manifest_data.get("entries") or []:
        pm = ent.get("prompt_mode", "zeroshot")
        label = f"{ent['lang']}_{ent['direction']}_{pm}"
        bid = ids_map.get(label)
        if not bid:
            print(f"Skipping {label}: no key in {ids_path.name}")
            continue
        try:
            job = client.batches.retrieve(bid)
        except Exception as ex:
            print(f"{label}: retrieve error: {ex}", file=sys.stderr)
            continue
        if job.status != "completed":
            rc = getattr(job, "request_counts", None)
            extra = ""
            if rc is not None:
                failed = getattr(rc, "failed", None)
                extra = f" ({rc.completed}/{rc.total} done"
                if failed is not None:
                    extra += f", failed={failed}"
                extra += ")"
            print(f"{label}: status {job.status}{extra}; skipping")
            continue

        out_id = getattr(job, "output_file_id", None)
        err_id = getattr(job, "error_file_id", None)
        if not out_id:
            print(f"{label}: completed but no output_file_id")
            if err_id:
                err_path = paths.batch_results / f"{label}_errors.jsonl"
                try:
                    ec = client.files.content(err_id)
                    err_body = getattr(ec, "text", None) or ec.read().decode("utf-8")
                    err_path.write_text(err_body, encoding="utf-8")
                    print(f"  batch errors saved: {err_path}")
                except Exception as ex:
                    print(f"  failed to download error_file_id: {ex}", file=sys.stderr)
            continue

        content = client.files.content(out_id)
        text_body = getattr(content, "text", None) or content.read().decode("utf-8")
        outp = paths.batch_results / f"{label}_output.jsonl"
        outp.write_text(text_body, encoding="utf-8")

        test_csv = paths.root / ent["test_csv"]
        if not test_csv.is_file():
            print(f"{label}: missing {test_csv}", file=sys.stderr)
            continue

        df = parse_batch_output_file(outp, test_csv)
        mp = paths.metrics / f"{label}_scored.csv"
        df.to_csv(mp, index=False)
        processed += 1
        bleu = round(float(df["BLEU"].mean()), 2) if len(df) else float("nan")
        chrf = round(float(df["ChrF"].mean()), 2) if len(df) else float("nan")
        meteor = round(float(df["METEOR"].mean()), 2) if len(df) else float("nan")
        print(f"{label}: n={len(df)} BLEU={bleu} ChrF={chrf} METEOR={meteor} -> {mp}")

    if not processed:
        print("No completed batches with metrics were processed.", file=sys.stderr)
        return 1

    print(f"Summary: {build_metrics_summary(paths.metrics)}")
    return 0


def batch_status(ids_json: Path) -> int:
    if not ids_json.is_file():
        print(f"File not found: {ids_json}", file=sys.stderr)
        return 1
    data = json.loads(ids_json.read_text(encoding="utf-8"))
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print(f"{'Name':<18} {'Status':<14} {'Done':<12} {'batch id'}")
    print("-" * 72)
    for name in sorted(data.keys()):
        bid = data[name]
        try:
            job = client.batches.retrieve(bid)
            counts = job.request_counts
            prog = f"{counts.completed}/{counts.total}" if counts else "n/a"
            print(f"{name:<18} {job.status:<14} {prog:<12} {bid}")
        except Exception as e:
            print(f"{name:<18} {'error':<14} {'n/a':<12} {bid} ({e})")
    return 0
