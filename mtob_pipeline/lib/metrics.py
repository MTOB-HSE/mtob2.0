from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .batch import parse_custom_id


def add_translation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import sacrebleu
    except ImportError as e:
        raise SystemExit("pip install sacrebleu (required for BLEU/ChrF)") from e

    def bleu_row(ref: str, hyp: str) -> float:
        try:
            return float(sacrebleu.sentence_bleu(hyp, [ref]).score)
        except Exception:
            return float("nan")

    def chrf_row(ref: str, hyp: str) -> float:
        try:
            return float(sacrebleu.sentence_chrf(hyp, [ref]).score)
        except Exception:
            return float("nan")

    def meteor_row(ref: str, hyp: str) -> float:
        try:
            from nltk.tokenize import word_tokenize
            from nltk.translate.meteor_score import meteor_score
        except ImportError:
            return float("nan")
        try:
            return float(meteor_score([ref], hyp) * 100.0)
        except Exception:
            try:
                ref_tok = word_tokenize(ref.lower())
                hyp_tok = word_tokenize(hyp.lower())
                return float(meteor_score([ref_tok], hyp_tok) * 100.0)
            except Exception:
                return float("nan")

    out = df.copy()
    out["BLEU"] = out.apply(
        lambda r: bleu_row(str(r["Reference"]), str(r["LLM_Translation"])), axis=1
    )
    out["ChrF"] = out.apply(
        lambda r: chrf_row(str(r["Reference"]), str(r["LLM_Translation"])), axis=1
    )
    out["METEOR"] = out.apply(
        lambda r: meteor_row(str(r["Reference"]), str(r["LLM_Translation"])), axis=1
    )
    return out


def parse_batch_output_file(jsonl_path: Path, test_csv_path: Path) -> pd.DataFrame:
    test_df = pd.read_csv(test_csv_path)
    rows: list[dict[str, Any]] = []

    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        res = json.loads(line)
        if res.get("error"):
            continue
        parsed = parse_custom_id(res["custom_id"])
        if parsed is None:
            continue
        lang, direction, idx = parsed
        try:
            body = res["response"]["body"]
            msg = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            continue
        prediction = msg.split("\n")[0].strip()
        row = test_df.loc[test_df["Index"] == idx]
        if row.empty:
            continue
        rows.append(
            {
                "Lang": lang,
                "Direction": direction,
                "Index": idx,
                "Source": str(row["Source"].iloc[0]),
                "Reference": str(row["Reference"].iloc[0]),
                "LLM_Translation": prediction,
            }
        )
    if not rows:
        return pd.DataFrame()
    base = pd.DataFrame(rows).sort_values(["Lang", "Direction", "Index"]).reset_index(drop=True)
    return add_translation_metrics(base)


def build_metrics_summary(metrics_dir: Path) -> Path:
    metrics_dir = metrics_dir.resolve()
    paths = sorted(metrics_dir.glob("*_scored.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_scored.csv files in {metrics_dir}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        label = path.name[: -len("_scored.csv")]
        rows.append(
            {
                "label": label,
                "n": len(df),
                "BLEU_mean": round(float(df["BLEU"].mean()), 2) if "BLEU" in df.columns else float("nan"),
                "ChrF_mean": round(float(df["ChrF"].mean()), 2) if "ChrF" in df.columns else float("nan"),
                "METEOR_mean": (
                    round(float(df["METEOR"].mean()), 2) if "METEOR" in df.columns else float("nan")
                ),
            }
        )

    if not rows:
        raise ValueError(f"All scored files in {metrics_dir} are empty")

    out = metrics_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
