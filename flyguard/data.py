"""S-Labs/prompt-injection-dataset: download, clean and describe.

Labels: 0 = benign, 1 = injection. Validation and test are balanced and hand-curated, so all
model selection uses validation; test is only scored by `python -m flyguard.evaluate --final`.

    python -m flyguard.data        # download (once) and print a data report
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from .paths import RAW, ensure_dirs

REPO_ID = "S-Labs/prompt-injection-dataset"
SPLITS = ("train", "validation", "test")
LABELS = {0: "benign", 1: "injection"}


def download(force: bool = False) -> dict[str, Path]:
    ensure_dirs()
    paths = {}
    for split in SPLITS:
        target = RAW / "data" / f"{split}.csv"
        if force or not target.exists():
            hf_hub_download(REPO_ID, f"data/{split}.csv", repo_type="dataset", local_dir=RAW)
        paths[split] = target
    return paths


def load_splits(clean: bool = True) -> dict[str, pd.DataFrame]:
    """Load train/validation/test.

    clean=True removes from train: texts with conflicting labels, exact duplicates, and texts that
    also appear in validation or test. Test gets an `in_validation` column marking the texts it
    shares with validation, so scores can also be reported on the "test-clean" subset.
    """
    paths = download()
    d = {s: pd.read_csv(p, dtype={"text": str, "label": "int64"}, keep_default_na=False) for s, p in paths.items()}
    if clean:
        tr = d["train"]
        n_labels = tr.groupby("text").label.nunique()
        held_out = set(d["validation"].text) | set(d["test"].text)
        drop = set(n_labels[n_labels > 1].index) | held_out
        d["train"] = tr[~tr.text.isin(drop)].drop_duplicates("text").reset_index(drop=True)
    d["test"] = d["test"].assign(in_validation=d["test"].text.isin(set(d["validation"].text)))
    return d


def describe(d: dict[str, pd.DataFrame]) -> str:
    lines = []
    for split, df in d.items():
        counts = df.label.value_counts().sort_index()
        length = df.text.str.len().groupby(df.label).median()
        lines.append(
            f"{split:<10} n={len(df):>6}  benign={counts.get(0, 0):>5}  injection={counts.get(1, 0):>5}  "
            f"median chars: benign={length.get(0, 0):.0f} injection={length.get(1, 0):.0f}"
        )
    if "in_validation" in d["test"]:
        shared = d["test"][d["test"].in_validation]
        val_labels = dict(zip(d["validation"].text, d["validation"].label))
        disagree = sum(val_labels[t] != y for t, y in zip(shared.text, shared.label))
        lines.append(f"test texts shared with validation: {len(shared)} (label disagreements: {disagree}); "
                     f"test-clean n={len(d['test']) - len(shared)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raw = load_splits(clean=False)
    clean = load_splits(clean=True)
    print("raw:\n" + describe(raw))
    print(f"\ncleaned train: {len(raw['train'])} -> {len(clean['train'])} rows")
    print(describe(clean))
