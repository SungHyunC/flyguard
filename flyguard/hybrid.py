"""Hybrid: a fine-tuned encoder is the fly's nose, and the mushroom body makes the final decision.

Cross-fitting keeps the mushroom body honest. The training prompts are split into K folds. The
encoder of fold k is fine-tuned on the other K-1 folds; the antennal lobe and mushroom body of
fold k learn only from fold k, prompts that encoder never saw. (Counting KC activity on prompts the
encoder was trained on would make them look perfectly separable, and the dopamine weights would be
overconfident on new prompts.) The K fly circuits vote by averaging their probabilities, and the
threshold for the vote is chosen on validation.

    python -m flyguard.hybrid train                    # ModernBERT nose, K=5 (about 8 min, RTX 4070 Ti SUPER)
    python -m flyguard.hybrid train --model sentence-transformers/all-MiniLM-L6-v2    # small nose (~2 min)
    python -m flyguard.hybrid predict in.csv out.csv   # leaderboard file; `in.csv` needs a `text` column

Each nose is stored in artifacts/hybrid/<name>/; `predict` uses the one with the best validation
accuracy unless --name is given.

For comparison every fold also keeps a linear readout (logistic regression on the same out-of-fold
embeddings) and the classification head of its fine-tuned encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .connectome import Wiring
from .data import load_splits
from .encoders import FineTunedEncoder, cached
from .metrics import best_threshold, binary_metrics
from .mushroom_body import AntennalLobe, FlyGuard, FlyMushroomBody
from .paths import ARTIFACTS, DEMO, ensure_dirs
from .train_encoder import DEFAULT_MODEL, fine_tune

OUT = ARTIFACTS / "hybrid"
READOUTS = ("fly", "linear", "head")


def fold_encoder(path: Path) -> FineTunedEncoder:
    enc = FineTunedEncoder(path)
    # embedding cache key: hybrid name, fold, and training time (a retrained fold gets new embeddings)
    enc.name = f"{path.parents[1].name}-{path.parent.name}-{enc.meta['trained_at']}"
    return enc


def available() -> dict[str, dict]:
    """Trained hybrids by name, best validation accuracy of the fly readout first."""
    found = {p.parent.name: json.loads(p.read_text()) for p in OUT.glob("*/hybrid.json")}
    return dict(sorted(found.items(), key=lambda kv: -kv[1]["validation"]["fly"]["acc"]))


class HybridGuard:
    """K cross-fitted circuits (fine-tuned encoder -> antennal lobe -> mushroom body) voting together."""

    def __init__(self, path: Path):
        self.name = path.name
        self.meta = json.loads((path / "hybrid.json").read_text())
        self.folds = []
        for k in range(self.meta["k"]):
            enc = fold_encoder(path / f"fold{k}" / "encoder")
            guard = FlyGuard.load(path / f"fold{k}" / "guard", enc)
            linear = joblib.load(path / f"fold{k}" / "linear.joblib")
            self.folds.append((enc, guard, linear))

    def fold_scores(self, texts, readout: str = "fly", cache_split: str | None = None) -> np.ndarray:
        """(K, n) P(injection) from each fold's circuit."""
        texts, out = list(texts), []
        for enc, guard, linear in self.folds:
            if readout == "head":
                out.append(enc.predict_proba(texts))
                continue
            E = cached(enc, cache_split, texts) if cache_split else enc.encode(texts)
            if readout == "fly":
                out.append(guard.mb.predict_proba(guard.lobe.transform(E)))
            else:
                out.append(linear.predict_proba(E)[:, 1])
        return np.stack(out)

    def predict_proba(self, texts, readout: str = "fly", cache_split: str | None = None) -> np.ndarray:
        return self.fold_scores(texts, readout, cache_split).mean(0)

    def predict(self, texts, readout: str = "fly") -> np.ndarray:
        return (self.predict_proba(texts, readout) > self.meta["threshold"][readout]).astype(int)


def train(out: Path, k: int = 5, epochs: int = 3, model_name: str = DEFAULT_MODEL, seed: int = 0) -> None:
    ensure_dirs()
    d = load_splits()
    tr, va = d["train"], d["validation"]
    texts, y = tr.text.to_numpy(), tr.label.to_numpy()
    yva = va.label.to_numpy()
    # Same wiring and homeostasis setting as the web demo (chosen on validation by flyguard.build).
    wiring = Wiring.load(DEMO / "wiring.npz")
    homeostasis = json.loads((DEMO / "meta.json").read_text())["homeostasis"]

    val = {r: [] for r in READOUTS}
    folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(texts, y))
    for f, (fit_idx, held_idx) in enumerate(folds):
        fdir = out / f"fold{f}"
        if not (fdir / "encoder" / "meta.json").exists():          # finished folds are reused
            fine_tune(list(texts[fit_idx]), y[fit_idx], list(va.text), yva, fdir / "encoder", model_name,
                      epochs=epochs, seed=42 + f, save_dtype=torch.bfloat16)
            torch.cuda.empty_cache()
        enc = fold_encoder(fdir / "encoder")
        E_held = cached(enc, f"train-fold{f}", texts[held_idx])   # this encoder never saw these prompts
        E_val = cached(enc, "validation", va.text)

        lobe = AntennalLobe(wiring.n_glom, seed=f).fit(E_held)
        P_held, P_val = lobe.transform(E_held), lobe.transform(E_val)
        mb = FlyMushroomBody(wiring, homeostasis=homeostasis).fit(P_held, y[held_idx]).calibrate(P_val, yva)
        FlyGuard(enc, lobe, mb).save(fdir / "guard")
        linear = LogisticRegression(C=4.0, max_iter=3000).fit(E_held, y[held_idx])
        joblib.dump(linear, fdir / "linear.joblib")

        val["fly"].append(mb.predict_proba(P_val))
        val["linear"].append(linear.predict_proba(E_val)[:, 1])
        val["head"].append(enc.predict_proba(va.text))
        print(f"fold {f}: encoder trained on {len(fit_idx)}, circuit learned from {len(held_idx)} unseen prompts | val acc "
              + "  ".join(f"{r}={binary_metrics(yva, val[r][-1], best_threshold(yva, val[r][-1]))['acc']:.3f}" for r in READOUTS),
              flush=True)
        del enc
        torch.cuda.empty_cache()

    meta = {"k": k, "base_model": model_name, "wiring": wiring.source, "homeostasis": homeostasis,
            "threshold": {}, "validation": {}}
    for r in READOUTS:
        p = np.mean(val[r], axis=0)
        meta["threshold"][r] = best_threshold(yva, p)
        meta["validation"][r] = {m: round(v, 4) for m, v in binary_metrics(yva, p, meta["threshold"][r]).items()}
    (out / "hybrid.json").write_text(json.dumps(meta, indent=2))
    print("vote of all folds, validation:", json.dumps(meta["validation"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--name", help="folder under artifacts/hybrid (default: model name)")
    p = sub.add_parser("predict")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--readout", choices=READOUTS, default="fly")
    p.add_argument("--name", help="which trained hybrid (default: best on validation)")
    args = ap.parse_args()

    if args.cmd == "train":
        train(OUT / (args.name or args.model.split("/")[-1].lower()), args.k, args.epochs, args.model)
        return
    df = pd.read_csv(args.input, keep_default_na=False)
    hybrid = HybridGuard(OUT / (args.name or next(iter(available()))))
    prob = hybrid.predict_proba(df.text, args.readout)
    df["p_injection"] = prob.round(5)
    df["pred_label"] = (prob > hybrid.meta["threshold"][args.readout]).astype(int)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} predictions ({hybrid.name}, {args.readout} readout) to {args.output}")


if __name__ == "__main__":
    main()
