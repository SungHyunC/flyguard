"""Build everything the web demo loads.

    python -m flyguard.build                     # real FlyWire wiring if built, otherwise random
    python -m flyguard.build --wiring random

Fits on train and calibrates on validation only, then writes
  artifacts/demo/                 the demo FlyGuard (MiniLM nose + antennal lobe + mushroom body)
  artifacts/tfidf.joblib          the classical comparator (used when no fine-tuned encoder exists)
  artifacts/demo/samples.json     curated validation prompts with precomputed traces
  artifacts/demo/batch_validation.json   validation predictions of both models for batch mode
  web/fallback.js                 the same data for a server-less page
"""
from __future__ import annotations

import argparse
import json
import re

import joblib
import numpy as np

from .baselines import tfidf_model
from .brain import layout
from .connectome import Wiring
from .data import load_splits
from .encoders import FineTunedEncoder, MiniLMEncoder, cached
from .hybrid import OUT as HYBRID_OUT
from .hybrid import HybridGuard, available
from .metrics import binary_metrics
from .mushroom_body import AntennalLobe, FlyGuard, FlyMushroomBody
from .paths import ARTIFACTS, DEMO, WEB, ensure_dirs

FLYWIRE_WIRING = ARTIFACTS / "flywire" / "wiring.npz"

# Benign prompts that contain attack-like words are the interesting ones to show live.
TRIGGER = re.compile(r"\b(?:ignore|jailbreak|system|bypass|hidden|prompt|instruction|override|pretend|"
                     r"secret|reveal|unrestricted|rules?|forget|admin|password|hack)", re.I)
ATTACK_KINDS = {
    "override": r"\b(?:ignore|disregard|forget)\b",
    "roleplay": r"\b(?:pretend|DAN|act as|roleplay|persona|unrestricted)\b",
    "extraction": r"\b(?:system prompt|reveal|show (?:me )?your|instructions|configuration)\b",
    "obfuscation": r"[A-Za-z][0-9@$][A-Za-z]|[A-Za-z0-9+/]{16,}={0,2}",
    "urgency": r"\b(?:urgent|emergency|immediately|right now|life)\b",
    "hypothetical": r"\b(?:imagine|hypothetical|suppose|what if|fiction)\b",
}


def comparator():
    """Track A model if trained, otherwise TF-IDF. Returns (name, predict_proba, threshold)."""
    if (ARTIFACTS / "encoder" / "meta.json").exists():
        ft = FineTunedEncoder()
        return ft.label, ft.predict_proba, ft.threshold
    tfidf = joblib.load(ARTIFACTS / "tfidf.joblib")
    return "TF-IDF", lambda texts: tfidf.predict_proba(list(texts))[:, 1], 0.5


def batch_items(split, df, mb, P, comp_name, comp_p, comp_threshold, hybrid=None) -> dict:
    """Per-prompt predictions for the page's batch mode: the demo fly, the comparator and, when
    `hybrid` = (name, P(injection) of the fly vote, threshold) is given, the hybrid fly vote."""
    y = df.label.to_numpy()
    cols = {"fly": (mb.predict_proba(P), mb.predict(P)),
            "comp": (np.asarray(comp_p), (np.asarray(comp_p) > comp_threshold).astype(int))}
    out = {"split": split, "comparator": comp_name}
    if hybrid is not None:
        name, p, threshold = hybrid
        cols["hyb"] = (np.asarray(p), (np.asarray(p) > threshold).astype(int))
        out["hybrid"] = name
    for key, (_, pred) in cols.items():
        out[f"{key}_acc"] = float(np.mean(pred == y))
    out["items"] = [{"text": t, "label": int(y[i]),
                     **{k: int(pred[i]) for k, (_, pred) in cols.items()},
                     **{f"{k}_p": round(float(p[i]), 4) for k, (p, _) in cols.items()}}
                    for i, t in enumerate(df.text)]
    return out


def best_hybrid() -> HybridGuard | None:
    """The hybrid with the best validation accuracy, or None before `python -m flyguard.hybrid train`."""
    names = list(available())
    return HybridGuard(HYBRID_OUT / names[0]) if names else None


def curate(val, n_each: int = 12, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pick = lambda df, n: df.iloc[rng.permutation(len(df))[:n]]
    benign, attack = val[val.label == 0], val[val.label == 1]
    items = [(t, 0, "hard-negative") for t in pick(benign[benign.text.str.contains(TRIGGER)], 2 * n_each).text]
    items += [(t, 0, "benign") for t in pick(benign[~benign.text.str.contains(TRIGGER)], n_each).text]
    seen = set()
    for kind, pattern in ATTACK_KINDS.items():
        hits = attack[attack.text.str.contains(pattern, regex=True) & ~attack.text.isin(seen)]
        for t in pick(hits, max(2, n_each // 3)).text:
            seen.add(t)
            items.append((t, 1, kind))
    return [{"text": t, "label": y, "tag": tag} for t, y, tag in items]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", choices=["auto", "random", "flywire"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ensure_dirs()

    d = load_splits()
    tr, va = d["train"], d["validation"]
    enc = MiniLMEncoder()
    E = {s: cached(enc, s, d[s].text) for s in ("train", "validation")}

    use_flywire = args.wiring == "flywire" or (args.wiring == "auto" and FLYWIRE_WIRING.exists())
    if use_flywire:
        real = Wiring.load(FLYWIRE_WIRING)
        candidates = [real.transformed(w) for w in ("count", "log1p", "binary")]
    else:
        candidates = [Wiring.random(seed=args.seed)]
    lobe = AntennalLobe(candidates[0].n_glom, args.seed).fit(E["train"])
    P = {s: lobe.transform(E[s]) for s in E}

    # Synapse-weight transform and homeostasis: whichever validation prefers.
    best = None
    for wiring in candidates:
        for homeostasis in (True, False):
            mb = FlyMushroomBody(wiring, homeostasis=homeostasis).fit(P["train"], tr.label).calibrate(P["validation"], va.label)
            m = binary_metrics(va.label, mb.score(P["validation"]), mb.theta)
            print(f"{wiring.source}  homeostasis={homeostasis}: val acc={m['acc']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")
            if best is None or m["acc"] > best[0]["acc"]:
                best = (m, mb)
    wiring = best[1].wiring
    guard = FlyGuard(enc, lobe, best[1])
    guard.save(DEMO)

    tfidf = tfidf_model().fit(tr.text, tr.label)
    joblib.dump(tfidf, ARTIFACTS / "tfidf.joblib")
    name, comp_proba, comp_threshold = comparator()

    hybrid = best_hybrid()
    hyb = None if hybrid is None else (hybrid.name, hybrid.predict_proba(va.text, "fly", cache_split="validation"),
                                       hybrid.meta["threshold"]["fly"])
    batch = batch_items("validation", va, guard.mb, P["validation"], name, comp_proba(va.text), comp_threshold, hyb)
    (DEMO / "batch_validation.json").write_text(json.dumps(batch))

    samples = curate(va)
    texts = [s["text"] for s in samples]
    comp = comp_proba(texts)
    for s, rates, c in zip(samples, guard.rates(texts), comp):
        s["fly"] = guard.mb.trace(rates)
        s["comparator"] = {"name": name, "p": round(float(c), 4), "decision": "injection" if c > comp_threshold else "benign"}
    (DEMO / "samples.json").write_text(json.dumps(samples))

    fallback = {"brain": layout(wiring), "samples": samples, "batch": batch}
    (WEB / "fallback.js").write_text("window.FLYGUARD_FALLBACK = " + json.dumps(fallback) + ";\n", encoding="utf-8")
    print(f"saved demo to {DEMO} | fly val acc={batch['fly_acc']:.3f} | {name} val acc={batch['comp_acc']:.3f} "
          f"| {len(samples)} samples")


if __name__ == "__main__":
    main()
