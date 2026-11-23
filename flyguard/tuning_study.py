"""Tuning study: same encoders, different ways of adapting them, measured in and out of distribution.

    python -m flyguard.tuning_study               # all encoders and methods (about 40 min)
    python -m flyguard.tuning_study --report      # rebuild reports/tuning.md

Question: where does the accuracy come from, and what does each kind of tuning cost out of
distribution? Fine-tuning every weight can distort features learned in pretraining and hurt
transfer (Kumar et al., ICLR 2022), so methods are ordered from "touch nothing" to "touch everything":

  frozen_lp    frozen encoder, logistic regression on the mean-pooled last layer
  frozen_fly   frozen encoder, mushroom body readout (real FlyWire wiring, dopamine rule)
  bitfit       train only bias terms and the head (Ben Zaken et al., 2022)
  lora         rank-8 low-rank adapters on every linear layer of the encoder (Hu et al., 2022)
  top2         unfreeze the top two transformer layers and the head
  full         train every weight
  lpft         fit the head first with the encoder frozen, then train every weight (Kumar et al., 2022)

Every run trains on the S-Labs train split for 3 epochs and keeps the epoch with the best validation
accuracy. After each epoch it also scores a fixed out-of-distribution panel (up to 300 rows from each
external suite), so the curves show what improves and what breaks as training goes on. For the fully
fine-tuned models it also probes every layer (logistic regression on that layer's mean-pooled states)
and measures how far each layer moved from its pretrained version (linear CKA).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .connectome import Wiring
from .data import load_splits
from .metrics import best_threshold
from .mushroom_body import AntennalLobe, FlyMushroomBody
from .paths import DEMO, REPORTS

OUT = REPORTS / "tuning"
ENCODERS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "ModernBERT": "answerdotai/ModernBERT-base",
    "다국어 e5": "intfloat/multilingual-e5-base",
}
PREFIX = {"다국어 e5": "query: "}       # e5 was trained with this prefix on every input
METHODS = ["frozen_lp", "frozen_fly", "bitfit", "lora", "top2", "full", "lpft"]
LR = {"bitfit": 5e-4, "lora": 5e-4, "top2": 5e-5, "full": 3e-5, "lpft": 1e-5}
PANEL_PER_SUITE = 300
MAX_LEN = 256


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


class Classifier(nn.Module):
    """Encoder + mean pooling + linear head, identical for every encoder so methods are comparable."""

    def __init__(self, name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def pooled(self, enc, layer: int = -1):
        out = self.encoder(**enc, output_hidden_states=layer != -1)
        h = out.hidden_states[layer] if layer != -1 else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1)

    def forward(self, enc):
        return self.head(self.pooled(enc))


def prepare(model: Classifier, method: str) -> None:
    """Freeze, wrap or unfreeze the encoder for one tuning method."""
    enc = model.encoder
    for p in enc.parameters():
        p.requires_grad = method in ("full", "lpft")
    if method == "bitfit":
        names = {n for n, _ in enc.named_parameters() if n.endswith("bias")}
        if not names:                                      # ModernBERT has no bias terms: tune LayerNorm gains instead
            names = {n for n, _ in enc.named_parameters() if "norm" in n}
        for n, p in enc.named_parameters():
            p.requires_grad = n in names
    elif method == "lora":
        targets = [(n, m) for n, m in enc.named_modules() if isinstance(m, nn.Linear) and "embed" not in n]
        for n, m in targets:
            parent = enc.get_submodule(n.rsplit(".", 1)[0]) if "." in n else enc
            setattr(parent, n.rsplit(".", 1)[-1], LoRALinear(m))
    elif method == "top2":
        layers = [m for n, m in enc.named_modules() if n.count(".") == 1 and n.split(".")[0] in ("layers", "layer")]
        if not layers:                                     # BERT-style: encoder.layer.N
            layers = [m for n, m in enc.named_modules() if n.startswith("encoder.layer.") and n.count(".") == 2]
        for layer in layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True


def encode_batches(tok, texts, device, batch=128):
    for i in range(0, len(texts), batch):
        yield tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)


@torch.no_grad()
def embed(model: Classifier, tok, texts, device, layers=None):
    """Mean-pooled states: the last layer, or every layer listed (returns a list per layer)."""
    model.eval()
    if layers is None:
        out = []
        for enc in encode_batches(tok, texts, device):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out.append(model.pooled(enc).float().cpu())
        return torch.cat(out).numpy()
    per = {l: [] for l in layers}
    for enc in encode_batches(tok, texts, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hs = model.encoder(**enc, output_hidden_states=True).hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for l in layers:
            per[l].append(((hs[l].float() * mask).sum(1) / mask.sum(1)).cpu())
    return {l: torch.cat(v).numpy() for l, v in per.items()}


@torch.no_grad()
def proba(model: Classifier, tok, texts, device):
    model.eval()
    out = []
    for enc in encode_batches(tok, texts, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(torch.softmax(model(enc).float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def evaluate(scores: dict, sets: dict) -> dict:
    """Validation-chosen threshold, then accuracy in distribution and balanced accuracy / AUC per OOD group."""
    yv = sets["validation"]["label"]
    t = best_threshold(yv, scores["validation"])
    res = {"threshold": float(t)}
    for split in ("validation", "test"):
        res[f"{split}_acc"] = float(np.mean((scores[split] > t) == sets[split]["label"]))
    panel = sets["panel"]
    s, y = scores["panel"], panel["label"]
    group, suites = np.asarray(panel["group"]), np.asarray(panel["suite"])
    for g in sorted(set(group)):
        ba, auc = [], []
        for suite in pd.unique(suites[group == g]):
            m = suites == suite
            pred, yy = s[m] > t, y[m]
            parts = [pred[yy == 1].mean()] if (yy == 1).any() else []
            parts += [1 - pred[yy == 0].mean()] if (yy == 0).any() else []
            ba.append(np.mean(parts))
            if (yy == 1).any() and (yy == 0).any():
                auc.append(roc_auc_score(yy, s[m]))
        res[f"ood_{g}_ba"] = float(np.mean(ba))
        if auc:
            res[f"ood_{g}_auc"] = float(np.mean(auc))
    res["ood_mean_ba"] = float(np.mean([v for k, v in res.items() if k.startswith("ood_") and k.endswith("_ba")]))
    return res


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def load_sets():
    d = load_splits()
    rows = pd.read_parquet(REPORTS / "generalization" / "rows.parquet")
    ext = rows[~rows.suite.str.startswith("slabs")]
    rng = np.random.default_rng(0)
    panel = pd.concat([g.iloc[rng.permutation(len(g))[:PANEL_PER_SUITE]] for _, g in ext.groupby("suite")], ignore_index=True)
    sets = {s: {"text": d[s].text.tolist(), "label": d[s].label.to_numpy()} for s in ("train", "validation", "test")}
    sets["panel"] = {"text": panel.text.tolist(), "label": panel.label.to_numpy(), "group": panel.group, "suite": panel.suite}
    return sets


def run_frozen(enc_name, path, sets, device, results):
    model = Classifier(path).to(device)
    tok = AutoTokenizer.from_pretrained(path)
    E = {s: embed(model, tok, sets[s]["text"], device) for s in sets}
    lr = LogisticRegression(C=4.0, max_iter=4000).fit(E["train"], sets["train"]["label"])
    results.append({"encoder": enc_name, "method": "frozen_lp", "epoch": 0, "trainable": E["train"].shape[1] + 1,
                    **evaluate({s: lr.predict_proba(E[s])[:, 1] for s in E if s != "train"}, sets)})
    wiring = Wiring.load(DEMO / "wiring.npz")
    lobe = AntennalLobe(wiring.n_glom, 0).fit(E["train"])
    P = {s: lobe.transform(E[s]) for s in E}
    mb = FlyMushroomBody(wiring, homeostasis=False).fit(P["train"], sets["train"]["label"]).calibrate(P["validation"], sets["validation"]["label"])
    results.append({"encoder": enc_name, "method": "frozen_fly", "epoch": 0, "trainable": 0,
                    **evaluate({s: mb.predict_proba(P[s]) for s in P if s != "train"}, sets)})
    print(enc_name, "frozen:", {k: round(v, 3) for k, v in results[-1].items() if isinstance(v, float)}, flush=True)
    del model
    torch.cuda.empty_cache()


def run_tuned(enc_name, path, method, sets, device, results, seed=0, epochs=3, probe=False, extra=None, label=None):
    """Train one encoder with one method. `extra(model, tok, device, threshold)` adds measurements to every
    epoch's record (flyguard.intervene); `label` renames the method in the records."""
    torch.manual_seed(seed)
    model = Classifier(path).to(device)
    tok = AutoTokenizer.from_pretrained(path)
    if method == "lpft":                       # stage 1: logistic head on frozen features
        E = embed(model, tok, sets["train"]["text"], device)
        lr = LogisticRegression(C=4.0, max_iter=4000).fit(E, sets["train"]["label"])
        w = torch.tensor(lr.coef_[0], dtype=torch.float32)
        model.head.weight.data = torch.stack([-w / 2, w / 2]).to(device)
        model.head.bias.data = torch.tensor([-lr.intercept_[0] / 2, lr.intercept_[0] / 2], dtype=torch.float32, device=device)
    pre_layers = None
    if probe:
        n_layers = model.encoder.config.num_hidden_layers
        layers = list(range(n_layers + 1))
        probe_texts = sets["validation"]["text"]
        pre_layers = embed(model, tok, probe_texts, device, layers)
    prepare(model, method)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=LR[method], weight_decay=0.01)
    texts, y = sets["train"]["text"], sets["train"]["label"]
    steps = epochs * math.ceil(len(texts) / 32)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(texts))
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            enc = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(enc)
            loss = F.cross_entropy(logits.float(), torch.as_tensor(y[idx], dtype=torch.long, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        res = evaluate({s: proba(model, tok, sets[s]["text"], device) for s in ("validation", "test", "panel")}, sets)
        if extra is not None:
            res.update(extra(model, tok, device, res["threshold"]))
        res.update(encoder=enc_name, method=label or method, epoch=epoch, seed=seed, trainable=trainable, seconds=round(time.time() - t0, 1))
        results.append(res)
        print(enc_name, method, f"seed {seed} epoch {epoch}:", {k: round(v, 3) for k, v in res.items() if isinstance(v, float)}, flush=True)
    if probe:
        post = embed(model, tok, probe_texts, device, layers)
        yv = sets["validation"]["label"]
        half = len(yv) // 2
        for l in layers:
            rec = {"encoder": enc_name, "layer": l}
            for tag, H in (("pretrained", pre_layers[l]), ("finetuned", post[l])):
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(H[:half], yv[:half])
                rec[f"{tag}_probe_acc"] = float(clf.score(H[half:], yv[half:]))
            rec["cka"] = linear_cka(pre_layers[l], post[l])
            with (OUT / "probes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    del model, opt
    torch.cuda.empty_cache()


def done_runs() -> set:
    """(encoder, method, seed) already in results.jsonl, so an interrupted study resumes where it stopped."""
    path = OUT / "results.jsonl"
    if not path.exists():
        return set()
    df = pd.read_json(path, lines=True)
    seeds = df["seed"].fillna(0).astype(int) if "seed" in df else pd.Series(0, index=df.index)
    return set(zip(df.encoder, df.method, seeds))


def save(results: list) -> None:
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(encoders, methods, seeds) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_sets()
    done = done_runs()
    for enc_name in encoders:
        path = ENCODERS[enc_name]
        prefix = PREFIX.get(enc_name, "")
        sets = {s: {**v, "text": [prefix + t for t in v["text"]]} for s, v in base.items()}
        if any(m.startswith("frozen") for m in methods) and (enc_name, "frozen_lp", 0) not in done:
            results = []
            run_frozen(enc_name, path, sets, device, results)
            save(results)
        for method in [m for m in methods if not m.startswith("frozen")]:
            for seed in seeds:
                if (enc_name, method, seed) in done:
                    print(f"skip (done) {enc_name} {method} seed {seed}", flush=True)
                    continue
                results = []
                run_tuned(enc_name, path, method, sets, device, results, seed=seed, probe=(method == "full" and seed == seeds[0]))
                save(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        run(args.encoders, args.methods, args.seeds)
    from .tuning_report import write_report
    write_report()
