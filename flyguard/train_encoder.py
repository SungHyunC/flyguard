"""Track A: fine-tune a transformer encoder for the leaderboard (plain PyTorch loop, no accelerate).

    python -m flyguard.train_encoder                     # ModernBERT-base
    python -m flyguard.train_encoder --model sentence-transformers/all-MiniLM-L6-v2 --lr 5e-5

Keeps the epoch with the best validation accuracy, picks the decision threshold on validation, and
saves model, tokenizer and meta.json to artifacts/encoder/. Test is never touched here.
`fine_tune` is also used by `flyguard.hybrid` to train the cross-fitting fold encoders.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .data import load_splits
from .metrics import best_threshold, binary_metrics
from .paths import ARTIFACTS, ensure_dirs

OUT = ARTIFACTS / "encoder"
DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def _autocast(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_proba(model, tok, texts: list[str], device, batch_size: int = 128, max_len: int = 256) -> np.ndarray:
    """P(injection) for each text."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with _autocast(device):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def fine_tune(texts: list[str], labels: np.ndarray, val_texts: list[str], val_labels: np.ndarray, out: Path,
              model_name: str = DEFAULT_MODEL, epochs: int = 4, lr: float = 5e-5, batch_size: int = 32,
              max_len: int = 256, seed: int = 42, save_dtype: torch.dtype | None = None) -> dict:
    """Fine-tune `model_name` on (texts, labels); keep the epoch with the best validation accuracy in `out`."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(texts) / batch_size)
    total = epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    print(f"{model_name} on {device}: {len(texts)} train texts, {total} steps -> {out}", flush=True)

    rng = np.random.default_rng(seed)
    history, best_acc, best_scores = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        order = rng.permutation(len(texts))
        for step, i in enumerate(range(0, len(order), batch_size), 1):
            idx = order[i:i + batch_size]
            enc = tok([texts[k] for k in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
            with _autocast(device):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / 100:.4f}", flush=True)
                running = 0.0

        scores = predict_proba(model, tok, val_texts, device, max_len=max_len)
        m = binary_metrics(val_labels, scores, 0.5)
        history.append({"epoch": epoch, **{k: round(v, 4) for k, v in m.items()}, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: val {history[-1]}", flush=True)
        if m["acc"] > best_acc:
            best_acc, best_scores = m["acc"], scores
            state = None if save_dtype is None else {k: v.to(save_dtype) for k, v in model.state_dict().items()}
            model.save_pretrained(out, state_dict=state)
            tok.save_pretrained(out)

    threshold = best_threshold(val_labels, best_scores)
    meta = {
        "base_model": model_name,
        "max_len": max_len,
        "threshold": threshold,
        "val_at_threshold": {k: round(v, 4) for k, v in binary_metrics(val_labels, best_scores, threshold).items()},
        "history": history,
        "train_size": len(texts),
        "trained_at": time.strftime("%Y%m%d-%H%M%S"),
        "args": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "seed": seed},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "val_scores.npy", best_scores)
    print(f"saved to {out}  threshold={threshold:.3f}  val={meta['val_at_threshold']}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    d = load_splits()
    tr, va = d["train"], d["validation"]
    fine_tune(list(tr.text), tr.label.to_numpy(), list(va.text), va.label.to_numpy(), OUT, args.model,
              args.epochs, args.lr, args.batch_size, args.max_len, args.seed)


if __name__ == "__main__":
    main()
