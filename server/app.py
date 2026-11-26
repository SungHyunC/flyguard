"""FlyGuard web server. Run from the project root after `python -m flyguard.build`:

    uvicorn server.app:app --host 127.0.0.1 --port 8000

Noses for the fly:
  - "hybrid":  the cross-fitted fine-tuned encoders of `flyguard.hybrid`; each of the K flies smells the
               prompt with its own nose and mushroom body, and the flies vote (mean P(injection)).
  - "frozen":  one fly with the frozen MiniLM nose (fast; also what the offline fallback replays).
  - "e5":      one fly with the frozen multilingual e5 nose and the same brain; it reads Korean.
               Offered after `python -m flyguard.build --nose e5`.

Cascade (docs/analysis.md, section 4): a prompt far from every S-Labs training prompt is handed to an LLM
judge (Qwen3-4B) whose verdict becomes the final one. "Far" = the cosine distance to the nearest training
prompt in the frozen MiniLM space is larger than for 90% of S-Labs validation prompts. The LLM loads only when
the GPU has room for it after the other models (FLYGUARD_LLM=on forces it, FLYGUARD_LLM=off disables it).

Shortcut lab (/api/lab): one prompt and harmless character variants of it, scored by the ModernBERT trained on
the original S-Labs train and the one trained with the character augmentation (flyguard.augment), plus the flies.

Loads the saved models once (never refits), works offline, and falls back to the CPU without a GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # a live demo must not depend on the network

import json
import threading

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flyguard.brain import layout
from flyguard.build import best_hybrid, comparator
from flyguard.data import load_splits
from flyguard.encoders import E5Encoder, FineTunedEncoder, MiniLMEncoder, cached
from flyguard.mushroom_body import FlyGuard
from flyguard.paths import ARTIFACTS, DEMO, DEMO_E5, REPORTS, WEB
from flyguard.shortcuts import char_variants

MAX_CHARS = 1000
LLM_NAME, LLM_LABEL, LLM_REPORT_NAME = "Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B", "Qwen3-4B 제로샷 (거대언어모델)"
ROUTE_SHARE = 0.10        # share of S-Labs validation prompts the router would hand to the LLM
LLM_GB = 9.5              # GPU memory the LLM judge needs (weights + one prompt)

guard = FlyGuard.load(DEMO, MiniLMEncoder())
guard_e5 = FlyGuard.load(DEMO_E5, E5Encoder()) if (DEMO_E5 / "meta.json").exists() else None
hybrid = best_hybrid()
comp_name, comp_proba, comp_threshold = comparator()
aug = FineTunedEncoder(ARTIFACTS / "encoder_aug") if (ARTIFACTS / "encoder_aug" / "meta.json").exists() else None
brain = layout(guard.mb.wiring)
samples = json.loads((DEMO / "samples.json").read_text())
lock = threading.Lock()                         # one GPU, one request at a time

noses = []
if hybrid is not None:
    noses.append({"id": "hybrid", "label": f"미세조정 {hybrid.name} 코", "flies": hybrid.meta["k"],
                  "val_acc": hybrid.meta["validation"]["fly"]["acc"]})
noses.append({"id": "frozen", "label": "고정 MiniLM 코", "flies": 1,
              "val_acc": json.loads((DEMO / "batch_validation.json").read_text())["fly_acc"]})
if guard_e5 is not None:
    noses.append({"id": "e5", "flies": 1, **json.loads((DEMO_E5 / "nose.json").read_text(encoding="utf-8"))})

# Router: how unfamiliar a prompt is = cosine distance to the nearest S-Labs train prompt (frozen MiniLM).
splits = load_splits()
train_emb = cached(guard.encoder, "train", splits["train"].text)
val_dist = 1 - (cached(guard.encoder, "validation", splits["validation"].text) @ train_emb.T).max(1)
route_tau = float(np.quantile(val_dist, 1 - ROUTE_SHARE))


def load_llm():
    """(score function, threshold) of the LLM judge, or None when it is off or does not fit."""
    mode = os.environ.get("FLYGUARD_LLM", "auto").lower()
    if mode == "off":
        return None
    if mode != "on":
        if not torch.cuda.is_available():
            print("LLM judge off: no GPU (FLYGUARD_LLM=on loads it anyway)", flush=True)
            return None
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < LLM_GB:
            print(f"LLM judge off: {free:.1f} GB GPU memory free, about {LLM_GB} GB needed "
                  "(FLYGUARD_LLM=on loads it anyway)", flush=True)
            return None
    try:
        from flyguard.generalize import llm_scorer
        score = llm_scorer(LLM_NAME)
    except Exception as err:                    # not downloaded, out of memory, ...
        print(f"LLM judge off: {err}", flush=True)
        return None
    models = REPORTS / "generalization" / "models.json"
    meta = json.loads(models.read_text(encoding="utf-8")) if models.exists() else {}
    threshold = meta.get(LLM_REPORT_NAME, {}).get("threshold", 0.5)   # chosen on S-Labs validation
    print(f"LLM judge on: {LLM_NAME}, threshold {threshold:.3g}", flush=True)
    return score, threshold


def route(text: str) -> dict:
    """Familiar prompts stay with the fly; unfamiliar ones go to the LLM judge when it is loaded."""
    d = float(1 - (guard.encoder.encode([text]) @ train_emb.T).max())
    out = {"distance": round(d, 3), "tau": round(route_tau, 3), "kind": "fly", "llm_ready": llm is not None}
    if d > route_tau:
        out["kind"] = "unfamiliar"
        if llm is not None:
            q = float(llm[0]([text])[0])
            out.update(kind="llm", llm={"name": LLM_LABEL, "p": q, "threshold": llm[1],
                                        "decision": "injection" if q > llm[1] else "benign"})
    return out


def smell(text: str, nose: str) -> tuple[list[dict], dict]:
    """Traces of every fly and their vote."""
    if nose == "hybrid" and hybrid is not None:
        traces = [g.classify(text) for _, g, _ in hybrid.folds]
        p = float(np.mean([t["p"] for t in traces]))
        decision = "injection" if p > hybrid.meta["threshold"]["fly"] else "benign"
    else:
        fly = guard_e5 if nose == "e5" and guard_e5 is not None else guard
        traces = [fly.classify(text)]
        p, decision = traces[0]["p"], traces[0]["decision"]
    n_inj = sum(t["decision"] == "injection" for t in traces)
    return traces, {"p": round(p, 4), "decision": decision, "injection": n_inj, "benign": len(traces) - n_inj}


for nose in noses:                              # warm up every model before the first visitor
    smell("warm up the noses", nose["id"])
comp_proba(["warm up the comparator"])
if aug is not None:
    aug.predict_proba(["warm up the lab"])
llm = load_llm()                                # last, so it only takes the memory the others left

# Shortcut lab: the models compared on each character variant.
lab_models = [{"id": "orig", "label": f"{comp_name} (원래 train)",
               "score": lambda texts: [(float(p), p > comp_threshold) for p in comp_proba(texts)]}]
if aug is not None:
    lab_models.append({"id": "aug", "label": f"{comp_name} (글자 보강 train)",
                       "score": lambda texts: [(float(p), p > aug.threshold) for p in aug.predict_proba(texts)]})
lab_models.append({"id": "fly", "label": "초파리 (고정 MiniLM 코)",
                   "score": lambda texts: [(t["p"], t["decision"] == "injection") for t in map(guard.classify, texts)]})
if guard_e5 is not None:
    lab_models.append({"id": "e5", "label": "초파리 (고정 다국어 e5 코)",
                       "score": lambda texts: [(t["p"], t["decision"] == "injection") for t in map(guard_e5.classify, texts)]})

app = FastAPI(title="FlyGuard")


@app.middleware("http")
async def revalidate(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")   # browsers re-check files, so edits show on reload
    return response


class Prompt(BaseModel):
    text: str
    nose: str = "hybrid"


class LabRequest(BaseModel):
    text: str


@app.get("/api/brain")
def get_brain():
    return brain


@app.get("/api/noses")
def get_noses():
    return noses


@app.get("/api/samples")
def get_samples():
    return samples


@app.get("/api/cascade")
def get_cascade():
    return {"tau": round(route_tau, 3), "share": ROUTE_SHARE, "llm": LLM_LABEL if llm is not None else None}


@app.post("/api/classify")
def classify(prompt: Prompt):
    text = prompt.text.strip()[:MAX_CHARS]
    if not text:
        raise HTTPException(400, "empty prompt")
    with lock:
        traces, vote = smell(text, prompt.nose)
        p = float(comp_proba([text])[0])
        r = route(text)
    final = ({"p": r["llm"]["p"], "decision": r["llm"]["decision"], "by": "llm"} if r["kind"] == "llm"
             else {"p": vote["p"], "decision": vote["decision"], "by": "fly"})
    # The fly shown in the brain panel by default: agrees with the vote and is closest to its probability.
    agree = [i for i, t in enumerate(traces) if t["decision"] == vote["decision"]] or list(range(len(traces)))
    rep = min(agree, key=lambda i: abs(traces[i]["p"] - vote["p"]))
    nose = prompt.nose if prompt.nose in {n["id"] for n in noses} else "frozen"
    return {"text": text, "nose": nose, "flies": traces, "rep": rep, "fly": traces[rep], "vote": vote,
            "route": r, "final": final,
            "comparator": {"name": comp_name, "p": round(p, 4), "decision": "injection" if p > comp_threshold else "benign"}}


@app.post("/api/lab")
def lab(req: LabRequest):
    text = req.text.strip()[:MAX_CHARS]
    if not text:
        raise HTTPException(400, "empty prompt")
    variants = {name: texts[0] for name, texts in char_variants([text]).items()}
    with lock:
        scores = {m["id"]: m["score"](list(variants.values())) for m in lab_models}
    rows = [{"variant": name, "text": t,
             "scores": {mid: {"p": round(s[i][0], 4), "decision": "injection" if s[i][1] else "benign"}
                        for mid, s in scores.items()}}
            for i, (name, t) in enumerate(variants.items())]
    return {"models": [{"id": m["id"], "label": m["label"]} for m in lab_models], "rows": rows}


@app.get("/api/batch")
def get_batch():
    """Test predictions once `evaluate --final` has run, validation predictions until then."""
    path = REPORTS / "batch_test.json"
    return json.loads((path if path.exists() else DEMO / "batch_validation.json").read_text())


@app.get("/api/videos")
def get_videos():
    """NeuroMechFly videos with their prompts and verdicts (sim/flygym_choice.py writes videos.json)."""
    folder = WEB / "videos"
    meta = folder / "videos.json"
    known = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else []
    known = [m for m in known if (WEB / m["file"]).exists()]
    listed = {m["file"] for m in known}
    return known + [{"file": f"videos/{p.name}"} for p in sorted(folder.glob("*.mp4")) if f"videos/{p.name}" not in listed]


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
