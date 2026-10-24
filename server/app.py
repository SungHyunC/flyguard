"""FlyGuard web server. Run from the project root after `python -m flyguard.build`:

    uvicorn server.app:app --host 127.0.0.1 --port 8000

Two noses for the fly:
  - "hybrid":  the cross-fitted fine-tuned encoders of `flyguard.hybrid`; each of the K flies smells the
               prompt with its own nose and mushroom body, and the flies vote (mean P(injection)).
  - "frozen":  one fly with the frozen MiniLM nose (fast; also what the offline fallback replays).
Loads the saved models once (never refits), works offline, and falls back to the CPU without a GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # a live demo must not depend on the network

import json
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from flyguard.brain import layout
from flyguard.build import best_hybrid, comparator
from flyguard.encoders import MiniLMEncoder
from flyguard.mushroom_body import FlyGuard
from flyguard.paths import DEMO, REPORTS, WEB

MAX_CHARS = 1000

guard = FlyGuard.load(DEMO, MiniLMEncoder())
hybrid = best_hybrid()
comp_name, comp_proba, comp_threshold = comparator()
brain = layout(guard.mb.wiring)
samples = json.loads((DEMO / "samples.json").read_text())
lock = threading.Lock()                         # one GPU, one request at a time

noses = []
if hybrid is not None:
    noses.append({"id": "hybrid", "label": f"미세조정 {hybrid.name} 코", "flies": hybrid.meta["k"],
                  "val_acc": hybrid.meta["validation"]["fly"]["acc"]})
noses.append({"id": "frozen", "label": "고정 MiniLM 코", "flies": 1,
              "val_acc": json.loads((DEMO / "batch_validation.json").read_text())["fly_acc"]})


def smell(text: str, nose: str) -> tuple[list[dict], dict]:
    """Traces of every fly and their vote."""
    if nose == "hybrid" and hybrid is not None:
        traces = [g.classify(text) for _, g, _ in hybrid.folds]
        p = float(np.mean([t["p"] for t in traces]))
        decision = "injection" if p > hybrid.meta["threshold"]["fly"] else "benign"
    else:
        traces = [guard.classify(text)]
        p, decision = traces[0]["p"], traces[0]["decision"]
    n_inj = sum(t["decision"] == "injection" for t in traces)
    return traces, {"p": round(p, 4), "decision": decision, "injection": n_inj, "benign": len(traces) - n_inj}


for nose in noses:                              # warm up every model before the first visitor
    smell("warm up the noses", nose["id"])
comp_proba(["warm up the comparator"])

app = FastAPI(title="FlyGuard")


@app.middleware("http")
async def revalidate(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")   # browsers re-check files, so edits show on reload
    return response


class Prompt(BaseModel):
    text: str
    nose: str = "hybrid"


@app.get("/api/brain")
def get_brain():
    return brain


@app.get("/api/noses")
def get_noses():
    return noses


@app.get("/api/samples")
def get_samples():
    return samples


@app.post("/api/classify")
def classify(prompt: Prompt):
    text = prompt.text.strip()[:MAX_CHARS]
    if not text:
        raise HTTPException(400, "empty prompt")
    with lock:
        traces, vote = smell(text, prompt.nose)
        p = float(comp_proba([text])[0])
    # The fly shown in the brain panel by default: agrees with the vote and is closest to its probability.
    agree = [i for i, t in enumerate(traces) if t["decision"] == vote["decision"]] or list(range(len(traces)))
    rep = min(agree, key=lambda i: abs(traces[i]["p"] - vote["p"]))
    return {"text": text, "nose": prompt.nose if len(traces) > 1 else "frozen", "flies": traces, "rep": rep,
            "fly": traces[rep], "vote": vote,
            "comparator": {"name": comp_name, "p": round(p, 4), "decision": "injection" if p > comp_threshold else "benign"}}


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
