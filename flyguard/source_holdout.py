"""Source holdout: does training on more kinds of sources help on a kind of source never seen?

    python -m flyguard.source_holdout        # writes reports/source_holdout.md (run flyguard.generalize first)

For each external group (direct, jailbreak, agentic, indirect, mixed, over-defence, Korean) one multilingual e5
is fine-tuned on S-Labs train plus up to PER_SUITE rows from every external suite outside that group, then scored
on the held-out group only. Horizon's own generated suites (synthetic_*) share a generator, so when the held-out
group has one of them, all of them leave training. The baseline is the same recipe on S-Labs train alone
(artifacts/encoder_e5, same seed and settings). Both pick their threshold on S-Labs validation, so AUC, which needs
no threshold, is the cleaner comparison.

These models break the "external suites are for evaluation only" rule on purpose, so none is kept: each one is
trained into a temporary folder, scored and deleted. Nothing else in the project reads their results.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_splits
from .generalize import OUT as GEN
from .paths import ARTIFACTS, REPORTS
from .train_encoder import fine_tune, predict_proba

MODEL = "intfloat/multilingual-e5-base"
BASELINE = ARTIFACTS / "encoder_e5"
PER_SUITE, EVAL_CAP = 300, 1000
HOLD = ["direct", "jailbreak", "agentic", "indirect", "mixed", "over_defense", "korean"]
GROUP_KO = {"direct": "직접 인젝션", "jailbreak": "탈옥", "agentic": "에이전트", "indirect": "간접 인젝션 (긴 문서)",
            "mixed": "역할극·문서 혼합", "over_defense": "과잉 방어 (정상만)", "korean": "한국어"}
OUT = REPORTS / "source_holdout.json"


def family(suite: str) -> str:
    return "horizon_generated" if suite.startswith("synthetic_") else suite


def sample(g: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows, half of each label when the suite has both."""
    parts = [c.sample(min(len(c), n // g.label.nunique()), random_state=seed) for _, c in g.groupby("label")]
    return pd.concat(parts)


def suite_metrics(y, p, thr) -> dict:
    pred = p > thr
    out = {"n": int(len(y))}
    parts = []
    if (y == 1).any():
        out["tpr"] = float(pred[y == 1].mean())
        parts.append(out["tpr"])
    if (y == 0).any():
        out["fpr"] = float(pred[y == 0].mean())
        parts.append(1 - out["fpr"])
    out["ba"] = float(np.mean(parts))
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def score(path: Path, texts: list[str]) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    p = predict_proba(model, tok, texts, device, batch_size=64)
    del model
    torch.cuda.empty_cache()
    return p


def run() -> None:
    d = load_splits()
    rows = pd.read_parquet(GEN / "rows.parquet")
    ext = rows[rows.group != "id"]
    pool = {s: sample(g, PER_SUITE, 0) for s, g in ext.groupby("suite")}
    evals = {s: (g if len(g) <= EVAL_CAP else g.sample(EVAL_CAP, random_state=1)) for s, g in ext.groupby("suite")}
    base_meta = json.loads((BASELINE / "meta.json").read_text())
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res.setdefault("runs", {})
    res["settings"] = {"model": MODEL, "per_suite": PER_SUITE, "eval_cap": EVAL_CAP, "baseline": str(BASELINE.name),
                       "baseline_threshold": base_meta["threshold"]}

    all_eval = pd.concat(evals.values())
    base_p = dict(zip(all_eval.index, score(BASELINE, all_eval.text.tolist())))
    for g in HOLD:
        if g in res["runs"]:
            print("skip", g, flush=True)
            continue
        held = sorted(ext[ext.group == g].suite.unique())
        banned = {family(s) for s in held}
        train_suites = sorted(s for s in pool if ext.loc[ext.suite == s, "group"].iat[0] != g and family(s) not in banned)
        extra = pd.concat([pool[s] for s in train_suites])
        texts = list(d["train"].text) + extra.text.tolist()
        labels = np.concatenate([d["train"].label.to_numpy(), extra.label.to_numpy()]).astype(np.int64)
        print(f"=== hold out {g}: {held}; +{len(extra)} rows from {len(train_suites)} suites", flush=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"flyguard_holdout_{g}_"))
        try:
            meta = fine_tune(texts, labels, list(d["validation"].text), d["validation"].label.to_numpy(), tmp, model_name=MODEL)
            per = {}
            for s in held:
                e = evals[s]
                y = e.label.to_numpy()
                per[s] = {"baseline": suite_metrics(y, np.array([base_p[i] for i in e.index]), base_meta["threshold"]),
                          "holdout": suite_metrics(y, score(tmp, e.text.tolist()), meta["threshold"])}
                print(s, json.dumps(per[s]), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        res["runs"][g] = {"held_out": held, "train_suites": train_suites, "extra_rows": int(len(extra)),
                          "threshold": meta["threshold"], "val": meta["val_at_threshold"], "suites": per}
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(res)


def write_report(res) -> None:
    p = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"   # noqa: E731
    a = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"   # noqa: E731

    def mean(per, who, key):
        vals = [m[who][key] for m in per.values() if key in m[who]]
        return float(np.mean(vals)) if vals else None

    body, detail = [], []
    for g, r in res["runs"].items():
        per = r["suites"]
        n = sum(m["baseline"]["n"] for m in per.values())
        b_ba, h_ba = mean(per, "baseline", "ba"), mean(per, "holdout", "ba")
        b_auc, h_auc = mean(per, "baseline", "auc"), mean(per, "holdout", "auc")
        diff = "–" if b_ba is None else f"{(h_ba - b_ba) * 100:+.1f}"
        body.append(f"| {GROUP_KO.get(g, g)} | {n:,} | {r['extra_rows']:,} ({len(r['train_suites'])}개) | {a(b_auc)} → {a(h_auc)} "
                    f"| {p(b_ba)} → {p(h_ba)} | {diff} |")
        for s, m in per.items():
            detail.append(f"| {GROUP_KO.get(g, g)} | {s} | {m['baseline']['n']:,} | {a(m['baseline'].get('auc'))} → {a(m['holdout'].get('auc'))} "
                          f"| {p(m['baseline'].get('tpr'))} → {p(m['holdout'].get('tpr'))} | {p(m['baseline'].get('fpr'))} → {p(m['holdout'].get('fpr'))} "
                          f"| {p(m['baseline']['ba'])} → {p(m['holdout']['ba'])} |")
    st = res["settings"]
    report = f"""# 출처 홀드아웃: 다른 출처를 더 배우면 처음 보는 종류의 출처에서도 나아지나

`python -m flyguard.source_holdout`이 만들었어요.

- **모델:** 다국어 e5 전체 미세조정 1개(4에폭, 시드 42). 비교 기준은 같은 설정으로 S-Labs train만 배운 `{st['baseline']}`예요.
- **학습에 더한 것:** 평가할 묶음을 뺀 나머지 외부 묶음에서 묶음마다 최대 {st['per_suite']}문장(정상·공격 반반)이에요. Horizon이 직접 생성한 묶음(synthetic_*)은 생성기가 같아서, 그중 하나를 평가할 때는 모두 학습에서 뺐어요.
- **평가:** 뺀 묶음의 문장(묶음마다 최대 {st['eval_cap']:,}개)이에요. 두 모델이 같은 문장을 채점해요.
- **기준선:** 둘 다 S-Labs validation에서 골랐어요(S-Labs만 배운 모델은 {st['baseline_threshold']:.3f}). 그래서 기준선이 필요 없는 AUC가 더 깨끗한 비교예요.
- **규칙 예외:** 이 실험의 모델은 외부 묶음을 학습에 쓰므로 저장하지 않고 지웠어요. 다른 결과표에는 섞이지 않아요.
- **시드 1개라** 2~3%p 차이는 흔들림 안이에요(RQ2).

## 묶음 종류별 (묶음 평균, %)

| 평가한 종류 (학습에서 뺌) | 평가 문장 | 학습에 더한 외부 문장 (묶음 수) | AUC: S-Labs만 → 다른 출처 추가 | 균형 정확도: S-Labs만 → 추가 | 차이 (%p) |
|---|---|---|---|---|---|
{chr(10).join(body)}

## 묶음별

| 종류 | 묶음 | 문장 | AUC | 탐지율 | 오탐률 | 균형 정확도 |
|---|---|---|---|---|---|---|
{chr(10).join(detail)}
"""
    (REPORTS / "source_holdout.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'source_holdout.md'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="only rebuild the report from reports/source_holdout.json")
    if ap.parse_args().report:
        write_report(json.loads(OUT.read_text(encoding="utf-8")))
    else:
        run()
