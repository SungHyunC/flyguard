"""Long documents: is the problem truncation or dilution, and does a window threshold carry across sources?

    python -m flyguard.long_docs        # writes reports/long_docs.md (needs flyguard.generalize first)

1. Truncation vs dilution. S-Labs test injections are planted at the start, middle or end of clean PIArena
   documents (as in flyguard.shortcuts) and the fine-tuned ModernBERT reads the first 256, 1,024 or 2,048
   tokens. It was trained on 256-token prompts but its position encoding reaches 8K. If longer reading finds
   the late attacks, truncation was the problem; if even an attack at the very start is missed, the short
   attack is diluted by the surrounding text.
2. Window threshold with a held-out source. Reading in 700-character windows and keeping the highest score
   raises detection but also false alarms on clean documents. The window threshold is chosen on one source's
   clean documents only (PIArena, 5% false alarms) and tested on other sources (BIPIA, synthetic documents,
   LLMail), against the S-Labs threshold read whole and windowed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .data import load_splits
from .encoders import FineTunedEncoder
from .generalize import OUT as GEN
from .paths import ARTIFACTS, REPORTS
from .shortcuts import windows
from .train_encoder import predict_proba

N_PLANT, N_EVAL = 200, 800
LENGTHS = (256, 1024, 2048)
CAL_SOURCE, EVAL_SOURCES = "piarena", ("bipia", "synthetic_docs_test", "llmail_phase2")
CAL_FPR = 0.05


def plant(doc, attack, where):
    cut = {"start": 0, "middle": len(doc) // 2, "end": len(doc)}[where]
    if 0 < cut < len(doc) and doc.rfind(". ", 0, cut) > 0:
        cut = doc.rfind(". ", 0, cut) + 2
    return (doc[:cut] + " " + attack + " " + doc[cut:]).strip()


def window_max(score, docs):
    win = [windows(d) for d in docs]
    flat = score([w for ws in win for w in ws])
    out, k = [], 0
    for ws in win:
        out.append(flat[k:k + len(ws)].max())
        k += len(ws)
    return np.array(out)


def rates(y, pred):
    out = {}
    if (y == 1).any():
        out["tpr"] = float(pred[y == 1].mean())
    if (y == 0).any():
        out["fpr"] = float(pred[y == 0].mean())
    parts = [out["tpr"]] if "tpr" in out else []
    parts += [1 - out["fpr"]] if "fpr" in out else []
    out["ba"] = float(np.mean(parts))
    return out


def run() -> None:
    ft = FineTunedEncoder(ARTIFACTS / "encoder")
    thr = ft.threshold
    rows = pd.read_parquet(GEN / "rows.parquet")
    test = load_splits()["test"]
    rng = np.random.default_rng(0)
    clean = rows[(rows.suite == CAL_SOURCE) & (rows.label == 0)].text.tolist()
    carriers = [clean[i] for i in rng.choice(len(clean), N_PLANT, replace=False)]
    attacks = test[test.label == 1].sample(N_PLANT, random_state=1).text.tolist()
    res = {"threshold": thr, "lengths": {}, "windows": {}}

    # 1. truncation vs dilution
    for L in LENGTHS:
        score = lambda texts, L=L: predict_proba(ft.model, ft.tok, list(texts), ft.device, batch_size=16, max_len=L)
        r = {w: float((score([plant(c, a, w) for c, a in zip(carriers, attacks)]) > thr).mean())
             for w in ("start", "middle", "end")}
        r["clean_fpr"] = float((score(carriers) > thr).mean())
        res["lengths"][L] = r
        print("max_len", L, {k: round(v, 3) for k, v in r.items()}, flush=True)
        torch.cuda.empty_cache()

    # 2. window threshold chosen on one source, tested on others
    score = lambda texts: predict_proba(ft.model, ft.tok, list(texts), ft.device, max_len=256)
    planted = set(carriers)
    cal_docs = [d for d in clean if d not in planted]
    cal_max = window_max(score, cal_docs)
    thr_win = float(np.quantile(cal_max, 1 - CAL_FPR))
    res["window_threshold"] = thr_win
    for src in EVAL_SOURCES:
        g = rows[rows.suite == src]
        g = g.iloc[rng.permutation(len(g))[:N_EVAL]]
        y, docs = g.label.to_numpy(), g.text.tolist()
        whole, win = score(docs), window_max(score, docs)
        res["windows"][src] = {"n": len(docs), "whole_slabs": rates(y, whole > thr), "window_slabs": rates(y, win > thr),
                               "window_calibrated": rates(y, win > thr_win)}
        print(src, json.dumps(res["windows"][src]), flush=True)
    (REPORTS / "long_docs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(res)


def write_report(res) -> None:
    p = lambda v: "–" if v is None else f"{v * 100:.0f}"   # noqa: E731
    body = [[f"{int(L):,}토큰"] + [p(r[w]) for w in ("start", "middle", "end")] + [p(r["clean_fpr"])]
            for L, r in res["lengths"].items()]
    t1 = "\n".join(["| 읽는 길이 | 문서 앞 | 문서 중간 | 문서 끝 | 정상 문서 오탐률 |", "|---|---|---|---|---|"]
                   + ["| " + " | ".join(r) + " |" for r in body])
    names = {"bipia": "BIPIA", "synthetic_docs_test": "synthetic_docs", "llmail_phase2": "LLMail (공격만)"}
    body = []
    for src, r in res["windows"].items():
        for key, label in (("whole_slabs", "앞 256토큰만 · S-Labs 기준선"), ("window_slabs", "창 나눔 · S-Labs 기준선"),
                           ("window_calibrated", "창 나눔 · PIArena 정상으로 맞춘 기준선")):
            x = r[key]
            body.append([f"{names[src]} ({r['n']})", label, p(x.get("tpr")), p(x.get("fpr")), p(x["ba"])])
    t2 = "\n".join(["| 출처 (문서 수) | 읽는 방식 | 탐지율 | 오탐률 | 균형 정확도 |", "|---|---|---|---|---|"]
                   + ["| " + " | ".join(r) + " |" for r in body])
    report = f"""# 긴 문서: 잘림인가 희석인가, 창 기준선은 다른 출처로 옮겨지나

`python -m flyguard.long_docs`가 만들었어요. 모델은 미세조정 ModernBERT 1개(256토큰으로 학습)예요.

## 1. 더 길게 읽으면 늦게 숨긴 공격을 찾나 (PIArena 정상 문서 {N_PLANT}개에 S-Labs 공격을 심음, 탐지율 %)
- **읽는 길이만 바꿨어요.** 학습은 256토큰으로 했지만, 위치 표현은 8K토큰까지 읽을 수 있어요.
- **읽는 법:** 길게 읽어도 "문서 앞"부터 낮으면 잘림이 아니라 희석이 문제예요.

{t1}

## 2. 창 기준선을 한 출처의 정상 문서로 맞추면 다른 출처에서도 통하나 (%)
- **창 나눔:** 문서를 700자 창으로 나눠 가장 높은 점수를 써요.
- **기준선 맞추기:** PIArena 정상 문서(평가에 쓰는 문서와 겹치지 않음)에서 오탐 {CAL_FPR * 100:.0f}%가 되는 점수로 정했어요(값 {res['window_threshold']:.3f}). S-Labs 기준선은 {res['threshold']:.3f}예요.
- **평가:** 기준선을 맞출 때 쓰지 않은 다른 출처예요(출처마다 최대 {N_EVAL}개).

{t2}
"""
    (REPORTS / "long_docs.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'long_docs.md'}")


if __name__ == "__main__":
    run()
