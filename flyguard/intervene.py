"""Intervention: take the "unusual characters mean injection" shortcut out of the training data and check
whether the Korean false positives go away.

    python -m flyguard.intervene            # writes reports/intervention.md (about 8 minutes)
    python -m flyguard.intervene --report   # rebuild the report from reports/intervention.json

In S-Labs train the 22 prompts with non-ASCII characters are all obfuscation attacks, so a fine-tuned model
can learn that script alone predicts the label. The augmented train rewrites a random 20% of the benign AND
20% of the injection prompts with a harmless greeting in another script (Korean, Japanese, Chinese, Russian,
Spanish, emoji), so script no longer predicts the label. Nothing from the evaluation suites is used, and the
phrases of the character test (" 감사합니다.", " merci beaucoup") are not among the greetings. Both models
use the same recipe and seeds: full fine-tuning of ModernBERT, 3 epochs, the best validation epoch.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .external import load_suites
from .paths import REPORTS
from .shortcuts import CHAR_N, char_variants
from .tuning_report import best_epochs
from .tuning_study import ENCODERS, load_sets, proba, run_tuned

OUT = REPORTS / "intervention.json"
GREETINGS = ["안녕하세요!", "좋은 하루 되세요.", "ありがとうございます。", "よろしくお願いします。", "谢谢你。", "你好！",
             "Спасибо!", "Добрый день.", "¡Hola, buenos días!", "😊", "👍", "🙏"]
SEEDS = (0, 1)


def augment(train: dict, share: float = 0.2, seed: int = 0) -> dict:
    """Same prompts and labels; a share of each class gets a greeting before or after it."""
    rng = np.random.default_rng(seed)
    texts, labels = list(train["text"]), np.asarray(train["label"])
    for cls in (0, 1):
        idx = np.flatnonzero(labels == cls)
        for i in rng.choice(idx, int(share * len(idx)), replace=False):
            g = GREETINGS[rng.integers(len(GREETINGS))]
            texts[i] = f"{g} {texts[i]}" if rng.random() < 0.5 else f"{texts[i]} {g}"
    return {"text": texts, "label": labels}


def make_extra(test_benign: list[str], korean: dict):
    variants = char_variants(test_benign)

    def extra(model, tok, device, thr):
        out = {f"char::{name}": float((proba(model, tok, texts, device) > thr).mean()) for name, texts in variants.items()}
        for suite, df in korean.items():
            s, y = proba(model, tok, df.text.tolist(), device), df.label.to_numpy()
            pred = s > thr
            tpr, fpr = float(pred[y == 1].mean()), float(pred[y == 0].mean())
            out.update({f"{suite}_tpr": tpr, f"{suite}_fpr": fpr, f"{suite}_ba": (tpr + 1 - fpr) / 2,
                        f"{suite}_auc": float(roc_auc_score(y, s))})
        return out
    return extra


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = load_sets()
    test = pd.DataFrame({"text": sets["test"]["text"], "label": sets["test"]["label"]})
    benign = test[test.label == 0].sample(CHAR_N, random_state=2).text.tolist()
    ext = load_suites()
    korean = {"koreanguardrail": ext[ext.suite == "koreanguardrail"],
              "jhc56": ext[(ext.suite == "jhc56_registers") & (ext.register != "english")]}
    extra = make_extra(benign, korean)
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {(r["method"], r["seed"]) for r in results}
    for seed in SEEDS:
        for label, train in (("full", sets["train"]), ("full_aug", augment(sets["train"], seed=seed))):
            if (label, seed) in done:
                continue
            recs = []
            run_tuned("ModernBERT", ENCODERS["ModernBERT"], "full", {**sets, "train": train}, device, recs, seed=seed,
                      extra=extra, label=label)
            results += recs
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(results)


def write_report(results) -> None:
    df = pd.DataFrame(results)
    best = best_epochs(df).set_index("method")
    names = {"full": "전체 미세조정 (원래 train)", "full_aug": "전체 미세조정 (글자 보강 train)"}
    chars = [c for c in best.columns if c.startswith("char::")]
    pct = lambda v: f"{v * 100:.1f}"   # noqa: E731
    body = [[names[m], pct(best.loc[m, "test_acc"]), pct(best.loc[m, "ood_mean_ba"])]
            + [pct(best.loc[m, c]) for c in chars] for m in ("full", "full_aug") if m in best.index]
    char_tab = "\n".join(["| 학습 데이터 | S-Labs test 정확도 | 외부 패널 평균 BA | " + " | ".join(c[6:] for c in chars) + " |",
                          "|" + "---|" * (3 + len(chars))] + ["| " + " | ".join(r) + " |" for r in body])
    body = []
    for m in ("full", "full_aug"):
        if m not in best.index:
            continue
        r = best.loc[m]
        body.append([names[m]] + [f"{pct(r[f'{s}_tpr'])} / {pct(r[f'{s}_fpr'])} / {pct(r[f'{s}_ba'])} / {r[f'{s}_auc']:.2f}"
                                  for s in ("koreanguardrail", "jhc56")])
    ko_tab = "\n".join(["| 학습 데이터 | KoreanGuardrail 전체 | JHC56 한국어 말투 7종 |", "|---|---|---|"]
                       + ["| " + " | ".join(r) + " |" for r in body])
    seeds = int(best["seeds"].max()) if "seeds" in best else 1
    report = f"""# 개입 실험: "낯선 글자 = 공격" 지름길을 끊으면 한국어 오탐이 사라지나

`python -m flyguard.intervene`가 만들었어요.
- **원인:** S-Labs train에서 ASCII가 아닌 글자가 든 문장 22개는 전부 공격이에요.
- **개입:** train의 정상 문장 20%와 공격 문장 20%에 다른 문자(한국어·일본어·중국어·러시아어·스페인어·이모지)로 된 무해한 인사말을 앞이나 뒤에 붙였어요. 그러면 글자 종류만으로는 라벨을 맞힐 수 없어요.
- **누수 방지:** 평가 데이터는 하나도 쓰지 않았어요. 글자 시험에 쓰는 " 감사합니다.", " merci beaucoup"는 인사말 목록에 없어요.
- **같은 조건:** 두 모델은 같은 방법(ModernBERT 전체 미세조정, 3에폭, validation 최고 에폭)과 같은 시드로 학습했어요. 숫자는 시드 {seeds}개의 평균이에요.

## 1. 글자만 바꾼 정상 문장의 오탐률 (S-Labs test 정상 {CHAR_N}개, %)
{char_tab}

## 2. 한국어 (탐지율 / 오탐률 / 균형 정확도 % / AUC)
{ko_tab}
"""
    (REPORTS / "intervention.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'intervention.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        write_report(json.loads(OUT.read_text(encoding="utf-8")))
    else:
        run()
