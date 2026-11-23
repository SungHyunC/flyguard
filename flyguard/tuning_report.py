"""reports/tuning.md from reports/tuning/results.jsonl and probes.jsonl (flyguard.tuning_study)."""
from __future__ import annotations


import numpy as np
import pandas as pd

from .paths import REPORTS

OUT = REPORTS / "tuning"
METHOD_KO = {
    "frozen_lp": "고정 + 선형 판독", "frozen_fly": "고정 + 초파리 판독", "bitfit": "BitFit (편향만)",
    "lora": "LoRA (저랭크 어댑터)", "top2": "상위 2층만", "full": "전체 미세조정", "lpft": "LP-FT (선형 먼저, 그다음 전체)",
}
GROUP_KO = {"over_defense": "과잉 방어", "direct": "직접", "jailbreak": "탈옥", "agentic": "에이전트",
            "indirect": "간접", "mixed": "혼합", "korean": "한국어"}


def pct(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"


def params(n):
    n = int(n)
    return f"{n:,}" if n < 1000 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def best_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per encoder x method: the epoch with the best validation accuracy, averaged over seeds."""
    if "seed" not in df:
        df = df.assign(seed=0)
    df = df.assign(seed=df.seed.fillna(0))
    best = df.loc[df.groupby(["encoder", "method", "seed"]).validation_acc.idxmax()]
    num = best.select_dtypes("number").columns.drop("seed")
    agg = best.groupby(["encoder", "method"])[num].mean()
    agg["seeds"] = best.groupby(["encoder", "method"]).size()
    return agg.reset_index()


def write_report() -> None:
    df = pd.read_json(OUT / "results.jsonl", lines=True)
    best = best_epochs(df)
    groups = [g for g in GROUP_KO if f"ood_{g}_ba" in best]
    sections = []
    for enc in df.encoder.unique():
        b = best[best.encoder == enc].set_index("method")
        body = []
        for m in [m for m in METHOD_KO if m in b.index]:
            r = b.loc[m]
            body.append([METHOD_KO[m], params(r.trainable), pct(r.test_acc), pct(r.ood_mean_ba)]
                        + [pct(r.get(f"ood_{g}_ba")) for g in groups] + [int(r.seeds)])
        sections.append(f"### {enc}\n" + table(
            ["튜닝 방법", "학습 파라미터", "S-Labs test 정확도", "외부 평균 BA"] + [GROUP_KO[g] for g in groups] + ["시드"], body))

    # learning dynamics: per-epoch in/out of distribution for the tuned methods
    dyn = []
    tuned = df[df.method.isin(["lora", "full", "lpft", "top2", "bitfit"])]
    seed0 = tuned[tuned.get("seed", 0).fillna(0) == 0] if "seed" in tuned else tuned
    for (enc, m), g in seed0.groupby(["encoder", "method"], sort=False):
        for _, r in g.sort_values("epoch").iterrows():
            dyn.append([enc, METHOD_KO[m], int(r.epoch), pct(r.validation_acc), pct(r.test_acc), pct(r.ood_mean_ba),
                        pct(r.get("ood_korean_ba")), pct(r.get("ood_over_defense_ba")), pct(r.get("ood_indirect_ba"))])
    dynamics = table(["인코더", "방법", "에폭", "validation", "test", "외부 평균", "한국어", "과잉 방어", "간접"], dyn)

    probes = ""
    if (OUT / "probes.jsonl").exists():
        pr = pd.read_json(OUT / "probes.jsonl", lines=True)
        parts = []
        for enc, g in pr.groupby("encoder", sort=False):
            body = [[int(r.layer), pct(r.pretrained_probe_acc), pct(r.finetuned_probe_acc), f"{r.cka:.2f}"]
                    for _, r in g.sort_values("layer").iterrows()]
            parts.append(f"### {enc}\n" + table(["층", "사전학습 그대로 탐침 정확도", "전체 미세조정 후 탐침 정확도", "CKA (1 = 그대로)"], body))
        probes = "\n\n".join(parts)

    report = f"""# 튜닝 기법 비교: 어디서 성능이 나오고, 무엇을 잃는가

`python -m flyguard.tuning_study`가 만들었어요.
- **학습:** 모든 방법이 S-Labs train으로만 학습하고, validation 정확도가 가장 좋은 에폭을 골랐어요.
- **외부 평가:** 외부 묶음마다 최대 300문장을 뽑은 고정 패널로 균형 정확도(BA)를 쟀어요. 임계값은 S-Labs validation에서 골랐어요.
- **시드:** 시드 열이 2 이상이면 그 값들의 평균이에요.

## 1. 방법별 결과 (가장 좋은 에폭, %)
{chr(10).join(sections)}

## 2. 학습이 진행되면서 무엇이 변하나 (에폭별, %)
{dynamics}

## 3. 층별 탐침과 표현 변화 (전체 미세조정, validation 절반으로 학습하고 나머지 절반으로 평가)
- **탐침 정확도:** 그 층의 표현만으로 인젝션을 얼마나 구별할 수 있는지예요.
- **CKA:** 미세조정 전후 그 층 표현이 얼마나 닮았는지예요. 1이면 그대로이고, 작을수록 많이 바뀐 거예요.

{probes}
"""
    (REPORTS / "tuning.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'tuning.md'}")
