"""reports/generalization.md from the saved scores of flyguard.generalize.

The common metric is balanced accuracy (BA): the mean of recall on injections and recall on benign
prompts, using whichever classes a suite has. An all-benign suite therefore reports 1 - false positive
rate, an all-injection suite its recall, and 50% is chance on every suite. AUC is added where a suite
has both classes; it ignores the threshold and so separates "cannot rank" from "badly calibrated".
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .external import jhc56_llm_predictions
from .paths import REPORTS

OUT = REPORTS / "generalization"
GROUPS = {"id": "S-Labs test (학습 분포)", "over_defense": "과잉 방어 (정상만)", "direct": "직접 인젝션",
          "jailbreak": "탈옥", "agentic": "에이전트", "indirect": "간접 인젝션 (긴 문서)", "mixed": "역할극·문서 혼합",
          "korean": "한국어"}
REGISTERS = ["english", "polite", "gentle", "casual", "genz", "emoticon", "angry", "threat"]
REGISTER_KO = {"english": "영어 원문", "polite": "존댓말", "gentle": "극존칭", "casual": "반말", "genz": "젠지어",
               "emoticon": "이모티콘", "angry": "분노체", "threat": "협박체"}
TECHNIQUES = ["clean", "break_spacing", "zwsp_inject", "tensify", "chosung", "jamo_decompose"]
TECH_KO = {"clean": "원문", "break_spacing": "띄어쓰기 깨기", "zwsp_inject": "투명문자", "tensify": "된소리화",
           "chosung": "초성체", "jamo_decompose": "자모 분해"}


def slug(name):
    from .generalize import slug as s
    return s(name)


def metrics(y, s, thr):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if not len(y):
        return {}
    pred = s > thr
    parts, out = [], {"n": int(len(y))}
    if (y == 1).any():
        out["tpr"] = float(pred[y == 1].mean())
        parts.append(out["tpr"])
    if (y == 0).any():
        out["fpr"] = float(pred[y == 0].mean())
        parts.append(1 - out["fpr"])
    out["ba"] = float(np.mean(parts))
    if (y == 1).any() and (y == 0).any():
        out["auc"] = float(roc_auc_score(y, s))
    return out


def pct(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}"


def load():
    rows = pd.read_parquet(OUT / "rows.parquet")
    meta = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    scores = {m: np.load(OUT / f"scores_{slug(m)}.npy") for m in meta}
    return rows, meta, scores


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def write_report() -> None:
    rows, meta, scores = load()
    y = rows.label.to_numpy()
    models = list(meta)
    suites = [s for s in rows.suite.unique() if s != "slabs_validation"]
    per = {m: {s: metrics(y[(rows.suite == s).to_numpy()], scores[m][(rows.suite == s).to_numpy()], meta[m]["threshold"])
               for s in suites} for m in models}
    group_of = rows.drop_duplicates("suite").set_index("suite").group.to_dict()

    # 1. summary by group: macro BA over the suites of each group
    body = []
    for m in models:
        cells = [m]
        for g in GROUPS:
            vals = [per[m][s]["ba"] for s in suites if group_of[s] == g and per[m][s]]
            cells.append(pct(np.mean(vals)) if vals else "–")
        ood = [per[m][s]["ba"] for s in suites if group_of[s] != "id" and per[m][s]]
        cells += [pct(np.mean(ood)), meta[m]["latency_ms"]]
        body.append(cells)
    summary = table(["모델"] + list(GROUPS.values()) + ["외부 평균", "지연(ms/문장)"], body)

    # 2. per suite
    body = []
    for s in suites:
        n = per[models[0]][s].get("n", 0)
        body.append([f"{s} ({GROUPS[group_of[s]]}, {n})"] + [
            f"{pct(per[m][s].get('ba'))}" + (f" / {per[m][s]['auc']:.2f}" if "auc" in per[m][s] else "") for m in models])
    per_suite = table(["묶음 (유형, 문장 수)"] + models, body)

    # 1-4. does the S-Labs ranking predict the external ranking? bootstrap over suites for the uncertainty
    from scipy.stats import spearmanr
    ext_suites = [s for s in suites if group_of[s] != "id"]
    test_ba = np.array([per[m]["slabs_test"]["ba"] for m in models])
    M = np.array([[per[m][s].get("ba", np.nan) for s in ext_suites] for m in models])
    ext_ba = np.nanmean(M, axis=1)
    rho, p_rho = spearmanr(test_ba, ext_ba)
    rng = np.random.default_rng(0)
    boots = np.array([np.nanmean(M[:, rng.integers(0, len(ext_suites), len(ext_suites))], axis=1) for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    first = np.bincount(np.argmax(boots, axis=1), minlength=len(models)) / len(boots)
    rank_t = (-test_ba).argsort().argsort() + 1
    rank_e = (-ext_ba).argsort().argsort() + 1
    order = np.argsort(-ext_ba)
    rank_tab = table(["모델", "S-Labs test BA (순위)", "외부 평균 BA [묶음 재표집 95% 구간] (순위)", "재표집에서 1위인 비율"],
                     [[models[i], f"{test_ba[i] * 100:.1f} ({rank_t[i]}위)",
                       f"{ext_ba[i] * 100:.1f} [{lo[i] * 100:.1f}, {hi[i] * 100:.1f}] ({rank_e[i]}위)", f"{first[i] * 100:.0f}%"] for i in order])
    rank_note = (f"S-Labs test 순위와 외부 평균 순위의 스피어만 순위 상관은 ρ = {rho:.2f} (p = {p_rho:.2f}, 모델 {len(models)}개)예요. "
                 "1에 가까우면 안에서 잘하는 모델이 밖에서도 잘한다는 뜻이고, 0 근처면 안의 순위로 밖을 예측할 수 없다는 뜻이에요.")

    # 1-5. how much of the external loss is the threshold? Only suites with both classes. The oracle threshold
    # is picked on the suite itself, so it is a diagnosis of the gap, never a usable score.
    from sklearn.metrics import roc_curve
    two = [s for s in ext_suites if ((rows.suite == s) & (rows.label == 1)).any() and ((rows.suite == s) & (rows.label == 0)).any()]
    body = []
    for m in models:
        actual, oracle, aucs = [], [], []
        for s in two:
            sel = (rows.suite == s).to_numpy()
            ok = sel & ~np.isnan(scores[m])
            if not ok.any():
                continue
            fpr_c, tpr_c, _ = roc_curve(y[ok], scores[m][ok])
            oracle.append(0.5 + np.max(tpr_c - fpr_c) / 2)
            actual.append(per[m][s]["ba"])
            aucs.append(per[m][s]["auc"])
        a, o = np.mean(actual), np.mean(oracle)
        body.append([m, pct(a), pct(o), f"{(o - a) * 100:.1f}", f"{np.mean(aucs):.2f}"])
    threshold_tab = table(["모델", "S-Labs 임계값 그대로 BA", "묶음마다 최적 임계값일 때 BA (진단용)", "임계값 때문에 잃은 몫 (%p)", "AUC"], body)

    # 1-2. ranking ability per group (AUC, threshold-free), suites with both classes only
    ext_groups = [g for g in GROUPS if g != "id"]
    body = []
    for m in models:
        cells = [m]
        for g in GROUPS:
            vals = [per[m][s]["auc"] for s in suites if group_of[s] == g and "auc" in per[m][s]]
            cells.append(f"{np.mean(vals):.2f}" if vals else "–")
        body.append(cells)
    auc_tab = table(["모델"] + list(GROUPS.values()), body)

    # 1-3 and 3. detection rate / false-positive rate per group at both operating points
    def rates(key):
        body = []
        for m in models:
            t = meta[m][key]
            cells = [m]
            for g in ext_groups:
                vals = [metrics(y[(rows.suite == s).to_numpy()], scores[m][(rows.suite == s).to_numpy()], t)
                        for s in suites if group_of[s] == g]
                tpr = [x["tpr"] for x in vals if "tpr" in x]
                fpr = [x["fpr"] for x in vals if "fpr" in x]
                cells.append(f"{pct(np.mean(tpr)) if tpr else '–'} / {pct(np.mean(fpr)) if fpr else '–'}")
            body.append(cells)
        return table(["모델"] + [GROUPS[g] for g in ext_groups], body)
    rates_default, fpr1 = rates("threshold"), rates("threshold_fpr1")

    # 4. Korean obfuscation: BA by technique (strongest intensity, plus the clean control)
    kg = rows.suite == "koreanguardrail"
    from .external import load_suites
    ext = load_suites()
    tech = ext.loc[ext.suite == "koreanguardrail", ["technique", "intensity"]].reset_index(drop=True)
    idx = np.flatnonzero(kg.to_numpy())
    body = []
    for m in models:
        cells = [m]
        for t in TECHNIQUES:
            sel = idx[((tech.technique == t) & ((tech.intensity == 1.0) | (t == "clean"))).to_numpy()]
            cells.append(pct(metrics(y[sel], scores[m][sel], meta[m]["threshold"]).get("ba")))
        body.append(cells)
    korean_obf = table(["모델"] + [TECH_KO[t] for t in TECHNIQUES], body)

    # 5. registers: injection recall and benign false positives per register, with the Korean LLMs
    jr = rows.suite == "jhc56_registers"
    reg = ext.loc[ext.suite == "jhc56_registers", ["register"]].reset_index(drop=True)
    jidx = np.flatnonzero(jr.to_numpy())
    body = []
    for m in models:
        cells = [m]
        for r in REGISTERS:
            sel = jidx[(reg.register == r).to_numpy()]
            mm = metrics(y[sel], scores[m][sel], meta[m]["threshold"])
            cells.append(f"{pct(mm.get('tpr'))} / {pct(mm.get('fpr'))}")
        body.append(cells)
    llm = jhc56_llm_predictions()
    for (model, prompt), g in llm.groupby(["model", "prompt"], sort=False):
        cells = [f"{model} ({'정의 제공' if prompt == 'definition' else '기본' if prompt == 'baseline' else '가드 모델'}, 논문 공개 결과)"]
        for r in REGISTERS:
            gg = g[g.register == r]
            pred = gg.pred.fillna(0).astype(int).to_numpy()
            lab = gg.label.to_numpy()
            cells.append(f"{pct(pred[lab == 1].mean())} / {pct(pred[lab == 0].mean())}")
        body.append(cells)
    registers = table(["모델"] + [REGISTER_KO[r] for r in REGISTERS], body)

    # 6. length: BA by text length on the external English direct + indirect suites
    lens = rows.text.str.len().to_numpy()
    bins = [0, 100, 300, 1000, 3000, 10**9]
    names = ["~100자", "100~300자", "300~1,000자", "1,000~3,000자", "3,000자~"]
    ext_mask = (~rows.group.isin(["id", "korean"])).to_numpy()
    body = []
    for m in models:
        cells = [m]
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = np.flatnonzero(ext_mask & (lens >= lo) & (lens < hi))
            cells.append(pct(metrics(y[sel], scores[m][sel], meta[m]["threshold"]).get("ba")))
        body.append(cells)
    length = table(["모델"] + names, body)

    report = f"""# 일반화 진단: 재학습 없이 외부 데이터에 그대로 넣기

`python -m flyguard.generalize`가 만들었어요.
- **임계값:** 모든 모델은 배포할 때처럼 S-Labs validation에서 고른 임계값을 그대로 썼어요. 외부 데이터로는 아무것도 조정하지 않았어요.
- **공통 지표: 균형 정확도(BA)**
  - 인젝션 탐지율과 정상 통과율의 평균이에요. 50%가 찍기 수준이에요.
  - 정상만 있는 묶음은 "1 − 오탐률", 인젝션만 있는 묶음은 탐지율과 같아요.
- **거대언어모델 제로샷 (Qwen2.5-1.5B, Qwen3-4B):** 같은 지시문으로 "프롬프트 인젝션인가?"에 Yes/No로 답할 확률을 썼어요. 시간이 오래 걸려서 외부 묶음마다 같은 최대 {400}문장을 무작위로 뽑아 두 모델 모두 같은 문장으로 채점했어요. S-Labs validation·test는 전부 채점했어요.

## 1. 유형별 요약 (균형 정확도 %, 묶음 평균)
{summary}

### 1-2. 구별 능력만 따로 (AUC, 임계값과 무관, 정상·인젝션이 둘 다 있는 묶음의 평균)
BA는 낮은데 AUC가 높으면, 순서는 맞게 매기는데 S-Labs에서 고른 임계값이 새 데이터에 맞지 않는다는 뜻이에요. 둘 다 0.5 근처면 아예 구별을 못 하는 거예요.

{auc_tab}

### 1-3. 탐지율 / 오탐률 (기본 임계값, %)
인젝션을 몇 % 잡았는지와 정상을 몇 % 잘못 막았는지를 같이 봐야 해요. 둘 다 100%면 그 유형의 문장을 전부 인젝션이라고 한 거예요.

{rates_default}

### 1-4. 안의 순위로 밖의 순위를 예측할 수 있나
- **순위 상관:** {rank_note}
- **묶음 재표집:** 외부 평균은 어떤 묶음을 골랐느냐에 따라 달라져요. 그래서 17개 묶음을 중복을 허락해 다시 뽑는 일을 2,000번 반복해 구간을 냈어요.

{rank_tab}

### 1-5. 밖에서 잃은 점수 중 임계값 탓은 얼마인가 (정상·인젝션이 둘 다 있는 외부 묶음 {len(two)}개 평균)
- **진단용 최적 임계값:** 묶음마다 그 묶음의 정답을 보고 가장 좋은 임계값을 고른 BA예요. 실제로는 쓸 수 없는 값이고, "순서는 맞게 매기는데 기준선이 어긋나서 잃은 몫"을 재는 데만 써요.
- **읽는 법:**
  - 최적 임계값일 때 BA가 낮으면 문장을 구별하는 능력 자체가 부족한 거예요.
  - 두 BA의 차이가 크면 S-Labs에서 고른 임계값이 새 데이터에 안 맞는 거예요.

{threshold_tab}

## 2. 묶음별 (균형 정확도 % / AUC)
{per_suite}

## 3. 실제 트래픽 기준 운영점 (S-Labs validation 정상의 1%만 차단하도록 임계값을 정했을 때, 탐지율 % / 오탐률 %)
{fpr1}

## 4. 한국어 난독화 (KoreanGuardrail, 가장 강한 변형의 균형 정확도 %)
{korean_obf}

## 5. 말투만 바꾼 같은 문장 (JHC56, 인젝션 탐지율 % / 정상 오탐률 %)
{registers}

## 6. 문장 길이별 (외부 영어·다국어 묶음, 균형 정확도 %)
{length}
"""
    (REPORTS / "generalization.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'generalization.md'}")
