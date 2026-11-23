"""Why do detectors decide what they decide? Controlled probes on saved and fresh scores.

    python -m flyguard.shortcuts        # writes reports/shortcuts.md (needs flyguard.generalize first)

0. Character counterfactuals: benign S-Labs test prompts with harmless non-ASCII added (Korean, emoji,
   a zero-width space, Cyrillic look-alike letters) or harmless French. The false-positive jump measures
   whether "unusual characters" alone reads as an attack.
1. Word occlusion: delete one word at a time from S-Labs test prompts and record how much P(injection)
   drops. The words whose removal moves the score most are what the model leans on, and the share of
   injections that flip to "benign" after deleting a single word measures keyword reliance.
2. Hidden-instruction position: put S-Labs test injections at the start, middle or end of a clean long
   document (PIArena contexts with the attack removed) and check detection. Models read at most 256 or
   512 tokens, so an instruction placed late may never be seen. A sliding window over the whole
   document (score every window, keep the maximum) is the fix being tested, together with what it costs
   in false positives on the clean documents.
3. Calibration: expected calibration error of P(injection) in and out of distribution, from the scores
   saved by flyguard.generalize.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch

from .data import load_splits
from .generalize import OUT as GEN, scorers, slug
from .paths import REPORTS

OCCLUDE_N = 150          # prompts per class for occlusion
CHAR_N = 400             # benign prompts for the character counterfactuals
POSITION_N = 200         # injections planted per position
WINDOW_CHARS, STRIDE_CHARS = 700, 350
MODELS = ["TF-IDF + 로지스틱 회귀", "초파리 · 고정 MiniLM 코", "ModernBERT 미세조정 1개",
          "하이브리드 ModernBERT · 초파리 투표", "ProtectAI DeBERTa v2 (외부 전문 탐지기)"]


def words(text):
    return re.findall(r"\S+", text)


def char_variants(texts):
    """Benign prompts with harmless non-ASCII added. In S-Labs train the only 22 non-ASCII prompts are all
    obfuscation attacks (homoglyphs, zero-width characters), so a model may have learned "unusual characters
    means injection". The French control is non-English but plain ASCII, separating language from characters."""
    cyrillic = str.maketrans({"o": "о", "e": "е", "a": "а"})
    return {
        "원문 (대조)": texts,
        "끝에 ' merci beaucoup' (프랑스어, ASCII)": [t + " merci beaucoup" for t in texts],
        "끝에 ' 감사합니다.' (한국어)": [t + " 감사합니다." for t in texts],
        "끝에 ' 😊' (이모지)": [t + " 😊" for t in texts],
        "첫 단어 뒤에 투명문자 (U+200B)": [re.sub(r"^(\S+)", lambda m: m.group(1) + "​", t, count=1) for t in texts],
        "o·e·a를 생김새가 같은 키릴 글자로": [t.translate(cyrillic) for t in texts],
    }


def occlusion(score, texts):
    """Per text: base score, and the score after deleting each word."""
    variants, owner = [], []
    for i, t in enumerate(texts):
        w = words(t)
        for j in range(len(w)):
            variants.append(" ".join(w[:j] + w[j + 1:]))
            owner.append((i, w[j]))
    base = score(texts)
    drop = base[[i for i, _ in owner]] - score(variants)
    return base, owner, drop


def windows(doc):
    if len(doc) <= WINDOW_CHARS:
        return [doc]
    return [doc[i:i + WINDOW_CHARS] for i in range(0, max(1, len(doc) - WINDOW_CHARS + STRIDE_CHARS), STRIDE_CHARS)]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def run() -> None:
    d = load_splits()
    meta = json.loads((GEN / "models.json").read_text(encoding="utf-8"))
    rows = pd.read_parquet(GEN / "rows.parquet")
    rng = np.random.default_rng(0)
    test = d["test"]
    inj = test[test.label == 1].sample(OCCLUDE_N, random_state=0).text.tolist()
    ben = test[test.label == 0].sample(OCCLUDE_N, random_state=0).text.tolist()

    piarena = rows[(rows.suite == "piarena") & (rows.label == 0)].text.tolist()
    carriers = [piarena[i] for i in rng.choice(len(piarena), POSITION_N, replace=False)]
    attacks = test[test.label == 1].sample(POSITION_N, random_state=1).text.tolist()

    def plant(doc, attack, where):
        cut = {"start": 0, "middle": len(doc) // 2, "end": len(doc)}[where]
        cut = doc.rfind(". ", 0, cut) + 2 if 0 < cut < len(doc) and doc.rfind(". ", 0, cut) > 0 else cut
        return (doc[:cut] + " " + attack + " " + doc[cut:]).strip()

    benign = test[test.label == 0].sample(CHAR_N, random_state=2).text.tolist()
    variants = char_variants(benign)

    factories = {name: make for name, _, make in scorers(d)}
    results = {"occlusion": {}, "position": {}, "characters": {}}
    for name in MODELS:
        score = factories[name]()
        thr = meta[name]["threshold"]
        # 0. character counterfactuals: share of benign prompts flagged as injection
        results["characters"][name] = {v: float((score(t) > thr).mean()) for v, t in variants.items()}
        # 1. occlusion
        base, owner, drop = occlusion(score, inj + ben)
        per_word = pd.DataFrame({"text": [i for i, _ in owner], "word": [w.lower().strip(".,!?\"'()") for _, w in owner], "fall": drop})
        injected = per_word[per_word.text < len(inj)]
        top = injected.groupby("word")["fall"].agg(["mean", "size"]).query("size >= 5").sort_values("mean", ascending=False).head(8)
        flips = sum(1 for i in range(len(inj)) if base[i] > thr and (base[i] - per_word[per_word.text == i]["fall"].max()) <= thr)
        detected = int((base[:len(inj)] > thr).sum())
        # in benign prompts, the words whose removal lowers P(injection) most are the ones pushing it up
        benign_up = per_word[per_word.text >= len(inj)].groupby("word")["fall"].agg(["mean", "size"]).query("size >= 3").sort_values("mean", ascending=False).head(5)
        results["occlusion"][name] = {
            "top_words": [(w, round(float(m), 3)) for w, m in top["mean"].items()],
            "flip_share": flips / max(detected, 1), "detected": detected,
            "benign_trigger_words": [(w, round(float(m), 3)) for w, m in benign_up["mean"].items()],
        }
        # 2. position
        pos = {}
        for where in ("start", "middle", "end"):
            docs = [plant(c, a, where) for c, a in zip(carriers, attacks)]
            plain = score(docs) > thr
            win = [windows(doc) for doc in docs]
            flat = score([w for ws in win for w in ws])
            k, best = 0, []
            for ws in win:
                best.append(flat[k:k + len(ws)].max())
                k += len(ws)
            pos[where] = {"truncated": float(plain.mean()), "windowed": float((np.array(best) > thr).mean())}
        clean_win = [windows(c) for c in carriers]
        flat = score([w for ws in clean_win for w in ws])
        k, best = 0, []
        for ws in clean_win:
            best.append(flat[k:k + len(ws)].max())
            k += len(ws)
        pos["clean_fpr"] = {"truncated": float((score(carriers) > thr).mean()), "windowed": float((np.array(best) > thr).mean())}
        results["position"][name] = pos
        print(name, json.dumps(results["occlusion"][name]["top_words"][:5], ensure_ascii=False), pos, flush=True)
        del score
        torch.cuda.empty_cache()

    # 3. calibration from saved scores
    y = rows.label.to_numpy()
    cal = {}
    for name in meta:
        s = np.load(GEN / f"scores_{slug(name)}.npy")
        ok = ~np.isnan(s)
        idm = ok & (rows.suite == "slabs_test").to_numpy()
        oodm = ok & ~rows.group.isin(["id"]).to_numpy()
        cal[name] = {"id": ece(y[idm], s[idm]), "ood": ece(y[oodm], s[oodm])}
    results["calibration"] = cal
    (REPORTS / "shortcuts.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(results)


def write_report(results) -> None:
    chars = results.get("characters", {})
    var_names = list(next(iter(chars.values()))) if chars else []
    char_tab = table(["모델"] + var_names, [[n] + [f"{r[v] * 100:.1f}" for v in var_names] for n, r in chars.items()])
    occ = [[n, ", ".join(f"{w} ({m:+.2f})" for w, m in r["top_words"][:6]), f"{r['flip_share'] * 100:.0f}% (탐지 {r['detected']}개 중)",
            ", ".join(w for w, _ in r["benign_trigger_words"][:4])] for n, r in results["occlusion"].items()]
    pos = [[n] + [f"{r[w]['truncated'] * 100:.0f} → {r[w]['windowed'] * 100:.0f}" for w in ("start", "middle", "end")]
           + [f"{r['clean_fpr']['truncated'] * 100:.0f} → {r['clean_fpr']['windowed'] * 100:.0f}"] for n, r in results["position"].items()]
    cal = [[n, f"{r['id']:.3f}", f"{r['ood']:.3f}"] for n, r in results["calibration"].items()]
    report = f"""# 판정 근거 분석: 모델은 무엇에 기대는가

`python -m flyguard.shortcuts`가 만들었어요. 임계값은 모두 S-Labs validation에서 고른 그대로예요.

## 0. 글자 종류만 바꾼 정상 문장 (S-Labs test 정상 {CHAR_N}개, 인젝션으로 잘못 막은 비율 %)
- **배경:** S-Labs train에서 ASCII가 아닌 글자가 들어간 문장은 22개뿐이고, 전부 난독화 공격이에요(키릴 동형 문자, 투명 문자 등). validation·test에는 하나도 없어요.
- **실험:** 내용은 그대로 두고 무해한 글자만 덧붙였어요. 오탐률이 크게 뛰면 모델이 "낯선 글자 = 공격"이라는 지름길을 배운 거예요.
- **프랑스어 대조군:** 영어는 아니지만 ASCII 글자예요. 그래서 "언어 때문인지, 글자 때문인지"를 갈라 줘요.

{char_tab}

## 1. 단어 하나씩 지워 보기 (S-Labs test 인젝션 {OCCLUDE_N}개, 정상 {OCCLUDE_N}개)
- **지웠을 때 인젝션 점수가 가장 많이 떨어진 단어:** 괄호 안은 평균 하락폭이에요.
- **한 단어로 뒤집힘:** 탐지된 인젝션 중, 단어 하나만 지워도 정상으로 바뀐 비율이에요. 높을수록 특정 단어 하나에 기대고 있다는 뜻이에요.
- **정상 문장에서 점수를 올린 단어:** 정상 문장 안에서 인젝션 점수를 끌어올린 단어예요.

{table(["모델", "인젝션 점수를 만든 단어", "한 단어로 뒤집힘", "정상 문장에서 점수를 올린 단어"], occ)}

## 2. 긴 문서에 숨긴 지시 (PIArena 정상 문서 {POSITION_N}개에 S-Labs 인젝션을 심음, 탐지율 %)
- **"잘림 → 창 나눔":** 지금처럼 앞부분만 읽을 때의 탐지율과, 문서를 {WINDOW_CHARS}자 창으로 나눠 가장 높은 점수를 쓸 때의 탐지율이에요.
- **정상 문서 오탐률:** 인젝션을 심지 않은 원래 문서를 잘못 차단한 비율이에요.

{table(["모델", "문서 앞", "문서 중간", "문서 끝", "정상 문서 오탐률"], pos)}

## 3. 확률 보정 (기대 보정 오차 ECE, 0이면 완벽)
"인젝션일 확률 80%"라고 말한 문장들이 실제로 80% 인젝션인지 봐요. 학습 분포는 S-Labs test, 외부는 외부 묶음 전체예요.

{table(["모델", "S-Labs test", "외부 묶음"], cal)}
"""
    (REPORTS / "shortcuts.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'shortcuts.md'}")


if __name__ == "__main__":
    run()
