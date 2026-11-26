"""What the fly readout offers besides accuracy: learning from few labels, noticing unfamiliar prompts,
and learning a new attack family online.

    python -m flyguard.fly_analysis            # writes reports/fly_analysis.md (run flyguard.generalize first)
    python -m flyguard.fly_analysis --report   # rebuild the report from reports/fly_analysis.json

1. Few labels. Frozen MiniLM embeddings; n labelled S-Labs train prompts (balanced, 5 draws per n). Three
   readouts see the same prompts: the mushroom body (FlyWire wiring, dopamine counting), logistic regression
   on the same 55 glomerulus rates (same input, gradient-trained rule) and logistic regression on the full
   384-d embedding. The antennal lobe (PCA) is fitted on unlabelled train prompts. Each readout picks its one
   setting (beta for the fly, C for regression) and its threshold on S-Labs validation.
2. Familiarity. Each KC->MBON synapse is depressed whenever its KC fires, so a prompt that drives KCs the
   fly has rarely used reaches a novelty MBON through undepressed synapses (FlyMushroomBody.novelty). No
   extra model is trained. Compared with common out-of-distribution signals: cosine distance to the nearest
   training prompt and the classifiers' own uncertainty. Then a cascade: the fly answers familiar prompts and
   hands unfamiliar ones to a slower model trained on other data, against a control that hands over the
   same number of prompts at random.
3. Learning Korean online. Multilingual e5 nose. The fly adds k labelled Korean prompts to its counts
   (partial_fit: no gradient, no old data); logistic regression is refit on all of S-Labs train plus the k
   prompts. KoreanGuardrail is split by item (the 11 obfuscated variants of an item stay on one side) into an
   adaptation pool and a held-out half; JHC56 checks transfer to another Korean source and S-Labs test checks
   forgetting. Thresholds are re-picked on S-Labs validation only: no Korean labels beyond the k.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .connectome import Wiring
from .data import load_splits
from .encoders import E5Encoder, MiniLMEncoder, cached
from .external import load_suites
from .generalize import OUT as GEN
from .generalize import slug
from .metrics import best_threshold
from .mushroom_body import AntennalLobe, FlyGuard, FlyMushroomBody
from .paths import DEMO, REPORTS

OUT = REPORTS / "fly_analysis.json"
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DRAWS = 5
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
KS = [0, 16, 64, 256, 1024]
WEIGHTS = [1, 10]
FLY = "초파리 · 고정 MiniLM 코"
MODERNBERT = "ModernBERT 미세조정 1개"
EXPERTS = ["ProtectAI DeBERTa v2 (외부 전문 탐지기)", "Qwen2.5-1.5B 제로샷 (거대언어모델)", "Qwen3-4B 제로샷 (거대언어모델)"]
BUDGETS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
DEPLOY_BASES = ["ModernBERT 미세조정 1개", "하이브리드 MiniLM · 초파리 투표", "하이브리드 ModernBERT · 초파리 투표",
                "하이브리드 다국어 e5 · 초파리 투표"]
DEPLOY_EXPERTS = ["ProtectAI DeBERTa v2 (외부 전문 탐지기)", "Qwen3-4B 제로샷 (거대언어모델)"]
DEPLOY_BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
READOUT_KO = {"fly": "초파리 버섯체 (도파민 계수)", "lr_glom": "로지스틱 회귀 · 같은 55개 사구체 입력",
              "lr_embed": "로지스틱 회귀 · 384차원 임베딩 전체", "lr_e5": "로지스틱 회귀 · 768차원 임베딩 전체"}


def ba(y, pred):
    parts = ([pred[y == 1].mean()] if (y == 1).any() else []) + ([1 - pred[y == 0].mean()] if (y == 0).any() else [])
    return float(np.mean(parts))


def lr_select(X, y, Xv, yv):
    """C chosen on validation AUC, as the fly chooses beta."""
    best = max(CS, key=lambda C: roc_auc_score(yv, LogisticRegression(C=C, max_iter=3000).fit(X, y).decision_function(Xv)))
    return best


def lr_fit(X, y, Xv, yv, C, sample_weight=None):
    m = LogisticRegression(C=C, max_iter=3000).fit(X, y, sample_weight=sample_weight)
    return m, best_threshold(yv, m.decision_function(Xv))


# 1. few labels -----------------------------------------------------------------------------------
def few_labels(d, wiring, enc) -> list[dict]:
    E = {s: cached(enc, s, d[s].text) for s in ("train", "validation", "test")}
    y = {s: d[s].label.to_numpy() for s in E}
    lobe = AntennalLobe(wiring.n_glom, 0).fit(E["train"])
    P = {s: lobe.transform(E[s]) for s in E}
    pos, neg = np.flatnonzero(y["train"] == 1), np.flatnonzero(y["train"] == 0)
    out = []
    for n in SIZES + [len(y["train"])]:
        full = n == len(y["train"])
        for seed in range(1 if full else DRAWS):
            rng = np.random.default_rng(seed)
            idx = np.arange(n) if full else np.concatenate([rng.choice(pos, n // 2, replace=False), rng.choice(neg, n - n // 2, replace=False)])
            yt = y["train"][idx]
            mb = FlyMushroomBody(wiring, homeostasis=False).fit(P["train"][idx], yt).calibrate(P["validation"], y["validation"])
            s = mb.score(P["test"])
            out.append({"n": n, "seed": seed, "model": "fly", "auc": roc_auc_score(y["test"], s), "acc": float(np.mean((s > mb.theta) == y["test"]))})
            for name, X in (("lr_glom", P), ("lr_embed", E)):
                C = lr_select(X["train"][idx], yt, X["validation"], y["validation"])
                m, t = lr_fit(X["train"][idx], yt, X["validation"], y["validation"], C)
                s = m.decision_function(X["test"])
                out.append({"n": n, "seed": seed, "model": name, "auc": roc_auc_score(y["test"], s), "acc": float(np.mean((s > t) == y["test"]))})
        last = pd.DataFrame(out[-3 * (1 if full else DRAWS):]).groupby("model").auc.mean().round(3).to_dict()
        print("few labels", n, last, flush=True)
    return out


# 2. familiarity and the cascade ----------------------------------------------------------------
def familiarity(d, enc) -> dict:
    rows = pd.read_parquet(GEN / "rows.parquet")
    meta = json.loads((GEN / "models.json").read_text(encoding="utf-8"))
    score = {m: np.load(GEN / f"scores_{slug(m)}.npy") for m in meta}
    y = rows.label.to_numpy()
    E = cached(enc, "generalization_rows", rows.text)
    E_train = cached(enc, "train", d["train"].text)
    guard = FlyGuard.load(DEMO, enc)
    H = guard.mb.kc_code(guard.lobe.transform(E))
    knn = np.concatenate([1 - (E[i:i + 2048] @ E_train.T).max(1) for i in range(0, len(E), 2048)])
    fly_p, fly_t = score[FLY], meta[FLY]["threshold"]
    mb_p = score[MODERNBERT]
    centred = E - guard.lobe.mean
    Z = centred @ guard.lobe.components.T       # what the antennal lobe keeps (28 principal directions)
    residual = np.sqrt(np.maximum((centred ** 2).sum(1) - (Z ** 2).sum(1), 0))
    signals = {
        "초파리 새로움 (버섯체 MBON)": guard.mb.novelty(H),
        "가장 가까운 학습 문장까지 거리": knn,
        "안테나엽이 버린 성분 크기 (PCA 잔차)": residual,
        "초파리 자기 불확실성 (임계값과의 거리)": -np.abs(fly_p - fly_t),
        "ModernBERT 불확실성": -np.abs(mb_p - meta[MODERNBERT]["threshold"]),
    }
    suite, group = rows.suite.to_numpy(), rows.group.to_numpy()
    test, val = suite == "slabs_test", suite == "slabs_validation"
    ext = group != "id"
    detect = {}
    for name, sig in signals.items():
        detect[name] = {g: float(roc_auc_score(np.r_[np.zeros(test.sum()), np.ones((group == g).sum())],
                                               np.r_[sig[test], sig[group == g]]))
                        for g in pd.unique(group[ext])}
    wrong = (fly_p > fly_t) != y
    errors = {name: {"external": float(roc_auc_score(wrong[ext], sig[ext])), "slabs_test": float(roc_auc_score(wrong[test], sig[test]))}
              for name, sig in signals.items()}

    def summary(pred, mask):
        ext_suites = [s for s in pd.unique(suite[mask & ext])]
        per = {s: ba(y[mask & (suite == s)], pred[mask & (suite == s)]) for s in ext_suites}
        korean = [per[s] for s in ext_suites if group[suite == s][0] == "korean"]
        over = [per[s] for s in ext_suites if group[suite == s][0] == "over_defense"]
        return {"test_acc": float(np.mean(pred[mask & test] == y[mask & test])),
                "external_ba": float(np.mean(list(per.values()))), "korean_ba": float(np.mean(korean)) if korean else None,
                "over_defense_ba": float(np.mean(over)) if over else None}

    def random_control(up, evalm, exp_pred, base_pred):
        """Hand over the same number of prompts, chosen at random (5 draws)."""
        m, idx = int(up[evalm].sum()), np.flatnonzero(evalm)
        draws = []
        for _ in range(5):
            r = np.zeros(len(y), bool)
            r[rng.choice(idx, m, replace=False)] = True
            draws.append(summary(np.where(r, exp_pred, base_pred), evalm))
        return {k: float(np.mean([x[k] for x in draws])) if draws[0][k] is not None else None for k in draws[0]}

    cascade = {}
    rng = np.random.default_rng(0)
    for expert in EXPERTS:
        if expert not in score:
            continue
        have = ~np.isnan(score[expert])
        evalm = have & (test | ext)
        fly_pred = fly_p > fly_t
        exp_pred = score[expert] > meta[expert]["threshold"]
        res = []
        for route_name, sig in (("novelty", signals["초파리 새로움 (버섯체 MBON)"]), ("knn", knn)):
            for q in BUDGETS:
                tau = -np.inf if q == 1.0 else np.inf if q == 0.0 else np.quantile(sig[val & have], 1 - q)
                up = sig > tau
                pred = np.where(up, exp_pred, fly_pred)
                rec = {"route": route_name, "budget": q, **summary(pred, evalm),
                       "escalated_test": float(up[evalm & test].mean()), "escalated_external": float(up[evalm & ext].mean()),
                       "latency_ms": meta[FLY]["latency_ms"] + float(up[evalm & ext].mean()) * meta[expert]["latency_ms"]}
                if route_name == "novelty" and 0 < q < 1:
                    rec["random"] = random_control(up, evalm, exp_pred, fly_pred)
                res.append(rec)
        cascade[expert] = res
        print("cascade", expert, [(r["budget"], round(r["external_ba"], 3)) for r in res if r["route"] == "novelty"], flush=True)

    # deployable pairs: an S-Labs-trained model answers familiar prompts (nearest-neighbour distance), a model
    # trained on other data answers the rest. Routing needs the frozen MiniLM embedding, so its cost is added.
    deploy = []
    route_ms = meta["MiniLM 고정 + 로지스틱 회귀"]["latency_ms"]
    for expert in DEPLOY_EXPERTS:
        if expert not in score:
            continue
        have = ~np.isnan(score[expert])
        evalm = have & (test | ext)
        exp_pred = score[expert] > meta[expert]["threshold"]
        for base in DEPLOY_BASES:
            base_pred = score[base] > meta[base]["threshold"]
            for q in DEPLOY_BUDGETS:
                tau = -np.inf if q == 1.0 else np.inf if q == 0.0 else np.quantile(knn[val & have], 1 - q)
                up = knn > tau
                esc = float(up[evalm & ext].mean())
                latency = (meta[expert]["latency_ms"] if q == 1.0 else meta[base]["latency_ms"] if q == 0.0
                           else meta[base]["latency_ms"] + route_ms + esc * meta[expert]["latency_ms"])
                rec = {"expert": expert, "base": base, "budget": q, **summary(np.where(up, exp_pred, base_pred), evalm),
                       "escalated_test": float(up[evalm & test].mean()), "escalated_external": esc, "latency_ms": latency}
                if 0 < q < 1:
                    rec["random"] = random_control(up, evalm, exp_pred, base_pred)
                deploy.append(rec)
            print("deploy", base, "->", expert, [(r["budget"], round(r["external_ba"], 3), round(r["test_acc"], 3))
                                                 for r in deploy if r["base"] == base and r["expert"] == expert], flush=True)
    return {"detect": detect, "errors": errors, "cascade": cascade, "deploy": deploy,
            "fly_latency_ms": meta[FLY]["latency_ms"], "expert_latency_ms": {e: meta[e]["latency_ms"] for e in EXPERTS if e in meta}}


# 3. learning Korean online ---------------------------------------------------------------------
def online_korean(d, wiring) -> list[dict]:
    ext = load_suites()
    kg = ext[ext.suite == "koreanguardrail"].reset_index(drop=True)
    jh = ext[(ext.suite == "jhc56_registers") & (ext.register != "english")].reset_index(drop=True)
    enc = E5Encoder()
    E = {s: cached(enc, s, d[s].text) for s in ("train", "validation", "test")}
    E["kg"], E["jh"] = cached(enc, "koreanguardrail", kg.text), cached(enc, "jhc56_korean", jh.text)
    del enc
    y = {s: d[s].label.to_numpy() for s in ("train", "validation", "test")}
    y["kg"], y["jh"] = kg.label.to_numpy(), jh.label.to_numpy()
    items = np.sort(kg["item"].unique())
    pool = kg["item"].isin(set(np.random.default_rng(0).choice(items, len(items) // 2, replace=False))).to_numpy()
    lobe = AntennalLobe(wiring.n_glom, 0).fit(E["train"])
    P = {s: lobe.transform(E[s]) for s in E}
    base = FlyMushroomBody(wiring, homeostasis=False).fit(P["train"], y["train"]).calibrate(P["validation"], y["validation"])
    H = {s: base.kc_code(P[s]) for s in ("validation", "test", "kg", "jh")}
    C = {"lr_glom": lr_select(P["train"], y["train"], P["validation"], y["validation"]),
         "lr_e5": lr_select(E["train"], y["train"], E["validation"], y["validation"])}
    X = {"lr_glom": P, "lr_e5": E}
    held = ~pool

    def measure(scores, thr):
        return {"kg_ba": ba(y["kg"][held], scores["kg"][held] > thr), "kg_auc": float(roc_auc_score(y["kg"][held], scores["kg"][held])),
                "jh_ba": ba(y["jh"], scores["jh"] > thr), "jh_auc": float(roc_auc_score(y["jh"], scores["jh"])),
                "test_acc": float(np.mean((scores["test"] > thr) == y["test"]))}

    out = []
    pos, neg = np.flatnonzero(pool & (y["kg"] == 1)), np.flatnonzero(pool & (y["kg"] == 0))
    for k in KS:
        for w in (WEIGHTS if k else [1]):
            for seed in range(DRAWS if k else 1):
                rng = np.random.default_rng(seed)
                idx = np.concatenate([rng.choice(pos, k // 2, replace=False), rng.choice(neg, k - k // 2, replace=False)]) if k else np.array([], int)
                yk = y["kg"][idx]
                # the fly: add the new prompts to the counts, then re-pick the threshold on S-Labs validation
                t0 = time.perf_counter()
                mb = copy.copy(base)
                mb.n0, mb.n1 = base.n0.copy(), base.n1.copy()
                if k:
                    mb.partial_fit(P["kg"][idx], yk, weight=w)
                mb.theta = best_threshold(y["validation"], mb._score(H["validation"]))
                ms = (time.perf_counter() - t0) * 1000
                wa0, wv0 = base.weights()
                wa1, wv1 = mb.weights()
                changed = float(np.mean((np.abs(wa1 - wa0) > 0.05) | (np.abs(wv1 - wv0) > 0.05)))
                out.append({"k": k, "weight": w, "seed": seed, "model": "fly", "update_ms": ms, "kc_changed": changed,
                            **measure({s: mb._score(H[s]) for s in H}, mb.theta)})
                for name in ("lr_glom", "lr_e5"):
                    Xs = X[name]
                    t0 = time.perf_counter()
                    Xt = np.vstack([Xs["train"], Xs["kg"][idx]])
                    yt = np.r_[y["train"], yk]
                    sw = np.r_[np.ones(len(y["train"])), np.full(k, float(w))]
                    m, t = lr_fit(Xt, yt, Xs["validation"], y["validation"], C[name], sample_weight=sw)
                    ms = (time.perf_counter() - t0) * 1000
                    out.append({"k": k, "weight": w, "seed": seed, "model": name, "update_ms": ms,
                                **measure({s: m.decision_function(Xs[s]) for s in ("kg", "jh", "test")}, t)})
            last = pd.DataFrame([r for r in out if r["k"] == k and r["weight"] == w]).groupby("model")[["kg_ba", "jh_ba", "test_acc"]].mean().round(3)
            print("online korean k", k, "weight", w, last.to_dict("index"), flush=True)
    return out


# 4. a residual glomerulus ----------------------------------------------------------------------
class ResidualLobe(AntennalLobe):
    """The demo's antennal lobe with one glomerulus re-wired to carry what PCA throws away: the size of the
    embedding's residual outside the kept principal directions, scaled to that glomerulus's usual rate."""

    def __init__(self, base: AntennalLobe, slot: int, E_train: np.ndarray):
        super().__init__(base.n_glom, base.seed)
        self.mean, self.components, self.channel, self.slot = base.mean, base.components, base.channel, slot
        usual = np.percentile(super().transform(E_train)[:, slot], 95)
        self.scale = usual / max(np.percentile(self.residual(E_train), 95), 1e-9)

    def residual(self, E: np.ndarray) -> np.ndarray:
        c = E - self.mean
        z = c @ self.components.T
        return np.sqrt(np.maximum((c ** 2).sum(1) - (z ** 2).sum(1), 0))

    def transform(self, E: np.ndarray) -> np.ndarray:
        P = super().transform(E)
        P[:, self.slot] = (self.residual(E) * self.scale).astype(np.float32)
        return P


def residual_channel(d, enc) -> dict:
    """Does the novelty MBON see unfamiliar prompts once one glomerulus carries the PCA residual?
    Same wiring, same settings, same training; only the input of the most widely connected glomerulus changes."""
    rows = pd.read_parquet(GEN / "rows.parquet")
    E = cached(enc, "generalization_rows", rows.text)
    Es = {s: cached(enc, s, d[s].text) for s in ("train", "validation", "test")}
    y = {s: d[s].label.to_numpy() for s in Es}
    demo = FlyGuard.load(DEMO, enc)
    slot = int(np.asarray(demo.mb.wiring.W.sum(0)).ravel().argmax())
    suite, group = rows.suite.to_numpy(), rows.group.to_numpy()
    test, ext = suite == "slabs_test", group != "id"
    out = {"slot": slot, "slot_kcs": int(np.asarray((demo.mb.wiring.W[:, slot] > 0).sum()))}
    for name, lobe in (("demo", demo.lobe), ("residual", ResidualLobe(demo.lobe, slot, Es["train"]))):
        P = {s: lobe.transform(Es[s]) for s in Es}
        mb = FlyMushroomBody(demo.mb.wiring, sparsity=demo.mb.sparsity, homeostasis=demo.mb.homeostasis)
        mb.fit(P["train"], y["train"]).calibrate(P["validation"], y["validation"])
        nov = mb.novelty(mb.kc_code(lobe.transform(E)))
        out[name] = {
            "test_acc": float(np.mean(mb.predict(P["test"]) == y["test"])),
            "detect": {g: float(roc_auc_score(np.r_[np.zeros(test.sum()), np.ones((group == g).sum())],
                                               np.r_[nov[test], nov[group == g]])) for g in pd.unique(group[ext])},
        }
        out[name]["detect_mean"] = float(np.mean(list(out[name]["detect"].values())))
        print("residual channel", name, round(out[name]["test_acc"], 3), round(out[name]["detect_mean"], 3), flush=True)
    return out


# report ------------------------------------------------------------------------------------------
def table(header, body):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(lines)


def pct(v):
    return "–" if v is None else f"{v * 100:.1f}"


def mean_sd(series):
    return f"{series.mean() * 100:.1f}" + (f" ± {series.std() * 100:.1f}" if len(series) > 1 else "")


def section_few(rows) -> str:
    few = pd.DataFrame(rows)
    body = []
    for n, g in few.groupby("n"):
        cells = [f"{n:,}"]
        for metric in ("auc", "acc"):
            cells += [mean_sd(g[g.model == m][metric]) for m in ("fly", "lr_glom", "lr_embed")]
        body.append(cells)
    tab = table(["학습 문장 수"] + [f"{READOUT_KO[m]} AUC" for m in ("fly", "lr_glom", "lr_embed")]
                + [f"{READOUT_KO[m]} 정확도" for m in ("fly", "lr_glom", "lr_embed")], body)
    return f"""## 1. 라벨이 적을 때 (같은 고정 MiniLM 입력, S-Labs test, 평균 ± 표준편차, {DRAWS}번 추출)
- **초파리 vs 같은 55개 사구체 입력의 로지스틱 회귀:** 입력이 완전히 같으니 학습 규칙(도파민 계수 vs 경사하강)만 비교돼요.
- **384차원 임베딩 전체:** 55개 사구체로 압축하면서 잃는 정보가 얼마인지 보여 줘요.
- **설정 선택:** 각 판독기는 설정 하나(초파리는 도파민 세기 β, 회귀는 규제 C)와 임계값을 S-Labs validation에서 골라요.

{tab}
"""


def section_familiarity(fam) -> str:
    groups = list(next(iter(fam["detect"].values())))
    gko = {"over_defense": "과잉 방어", "direct": "직접", "jailbreak": "탈옥", "agentic": "에이전트", "indirect": "간접",
           "mixed": "혼합", "korean": "한국어"}
    det = table(["신호"] + [gko.get(g, g) for g in groups] + ["평균"],
                [[n] + [f"{v[g]:.2f}" for g in groups] + [f"{np.mean(list(v.values())):.2f}"] for n, v in fam["detect"].items()])
    err = table(["신호", "외부 묶음에서 초파리 오답 예측 AUC", "S-Labs test에서 초파리 오답 예측 AUC"],
                [[n, f"{v['external']:.2f}", f"{v['slabs_test']:.2f}"] for n, v in fam["errors"].items()])
    casc = []
    for expert, rs in fam["cascade"].items():
        body = []
        for r in rs:
            if r["route"] != "novelty":
                continue
            knn = next(x for x in rs if x["route"] == "knn" and x["budget"] == r["budget"])
            rnd = r.get("random")
            body.append([f"{r['budget'] * 100:.0f}%", f"{r['escalated_external'] * 100:.0f}% / {r['escalated_test'] * 100:.0f}%",
                         pct(r["test_acc"]), pct(r["external_ba"]), pct(r["korean_ba"]),
                         pct(knn["external_ba"]), "–" if rnd is None else pct(rnd["external_ba"]), f"{r['latency_ms']:.0f}"])
        casc.append(f"**전문가 = {expert}** (전문가 혼자일 때 지연 {fam['expert_latency_ms'][expert]:.0f}ms, 초파리 {fam['fly_latency_ms']:.0f}ms)\n\n" + table(
            ["넘기는 비율 (S-Labs validation 기준)", "실제로 넘긴 비율 (외부 / S-Labs test)", "S-Labs test 정확도", "외부 평균 BA",
             "한국어 BA", "같은 비율, 가까운 이웃 거리로 넘기기", "같은 개수, 무작위로 넘기기", "외부 평균 지연(ms)"], body))
    casc = "\n\n".join(casc)
    deploy = ""
    if fam.get("deploy"):
        body = []
        for r in fam["deploy"]:
            rnd = r.get("random")
            who = (r["expert"].split(" (")[0] + "만" if r["budget"] == 1.0 else r["base"] + "만" if r["budget"] == 0.0
                   else f"{r['base']} → {r['expert'].split(' (')[0]}")
            body.append([who, f"{r['budget'] * 100:.0f}%", f"{r['escalated_external'] * 100:.0f}% / {r['escalated_test'] * 100:.0f}%",
                         pct(r["test_acc"]), pct(r["external_ba"]), pct(r["korean_ba"]), pct(r["over_defense_ba"]),
                         "–" if rnd is None else pct(rnd["external_ba"]), f"{r['latency_ms']:.0f}"])
        deploy = "### 2-4. 배포 조합: S-Labs로 학습한 모델 + 낯선 문장만 다른 모델에 넘기기\n" + (
            "- **흐름:** 가까운 이웃 거리(고정 MiniLM)로 낯선 문장을 골라 넘겨요. 기준선은 S-Labs validation 문장 중 넘기는 비율로 정했어요.\n"
            "- **비용:** 지연에는 기준 모델, 거리 계산(MiniLM), 넘긴 비율만큼의 전문 모델 시간이 다 들어가요.\n\n") + table(
            ["조합", "넘기는 비율 (S-Labs validation 기준)", "실제로 넘긴 비율 (외부 / S-Labs test)", "S-Labs test 정확도", "외부 평균 BA",
             "한국어 BA", "과잉 방어 BA", "같은 개수, 무작위로 넘기기", "외부 평균 지연(ms)"], body)
    return f"""## 2. 낯선 문장을 알아채는가

### 2-1. 학습 분포 밖 문장 구별 (S-Labs test 대비 각 외부 유형, AUC)
- **버섯체 새로움 신호:** 학습 중 자주 발화한 KC일수록 시냅스가 약해져요. 그래서 낯선 문장은 약해지지 않은 시냅스를 통해 강한 신호를 내요. 추가 모델 없이 회로 안에서 나와요.
- **읽는 법:** 0.5는 구별 못 함, 1.0은 완벽히 구별한다는 뜻이에요.

{det}

### 2-2. 초파리가 틀릴 문장을 미리 아는가 (AUC)
{err}

### 2-3. 낯선 문장만 큰 모델에 넘기기 (계단식 판정)
- **흐름:** 초파리가 먼저 판정하고, 새로움 신호가 높은 문장만 학습 데이터가 다른 전문 모델에 넘겨요.
- **비교 대조군:** 같은 개수를 무작위로 넘겼을 때, 그리고 가까운 이웃 거리로 넘겼을 때와 비교해요.

{casc}

{deploy}
"""


def section_online(rows) -> str:
    onl = pd.DataFrame(rows)
    parts = []
    for metric, title in (("kg_ba", "KoreanGuardrail 보지 않은 문항 (균형 정확도 %)"), ("kg_auc", "KoreanGuardrail 보지 않은 문항 (AUC %)"),
                          ("jh_ba", "JHC56 한국어 말투 7종 (다른 출처, 균형 정확도 %)"), ("test_acc", "S-Labs test 정확도 % (잊어버림 확인)")):
        body = []
        for model in ("fly", "lr_glom", "lr_e5"):
            for w in WEIGHTS:
                cells = [f"{READOUT_KO[model]} · 새 문장 가중치 {w}"]
                for k in KS:
                    g = onl[(onl.model == model) & (onl.k == k) & ((onl.weight == w) | (onl.k == 0))]
                    cells.append(mean_sd(g[metric]))
                body.append(cells)
        parts.append(f"**{title}**\n\n" + table(["판독기"] + [f"k={k}" for k in KS], body))
    cost = onl[onl.k > 0].groupby("model").update_ms.median()
    changed = onl[(onl.model == "fly") & (onl.k > 0)].groupby(["k", "weight"]).kc_changed.mean()
    cost_line = ", ".join(f"{READOUT_KO[m]} {v:,.0f}ms" for m, v in cost.items())
    changed_line = ", ".join(f"k={k}·가중치 {w}: {v * 100:.1f}%" for (k, w), v in changed.items())
    body = "\n\n".join(parts)
    return f"""## 3. 한국어 공격을 그 자리에서 배우기 (다국어 e5 코, k = 새로 본 한국어 라벨 수, {DRAWS}번 추출 평균)
- **초파리:** 새 문장의 KC 발화 횟수만 더해요. 경사하강도, 예전 데이터도 필요 없어요.
- **로지스틱 회귀:** S-Labs train 전체에 새 문장을 더해 처음부터 다시 학습해요.
- **"새 문장 가중치 10":** 새 문장 하나를 10개처럼 세요. 두 방법 모두 똑같이 적용했어요.
- **문항 단위 분할:** KoreanGuardrail은 문항 단위로 반을 나눠, 학습에 쓴 문항의 변형은 평가에 나오지 않아요.

{body}

- **업데이트 1번 걸린 시간 (중앙값):** {cost_line}
- **업데이트로 시냅스가 0.05 이상 바뀐 KC 비율 (초파리):** {changed_line}
"""


def section_residual(r) -> str:
    gko = {"over_defense": "과잉 방어", "direct": "직접", "jailbreak": "탈옥", "agentic": "에이전트", "indirect": "간접",
           "mixed": "혼합", "korean": "한국어"}
    groups = list(r["demo"]["detect"])
    body = [[label, pct(r[k]["test_acc"])] + [f"{r[k]['detect'][g]:.2f}" for g in groups] + [f"{r[k]['detect_mean']:.2f}"]
            for k, label in (("demo", "지금 초파리 (PCA 방향만)"), ("residual", "버린 성분 사구체 추가"))]
    tab = table(["안테나엽", "S-Labs test 정확도"] + [gko.get(g, g) for g in groups] + ["평균"], body)
    return f"""## 4. 안테나엽에 "버린 성분" 사구체 넣기 (새로움 신호 AUC, S-Labs test 대비 각 외부 유형)
- **바꾼 것:** 연결된 KC가 가장 많은 사구체 1개(KC {r['slot_kcs']}개와 연결)의 입력을, PCA가 버린 성분의 크기로 바꿨어요. 배선·학습 규칙·설정은 그대로예요.
- **이유:** 새로운 문장은 PCA가 버린 방향에서 달라서, 지금 초파리의 새로움 신호(MBON)는 그 차이를 볼 수 없었어요.

{tab}
"""


def write_report(res: dict) -> None:
    sections = []
    if "few_labels" in res:
        sections.append(section_few(res["few_labels"]))
    if "familiarity" in res:
        sections.append(section_familiarity(res["familiarity"]))
    if "online_korean" in res:
        sections.append(section_online(res["online_korean"]))
    if "residual" in res:
        sections.append(section_residual(res["residual"]))
    report = ("# 초파리 판독기만의 성질: 적은 라벨, 낯섦 감지, 온라인 학습\n\n"
              "`python -m flyguard.fly_analysis`가 만들었어요. 정확도 경쟁이 아니라, 초파리 회로가 구조적으로 무엇을 다르게 할 수 있는지 재요.\n\n"
              + "\n".join(sections))
    (REPORTS / "fly_analysis.md").write_text(report, encoding="utf-8")
    print(f"wrote {REPORTS / 'fly_analysis.md'}")


def run(parts) -> None:
    d = load_splits()
    wiring = Wiring.load(DEMO / "wiring.npz")
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if "few" in parts or "familiarity" in parts:
        enc = MiniLMEncoder()
        if "few" in parts:
            res["few_labels"] = few_labels(d, wiring, enc)
        if "familiarity" in parts:
            res["familiarity"] = familiarity(d, enc)
        del enc
    if "residual" in parts:
        res["residual"] = residual_channel(d, MiniLMEncoder())
    if "online" in parts:
        res["online_korean"] = online_korean(d, wiring)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["few", "familiarity", "online"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        write_report(json.loads(OUT.read_text(encoding="utf-8")))
    else:
        run(args.parts)
