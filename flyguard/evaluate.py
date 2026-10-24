"""Results table, controls, lesions and curves -> reports/results.md (+ results.json).

    python -m flyguard.evaluate            # validation only; safe to run any time
    python -m flyguard.evaluate --final    # adds test and test-clean, writes reports/batch_test.json

Thresholds, the dopamine dose and every other choice come from validation. Each run is appended to
reports/runs.jsonl, so the number of looks at test stays visible to the whole team.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .baselines import shallow_model, tfidf_model
from .build import DEMO, FLYWIRE_WIRING, batch_items, comparator
from .connectome import Wiring
from .data import describe, load_splits
from .encoders import CharNgramEncoder, MiniLMEncoder, cached
from .hybrid import OUT as HYBRID_OUT
from .hybrid import HybridGuard, available
from .metrics import best_threshold, binary_metrics
from .mushroom_body import AntennalLobe, FlyGuard, FlyMushroomBody
from .paths import REPORTS, ensure_dirs

SEEDS = range(5)
Y_VAL: np.ndarray | None = None   # validation labels, set in main()


class Table:
    def __init__(self, d, final: bool):
        self.final = final
        self.y = {s: d[s].label.to_numpy() for s in ("train", "validation", "test")}
        self.clean = ~d["test"].in_validation.to_numpy()
        self.rows = []

    def metrics(self, scores: dict, threshold: float) -> dict:
        out = {"validation": binary_metrics(self.y["validation"], scores["validation"], threshold)}
        if self.final:
            out["test"] = binary_metrics(self.y["test"], scores["test"], threshold)
            out["test_clean"] = binary_metrics(self.y["test"][self.clean], scores["test"][self.clean], threshold)
        return out

    def add(self, group: str, name: str, runs: list[tuple[dict, float]], note: str = "") -> list[dict]:
        """runs: one (scores per split, threshold) pair per seed."""
        per_seed = [self.metrics(s, t) for s, t in runs]
        row = {"group": group, "model": name, "note": note, "seeds": len(runs)}
        for split in per_seed[0]:
            for k in ("acc", "f1", "auc"):
                v = [m[split][k] for m in per_seed]
                row[f"{split}_{k}"] = float(np.mean(v))
                row[f"{split}_{k}_sd"] = float(np.std(v))
        self.rows.append(row)
        print(f"{group:<10} {name:<46} val acc={row['validation_acc']:.3f}"
              + (f"  test acc={row['test_acc']:.3f}" if self.final else ""), flush=True)
        return per_seed

    def markdown(self) -> str:
        cols = [("validation_acc", "val acc"), ("validation_f1", "val F1")]
        if self.final:
            cols += [("test_acc", "test acc"), ("test_f1", "test F1"), ("test_auc", "test AUC"), ("test_clean_acc", "test-clean acc")]
        head = "| 구분 | 모델 | " + " | ".join(c[1] for c in cols) + " | 비고 |"
        lines = [head, "|" + "---|" * (len(cols) + 3)]
        for r in self.rows:
            cells = []
            for key, _ in cols:
                sd = r.get(f"{key}_sd", 0)
                cells.append(f"{r[key]:.3f}" + (f" ± {sd:.3f}" if r["seeds"] > 1 else ""))
            lines.append(f"| {r['group']} | {r['model']} | " + " | ".join(cells) + f" | {r['note']} |")
        return "\n".join(lines)


def lr_scores(X: dict, y, C: float = 4.0) -> tuple[dict, float]:
    lr = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(X["train"], y)
    scores = {s: lr.predict_proba(X[s])[:, 1] for s in X if s != "train"}
    return scores, best_threshold(Y_VAL, scores["validation"])


def fit_mb(E: dict, wiring: Wiring, seed: int, y, train_idx=None, **kw) -> tuple[FlyMushroomBody, dict]:
    lobe = AntennalLobe(wiring.n_glom, seed).fit(E["train"])
    P = {s: lobe.transform(E[s]) for s in E}
    Ptr, ytr = (P["train"], y) if train_idx is None else (P["train"][train_idx], y[train_idx])
    mb = FlyMushroomBody(wiring, **kw).fit(Ptr, ytr).calibrate(P["validation"], Y_VAL)
    return mb, P


def mb_scores(mb: FlyMushroomBody, P: dict) -> tuple[dict, float]:
    return {s: mb.score(P[s]) for s in P if s != "train"}, mb.theta


def mcnemar(y, pred_a, pred_b) -> float:
    a_only = int(np.sum((pred_a == y) & (pred_b != y)))
    b_only = int(np.sum((pred_a != y) & (pred_b == y)))
    return 1.0 if a_only + b_only == 0 else binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue


def main() -> None:
    global Y_VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="also score test (log every use)")
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()

    d = load_splits()
    splits = ("train", "validation", "test") if args.final else ("train", "validation")
    ytr, Y_VAL = d["train"].label.to_numpy(), d["validation"].label.to_numpy()
    table = Table(d, args.final)
    texts = {s: d[s].text for s in splits}
    sections = {}

    # Shortcut probe: surface statistics only.
    probe = shallow_model()
    cv = cross_val_score(probe, list(d["train"].text), ytr, cv=5).mean()
    probe.fit(list(d["train"].text), ytr)
    sections["shortcut"] = (f"길이·문장부호·대문자·숫자 같은 겉모양 특징만 쓴 모델: train 5-fold 교차검증 정확도 {cv:.3f}, "
                            f"validation 정확도 {binary_metrics(Y_VAL, probe.predict_proba(list(d['validation'].text))[:, 1], 0.5)['acc']:.3f}. "
                            "train에만 있는 지름길이므로 길이에 기대는 모델은 test에서 손해를 봅니다.")
    table.add("기준", "겉모양 특징만 (지름길 점검)", [({s: probe.predict_proba(list(texts[s]))[:, 1] for s in splits if s != "train"}, 0.5)])

    tfidf = tfidf_model().fit(d["train"].text, ytr)
    tf = {s: tfidf.predict_proba(texts[s])[:, 1] for s in splits if s != "train"}
    table.add("기준", "TF-IDF(단어+문자) + 로지스틱 회귀", [(tf, best_threshold(Y_VAL, tf["validation"]))])

    name, comp_proba, comp_threshold = comparator()
    if name != "TF-IDF":
        ft = {s: comp_proba(texts[s]) for s in splits if s != "train"}
        table.add("트랙 A", f"{name} 미세조정", [(ft, comp_threshold)], "리더보드용")

    enc = MiniLMEncoder()
    E = {s: cached(enc, s, texts[s]) for s in splits}
    table.add("고정 인코더", "MiniLM + 로지스틱 회귀 (선형 판독)", [lr_scores(E, ytr)])

    real = Wiring.load(FLYWIRE_WIRING) if FLYWIRE_WIRING.exists() else None
    demo_meta = json.loads((DEMO / "meta.json").read_text())
    weight = demo_meta["wiring"].rsplit("[", 1)[-1].rstrip("]") if real is not None and "[" in demo_meta["wiring"] else "binary"
    hom = demo_meta["homeostasis"]
    base = real.transformed(weight) if real is not None else Wiring.random()
    claws = max(1, round(base.W.nnz / base.n_kc))
    random_w = lambda s: Wiring.random(base.n_kc, base.n_glom, claws=claws, seed=100 + s)
    label = "실제 FlyWire" if real is not None else "무작위"

    # Where is the bottleneck? Linear readouts of the glomeruli and of the KC code.
    mb0, P0 = fit_mb(E, base, 0, ytr, homeostasis=hom)
    table.add("대조군", f"사구체 {base.n_glom}채널 + 로지스틱 회귀", [lr_scores(P0, ytr)], "안테나엽까지만")
    H0 = {s: mb0.kc_code(P0[s]) for s in P0}
    table.add("대조군", f"KC 코드 {base.n_kc}개 + 로지스틱 회귀", [lr_scores(H0, ytr, C=1.0)], "도파민 규칙 대신 최적 판독")

    # The fly circuit with real, shuffled and random wiring, paired by antennal-lobe seed.
    runs = {}
    for key, make in [("real", lambda s: base), ("shuffled", lambda s: base.shuffled(seed=100 + s)), ("random", random_w)]:
        runs[key] = [fit_mb(E, make(s), s, ytr, homeostasis=hom) for s in SEEDS]
    note = f"시냅스 가중치 {weight}, 항상성 {'켬' if hom else '끔'}"
    per = {}
    per["real"] = table.add("초파리", f"버섯체: {label} 배선", [mb_scores(*r) for r in runs["real"]], note)
    per["shuffled"] = table.add("초파리", "버섯체: 차수 보존 셔플 배선", [mb_scores(*r) for r in runs["shuffled"]], "연결 상대만 무작위")
    per["random"] = table.add("초파리", f"버섯체: 무작위 배선 ({claws}-claw)", [mb_scores(*r) for r in runs["random"]], "같은 KC·사구체 수")

    split = "test" if args.final else "validation"
    y_cmp = d[split].label.to_numpy()
    lines = []
    for other in ("shuffled", "random"):
        diffs, ps = [], []
        for (mb_a, P_a), (mb_b, P_b) in zip(runs["real"], runs[other]):
            pa, pb = mb_a.predict(P_a[split]), mb_b.predict(P_b[split])
            diffs.append(np.mean(pa == y_cmp) - np.mean(pb == y_cmp))
            ps.append(mcnemar(y_cmp, pa, pb))
        lines.append(f"- 실제 − {'셔플' if other == 'shuffled' else '무작위'}: 정확도 차이 {np.mean(diffs) * 100:+.2f}%p "
                     f"(시드별 {', '.join(f'{x * 100:+.1f}' for x in diffs)}), McNemar p<0.05인 시드 {sum(p < 0.05 for p in ps)}/{len(ps)}개 ({split})")
    sections["wiring"] = "\n".join(lines)

    # Lesions.
    table.add("손상", "APL 억제 제거 (KC가 평소보다 세면 모두 발화)",
              [mb_scores(*fit_mb(E, base, s, ytr, homeostasis=hom, apl=False)) for s in SEEDS])
    table.add("변형", f"항상성 {'끔' if hom else '켬'}",
              [mb_scores(*fit_mb(E, base, s, ytr, homeostasis=not hom)) for s in SEEDS])

    # No transformer at all: character n-grams as the nose.
    cng = CharNgramEncoder(dim=max(50, base.n_glom)).fit(d["train"].text)
    C = {s: cng.encode(texts[s]) for s in splits}
    table.add("초파리", "버섯체: 트랜스포머 없음 (글자 n-gram 코)", [mb_scores(*fit_mb(C, base, s, ytr, homeostasis=hom)) for s in SEEDS])

    # The demo model exactly as the web page uses it.
    guard = FlyGuard.load(DEMO, enc)
    P_demo = {s: guard.lobe.transform(E[s]) for s in splits}
    table.add("데모", f"웹 데모 모델 ({demo_meta['wiring']})", [mb_scores(guard.mb, P_demo)])

    # Hybrids: a fine-tuned, cross-fitted nose read out linearly, by the mushroom body, or by its own head.
    hyb_vote = None                     # (name, fold scores per split, threshold) of the best hybrid's fly vote
    for hname in available():
        hg = HybridGuard(HYBRID_OUT / hname)
        k = hg.meta["k"]
        for readout, group, title, note in [
            ("head", "트랙 A", f"{hname} {k}-fold 앙상블 (분류 헤드 평균)", "교차 적합 폴드 인코더"),
            ("linear", "하이브리드", f"미세조정 {hname} 코 + 선형 판독", "폴드마다 인코더가 안 본 문장으로 학습"),
            ("fly", "하이브리드", f"미세조정 {hname} 코 + 버섯체", f"초파리 {k}마리 투표"),
        ]:
            folds = {s: hg.fold_scores(texts[s], readout, cache_split=s) for s in splits if s != "train"}
            table.add(group, title, [({s: f.mean(0) for s, f in folds.items()}, hg.meta["threshold"][readout])], note)
            if readout == "fly":
                hyb_vote = hyb_vote or (hname, folds, hg.meta["threshold"]["fly"])
                one = {s: f[0] for s, f in folds.items()}
                table.add("하이브리드", f"미세조정 {hname} 코 + 버섯체, 1마리", [(one, best_threshold(Y_VAL, one["validation"]))], "투표 효과 비교용")
        del hg
        torch.cuda.empty_cache()
    acc = {r["model"]: r[f"{split}_acc"] for r in table.rows}
    sections["2x2"] = "| 코(인코더) \\ 판독 | 선형 (로지스틱 회귀) | 버섯체 (도파민 규칙) |\n|---|---|---|\n" + \
        f"| 고정 MiniLM | {acc['MiniLM + 로지스틱 회귀 (선형 판독)']:.3f} | {acc[f'버섯체: {label} 배선']:.3f} |\n" + \
        "\n".join(f"| 미세조정 {h} (교차 적합) | {acc[f'미세조정 {h} 코 + 선형 판독']:.3f} | {acc[f'미세조정 {h} 코 + 버섯체']:.3f} |"
                  for h in available())

    # Curves on validation (random wiring, seeds 0-2).
    curves = []
    def curve(title, values, make_kw, train_idx_fn=None):
        accs = []
        for v in values:
            a = []
            for s in range(3):
                idx = train_idx_fn(v, s) if train_idx_fn else None
                kw = make_kw(v)
                w = Wiring.random(kw.pop("n_kc", 2000), 50, claws=7, seed=100 + s)
                mb, P = fit_mb(E, w, s, ytr, train_idx=idx, **kw)
                a.append(binary_metrics(Y_VAL, mb.score(P["validation"]), mb.theta)["acc"])
            accs.append(np.mean(a))
        curves.append(f"| {title} | " + " · ".join(f"{v}: {x:.3f}" for v, x in zip(values, accs)) + " |")
    curve("KC 수", [250, 1000, 2000, 8000, 20000], lambda v: {"n_kc": v, "homeostasis": False})
    curve("발화 비율", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5], lambda v: {"sparsity": v, "homeostasis": False})
    rng_idx = lambda n, s: np.random.default_rng(s).permutation(len(ytr))[:n]
    curve("학습 문장 수", [50, 200, 1000, 4000, len(ytr)], lambda v: {"homeostasis": False}, rng_idx)
    sections["curves"] = "| 변수 | validation 정확도 (무작위 배선, 시드 3개 평균) |\n|---|---|\n" + "\n".join(curves)

    if args.final:
        hyb = None if hyb_vote is None else (hyb_vote[0], hyb_vote[1]["test"].mean(0), hyb_vote[2])
        batch = batch_items("test", d["test"], guard.mb, P_demo["test"], name, comp_proba(texts["test"]), comp_threshold, hyb)
        (REPORTS / "batch_test.json").write_text(json.dumps(batch))

    report = f"""# FlyGuard 결과

생성: {time.strftime('%Y-%m-%d %H:%M')} · {'validation + test (--final)' if args.final else 'validation만'} · 임계값과 도파민 양(β)은 모두 validation에서 선택

## 데이터
```
{describe(d)}
```
{sections['shortcut']}

## 결과표
{table.markdown()}

- ± 는 시드 5개(안테나엽 채널 배정과 대조군 배선)의 표준편차. 2,101문장에서 정확도 0.93의 95% 신뢰구간은 약 ±1.1%p.
- 도파민 제거(학습 없음)는 모든 시냅스 가중치가 같아져 정확도 0.5(찍기)가 됩니다.

## 코 × 판독 ({split} 정확도)
{sections['2x2']}

- 미세조정 코는 교차 적합으로 만들었습니다. 폴드 k의 인코더는 나머지 폴드로 학습하고, 폴드 k의 버섯체와 선형 판독은 그 인코더가 본 적 없는 폴드 k 문장으로만 배웁니다.
- 고정 MiniLM 버섯체는 시드 5개 평균이고, 하이브리드는 폴드 회로들이 확률을 평균해 투표한 결과입니다.

## 실제 배선 vs 대조군 (같은 시드끼리 짝지어 비교)
{sections['wiring']}

안테나엽 채널 → 사구체 배정이 무작위라서, 실제 배선이 줄 수 있는 차이는 "누가 누구와 연결되는가"보다 차수 분포(KC당 입력 수, 사구체별 인기도)에서 나옵니다.

## 곡선
{sections['curves']}
"""
    (REPORTS / "results.md").write_text(report, encoding="utf-8")
    (REPORTS / "results.json").write_text(json.dumps(table.rows, indent=1))
    with open(REPORTS / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "final": args.final,
                            "acc": {r["model"]: round(r[f"{split}_acc"], 4) for r in table.rows}}, ensure_ascii=False) + "\n")
    print(f"wrote {REPORTS / 'results.md'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
