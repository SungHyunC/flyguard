"""Figures for docs/analysis.md, drawn from the saved results of the analysis scripts.

    python -m flyguard.figures        # writes docs/figures/*.png; skips a figure whose results are missing

Style: one validated categorical order (blue, orange, aqua; at most three series where marks overlap),
2px lines, 8px markers with a surface ring, hairline solid grids, text in ink colours only. Every number
drawn here is also in a reports/*.md table.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .paths import REPORTS, ROOT  # noqa: E402

FIG = ROOT / "docs" / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_DARK, BLUE_LIGHT, GRAY = "#184f95", "#86b6ef", "#b5b3ab"
SHORT = {
    "TF-IDF + 로지스틱 회귀": "TF-IDF", "MiniLM 고정 + 로지스틱 회귀": "MiniLM + 선형", "초파리 · 고정 MiniLM 코": "초파리 (고정 MiniLM)",
    "ModernBERT 미세조정 1개": "ModernBERT 미세조정", "하이브리드 ModernBERT · 초파리 투표": "하이브리드 MB · 초파리",
    "하이브리드 ModernBERT · 헤드 앙상블": "하이브리드 MB · 헤드", "하이브리드 MiniLM · 초파리 투표": "하이브리드 MiniLM · 초파리",
    "하이브리드 MiniLM · 헤드 앙상블": "하이브리드 MiniLM · 헤드", "ProtectAI DeBERTa v2 (외부 전문 탐지기)": "ProtectAI (외부)",
    "Qwen2.5-1.5B 제로샷 (거대언어모델)": "Qwen2.5-1.5B (LLM)", "Qwen3-4B 제로샷 (거대언어모델)": "Qwen3-4B (LLM)",
}


def style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.markersize": 8, "lines.markeredgewidth": 2, "lines.markeredgecolor": SURFACE,
    })


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def plain_log_ticks(ax, axis="x"):
    """Log axis ticks as plain numbers (1, 10, 100 or 16, 64, 256), not powers."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    getattr(ax, f"{axis}axis").set_major_formatter(fmt)
    getattr(ax, f"{axis}axis").set_minor_formatter(matplotlib.ticker.NullFormatter())


def place_labels(ax, xs, ys, texts, fontsize=8.5, obstacles=None):
    """Direct labels next to dots: try right, above, below and left of each dot, keep the first spot that
    overlaps no other label and no dot. Falls back to the right side."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    extra = ax.transData.transform(np.asarray(obstacles)) if obstacles is not None and len(obstacles) else np.zeros((0, 2))
    placed = []
    candidates = [(7, 0, "left", "center"), (7, 9, "left", "bottom"), (7, -9, "left", "top"), (0, 10, "center", "bottom"),
                  (0, -10, "center", "top"), (-7, 0, "right", "center"), (-7, 9, "right", "bottom"), (-7, -9, "right", "top")]
    for i in np.argsort(-np.asarray(ys)):
        best = None
        for dx, dy, ha, va in candidates:
            t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=INK2, ha=ha, va=va, bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none", alpha=0.85))
            bb = t.get_window_extent(renderer).expanded(1.04, 1.15)
            clash = any(bb.overlaps(p) for p in placed) or any(
                bb.contains(px, py) for j, (px, py) in enumerate(pts) if j != i) or any(bb.contains(px, py) for px, py in extra)
            if not clash:
                best = (t, bb)
                break
            t.remove()
        if best is None:
            t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=(7, 0), textcoords="offset points", fontsize=fontsize,
                            color=INK2, ha="left", va="center")
            best = (t, t.get_window_extent(renderer))
        placed.append(best[1])


# 1. generalization ------------------------------------------------------------------------------
def model_summary() -> pd.DataFrame:
    """Per model: S-Labs test BA, external mean BA (suite macro), latency and family."""
    from .generalize_report import load, metrics
    rows, meta, scores = load()
    y, suite = rows.label.to_numpy(), rows.suite.to_numpy()
    ext = [s for s in pd.unique(suite) if not s.startswith("slabs")]
    rec = []
    for m in meta:
        t = meta[m]["threshold"]
        ba = [metrics(y[suite == s], scores[m][suite == s], t).get("ba") for s in ext]
        test = metrics(y[suite == "slabs_test"], scores[m][suite == "slabs_test"], t)["ba"]
        rec.append({"model": m, "test": test, "external": float(np.nanmean([b for b in ba if b is not None])),
                    "latency": meta[m]["latency_ms"], "family": meta[m]["family"]})
    return pd.DataFrame(rec)


def generalization():
    df = model_summary()
    df["test_rank"] = df.test.rank(ascending=False, method="min").astype(int)
    df["ext_rank"] = df.external.rank(ascending=False, method="min").astype(int)
    df = df.sort_values("external")

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(df) + 1.2))
    yy = np.arange(len(df))
    ax.hlines(yy, df.external * 100, df.test * 100, color=GRAY, linewidth=2, zorder=1)
    ax.plot(df.test * 100, yy, "o", color=BLUE_DARK, label="S-Labs test (학습 분포)", zorder=3)
    ax.plot(df.external * 100, yy, "o", color=BLUE_LIGHT, label="외부 17개 묶음 평균", zorder=3)
    ax.set_yticks(yy, [f"{SHORT.get(m, m)}   ({a}위 → {b}위)" for m, a, b in zip(df.model, df.test_rank, df.ext_rank)])
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("균형 정확도 (%)  ·  괄호: S-Labs 순위 → 외부 순위")
    ax.set_xlim(50, 100)
    ax.set_title("S-Labs에서 잘하는 순서와 밖에서 버티는 순서는 다르다", loc="left")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig1_generalization_rank_flip.png")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    own = ~df.family.isin(["외부"])
    for mask, color, name in ((own, BLUE, "S-Labs로 학습한 모델"), (~own, ORANGE, "S-Labs를 안 본 모델 (외부 탐지기·LLM)")):
        ax.plot(df.latency[mask], df.external[mask] * 100, "o", color=color, label=name, linestyle="none")
    ax.set_xscale("log")
    ax.set_xlabel("한 문장 처리 시간 (ms, 로그 눈금)")
    ax.set_ylabel("외부 평균 균형 정확도 (%)")
    ax.set_title("느리고 큰 모델이 밖에서 더 버티는 것은 아니다", loc="left")
    lo, hi = df.external.min() * 100, df.external.max() * 100
    ax.set_ylim(lo - 2, hi + 2.5)
    ax.set_xlim(df.latency.min() / 1.6, df.latency.max() * 4)
    plain_log_ticks(ax)
    ax.legend(loc="lower right", fontsize=9)
    place_labels(ax, df.latency.to_numpy(), (df.external * 100).to_numpy(), [SHORT.get(m, m) for m in df.model])
    save(fig, "fig2_latency_vs_robustness.png")


# 2. tuning --------------------------------------------------------------------------------------
METHOD_SHORT = {"frozen_lp": "고정+선형", "frozen_fly": "고정+초파리", "bitfit": "BitFit", "lora": "LoRA",
                "top2": "상위 2층", "full": "전체", "lpft": "LP-FT"}
METHOD_KIND = {"frozen_lp": 0, "frozen_fly": 0, "bitfit": 1, "lora": 1, "top2": 1, "full": 2, "lpft": 2}


def tuning():
    path = REPORTS / "tuning" / "results.jsonl"
    if not path.exists():
        return
    from .tuning_report import best_epochs
    df = pd.read_json(path, lines=True)
    best = best_epochs(df)
    if "seed" in df:
        df = df.assign(seed=df.seed.fillna(0))
        b = df.loc[df.groupby(["encoder", "method", "seed"]).validation_acc.idxmax()]
        sd = b.groupby(["encoder", "method"])[["test_acc", "ood_mean_ba"]].std().fillna(0)
    encoders = list(dict.fromkeys(df.encoder))
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.6 * len(encoders), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    kinds = [(GRAY, "고정 (인코더 학습 없음)"), (BLUE, "일부만 학습 (BitFit · LoRA · 상위 2층)"), (ORANGE, "전체 학습 (전체 · LP-FT)")]
    for ax, enc in zip(axes, encoders):
        g = best[best.encoder == enc].set_index("method")
        for k, (color, name) in enumerate(kinds):
            ms = [m for m in g.index if METHOD_KIND[m] == k]
            if not ms:
                continue
            x, yv = g.loc[ms, "test_acc"] * 100, g.loc[ms, "ood_mean_ba"] * 100
            ax.plot(x, yv, "o", color=color, label=name, linestyle="none", zorder=3)
            for m in ms:
                if (enc, m) in sd.index and sd.loc[(enc, m), "ood_mean_ba"] > 0:
                    ax.errorbar(g.loc[m, "test_acc"] * 100, g.loc[m, "ood_mean_ba"] * 100, xerr=sd.loc[(enc, m), "test_acc"] * 100,
                                yerr=sd.loc[(enc, m), "ood_mean_ba"] * 100, color=color, linewidth=1, capsize=0, zorder=2)
        ax.set_title(enc, loc="left")
        ax.margins(x=0.18, y=0.12)
        place_labels(ax, (g.test_acc * 100).to_numpy(), (g.ood_mean_ba * 100).to_numpy(), [METHOD_SHORT[m] for m in g.index])
        ax.set_xlabel("S-Labs test 정확도 (%)")
    axes[0].set_ylabel("외부 패널 평균 균형 정확도 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("튜닝은 안의 점수를 올리지만, 밖의 점수는 튜닝 방법보다 인코더가 더 크게 정한다", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_tuning_tradeoff.png")

    probes = REPORTS / "tuning" / "probes.jsonl"
    if probes.exists():
        pr = pd.read_json(probes, lines=True).drop_duplicates(["encoder", "layer"], keep="last")
        encs = list(dict.fromkeys(pr.encoder))
        fig, axes = plt.subplots(2, len(encs), figsize=(4.4 * len(encs), 5.6), sharex="col", squeeze=False)
        for j, enc in enumerate(encs):
            g = pr[pr.encoder == enc].sort_values("layer")
            mk = dict(markersize=6) if len(g) <= 13 else dict(markersize=3.5, markeredgewidth=0)
            ax = axes[0, j]
            ax.plot(g.layer, g.pretrained_probe_acc * 100, "-o", color=GRAY, label="사전학습 그대로", **mk)
            ax.plot(g.layer, g.finetuned_probe_acc * 100, "-o", color=BLUE, label="전체 미세조정 후", **mk)
            ax.set_title(enc, loc="left")
            ax.set_ylim(max(80, min(g.pretrained_probe_acc.min(), g.finetuned_probe_acc.min()) * 100 - 2), 100.5)
            axes[1, j].plot(g.layer, g.cka, "-o", color=BLUE, **mk)
            axes[1, j].set_ylim(0, 1.05)
            axes[1, j].set_xlabel("층 (0 = 단어 임베딩)")
        axes[0, 0].set_ylabel("층별 탐침 정확도 (%)")
        axes[1, 0].set_ylabel("CKA: 미세조정 전후 닮은 정도")
        axes[0, 0].legend(loc="lower right", fontsize=9)
        fig.suptitle("인젝션 정보는 처음부터 아래층에 있고, 미세조정은 위층만 바꾼다", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        save(fig, "fig4_layer_probes.png")


# 3. shortcuts -----------------------------------------------------------------------------------
def shortcuts():
    path = REPORTS / "shortcuts.json"
    if not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    chars = res.get("characters")
    if chars:
        models = list(chars)
        variants = list(chars[models[0]])
        M = np.array([[chars[m][v] * 100 for v in variants] for m in models])
        fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(models) + 1.8))
        ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#2a78d6", "#0d366b"]),
                  vmin=0, vmax=100, aspect="auto")
        short = {"원문": "원문\n(대조)", "프랑스어": "+ 프랑스어\n(ASCII)", "한국어": "+ '감사합니다.'\n(한국어)", "이모지": "+ 이모지",
                 "투명문자": "+ 투명 문자\n(U+200B)", "키릴": "o·e·a를\n키릴 글자로"}
        ax.set_xticks(range(len(variants)), [next((lab for key, lab in short.items() if key in v), v) for v in variants],
                      fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(models)), [SHORT.get(m, m) for m in models], color=INK2)
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(len(models)):
            for j in range(len(variants)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9, color="#ffffff" if M[i, j] > 55 else INK)
        ax.set_title("정상 영어 문장에 무해한 글자만 붙여도 막아 버린 비율 (%)", loc="left")
        save(fig, "fig5_character_shortcut.png")


# 4. the fly ---------------------------------------------------------------------------------------
def fly():
    path = REPORTS / "fly_analysis.json"
    if not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    colors = {"fly": ORANGE, "lr_glom": BLUE, "lr_embed": AQUA, "lr_e5": AQUA}
    names = {"fly": "초파리 버섯체 (도파민 계수)", "lr_glom": "로지스틱 회귀 · 같은 55개 사구체", "lr_embed": "로지스틱 회귀 · 384차원 전체",
             "lr_e5": "로지스틱 회귀 · 768차원 전체"}
    if "few_labels" in res:
        few = pd.DataFrame(res["few_labels"])
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for m in ("fly", "lr_glom", "lr_embed"):
            g = few[few.model == m].groupby("n").auc.agg(["mean", "std"]).fillna(0)
            ax.fill_between(g.index, (g["mean"] - g["std"]) * 100, (g["mean"] + g["std"]) * 100, color=colors[m], alpha=0.1, linewidth=0)
            ax.plot(g.index, g["mean"] * 100, "-o", color=colors[m], label=names[m], markersize=6)
        ax.set_xscale("log", base=2)
        plain_log_ticks(ax)
        ax.set_xlabel("라벨 붙은 학습 문장 수 (로그 눈금)")
        ax.set_ylabel("S-Labs test AUC (%)")
        ax.set_title("라벨이 적을 때도 초파리 학습 규칙이 더 빠르지는 않다", loc="left")
        ax.legend(loc="lower right", fontsize=9)
        save(fig, "fig6_few_labels.png")
    if "familiarity" in res and res["familiarity"].get("deploy"):
        dep = pd.DataFrame(res["familiarity"]["deploy"])
        dep = dep[dep.expert.str.startswith("Qwen3")]
        singles = model_summary()
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        ax.plot(singles.test * 100, singles.external * 100, "o", color=GRAY, linestyle="none", label="모델 하나만", zorder=2)
        for (base, g), color in zip(dep.groupby("base", sort=False), (BLUE, ORANGE, AQUA)):
            g = g.sort_values("budget")
            ax.plot(g.test_acc * 100, g.external_ba * 100, "-o", color=color, markersize=5, zorder=3,
                    label=f"{SHORT.get(base, base)} → Qwen3-4B")
            ten = g[g.budget == 0.1]
            ax.plot(ten.test_acc * 100, ten.external_ba * 100, "o", color=color, markersize=11, zorder=4)
        ax.set_xlabel("S-Labs test 균형 정확도 (%)  ·  안에서의 성능")
        ax.set_ylabel("외부 17개 묶음 평균 균형 정확도 (%)  ·  밖에서의 성능")
        ax.set_title("안팎 성능 지도: 미세조정 모델 + 낯선 문장만 LLM에 넘기기", loc="left")
        ax.margins(x=0.08, y=0.1)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
        fig.text(0.01, -0.13, "큰 점: S-Labs validation 문장의 10%만 넘기도록 기준선을 정했을 때. 선 위의 점은 0·10·20·30·50·100%.",
                 fontsize=8, color=MUTED)
        place_labels(ax, (singles.test * 100).to_numpy(), (singles.external * 100).to_numpy(),
                     [SHORT.get(m, m) for m in singles.model], fontsize=7.5,
                     obstacles=np.column_stack([dep.test_acc * 100, dep.external_ba * 100]))
        save(fig, "fig7_inside_outside_map.png")
    if "online_korean" in res:
        onl = pd.DataFrame(res["online_korean"])
        ks = sorted(onl.k.unique())
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True)
        for col, w in enumerate((1, 10)):
            for row, (metric, ylab) in enumerate((("kg_ba", "보지 않은 한국어 문항 BA (%)"), ("test_acc", "S-Labs test 정확도 (%)"))):
                ax = axes[row, col]
                for m in ("fly", "lr_glom", "lr_e5"):
                    g = onl[(onl.model == m) & ((onl.weight == w) | (onl.k == 0))].groupby("k")[metric].mean()
                    ax.plot(range(len(ks)), [g.get(k, np.nan) * 100 for k in ks], "-o", color=colors[m] if m != "lr_e5" else AQUA,
                            label=names[m], markersize=6)
                ax.set_xticks(range(len(ks)), [str(k) for k in ks])
                if col == 0:
                    ax.set_ylabel(ylab)
                if row == 0:
                    ax.set_title(f"새 문장 가중치 {w}", loc="left")
                else:
                    ax.set_xlabel("새로 본 한국어 라벨 수 k")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle("한국어 예시를 몇 개 보여 주면: 배우는 속도(위)와 잊어버림(아래)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        save(fig, "fig8_online_korean.png")


def main():
    style()
    generalization()
    tuning()
    shortcuts()
    fly()


if __name__ == "__main__":
    main()
