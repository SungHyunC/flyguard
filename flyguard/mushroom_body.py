"""The fly's olfactory learning circuit as a prompt classifier.

    text --encoder--> embedding --AntennalLobe--> glomerulus rates (projection neurons)
         --Wiring--> Kenyon-cell drive --homeostasis, APL inhibition (top 5%)--> sparse KC code
         --dopamine-shaped KC->MBON synapses--> MBON approach / avoid --> pass or quarantine

Learning follows the valence logic of the mushroom body (Aso et al. 2014): punishment dopamine
(PPL1) depresses the synapses from active KCs onto the approach MBON, reward dopamine (PAM)
depresses those onto the avoid MBON. An injection prompt is a punishment, a benign prompt a reward.
Each prompt of class c delivers a dopamine dose beta/N_c (N_c = prompts of that class), so after
training

    w_approach = exp(-beta * n1 / N1)        w_avoid = exp(-beta * n0 / N0)

where n_c counts how often each KC fired for class c. The weights depend only on these counts, so
online learning (`partial_fit`, for a "watch the fly learn" animation) and the closed form (`fit`)
end in the same state, in any order, and beta can be re-tuned without re-training.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .connectome import Wiring
from .metrics import best_threshold

BETAS = (5, 10, 20, 50, 100, 200, 500)


class AntennalLobe:
    """Embedding -> non-negative glomerulus rates.

    PCA keeps ceil(G/2) directions; each is split into a rectified ON and OFF channel, and the
    channels are dealt to the G glomeruli in a seeded random order. Which channel lands in which
    glomerulus has no meaning, so compare wirings over several seeds with the same seeds for each.
    """

    def __init__(self, n_glom: int = 50, seed: int = 0):
        self.n_glom, self.seed = n_glom, seed

    def fit(self, E: np.ndarray) -> "AntennalLobe":
        k = math.ceil(self.n_glom / 2)
        self.mean = E.mean(0)
        _, _, Vt = np.linalg.svd(E - self.mean, full_matrices=False)
        self.components = Vt[:k].astype(np.float32)
        self.channel = np.random.default_rng(self.seed).permutation(2 * k)[: self.n_glom]
        return self

    def transform(self, E: np.ndarray) -> np.ndarray:
        Z = (E - self.mean) @ self.components.T
        on_off = np.hstack([np.maximum(Z, 0), np.maximum(-Z, 0)])
        return on_off[:, self.channel].astype(np.float32)


class FlyMushroomBody:
    def __init__(self, wiring: Wiring, sparsity: float = 0.05, homeostasis: bool = True, apl: bool = True):
        self.wiring, self.sparsity, self.homeostasis, self.apl = wiring, sparsity, homeostasis, apl
        self.k = max(1, round(sparsity * wiring.n_kc))
        totals = np.asarray(wiring.W.sum(1)).ravel()
        self.silent = totals == 0                       # KCs without olfactory input never fire
        scale = (1 / np.where(self.silent, 1, totals)).astype(np.float32)
        self._Wn = (sparse.diags(scale) @ wiring.W).astype(np.float32).tocsr()   # each KC averages its inputs
        self.mu = self.sd = None
        self.n0 = self.n1 = None
        self.N0 = self.N1 = 0
        self.beta, self.theta, self.sigma, self.platt = 50.0, 0.0, 1.0, (1.0, 0.0)

    # sensing -----------------------------------------------------------------------------------
    def _raw_drive(self, P: np.ndarray) -> np.ndarray:
        return np.asarray((self._Wn @ P.T).T)

    def develop(self, P: np.ndarray) -> "FlyMushroomBody":
        """Homeostasis: learn each KC's usual drive from unlabelled prompts, so a KC with many or
        strong inputs does not win every time (that would make real wiring lose for a
        methodological rather than a biological reason)."""
        D = self._raw_drive(P)
        self.mu = D.mean(0)
        sd = D.std(0)
        self.sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
        return self

    def drive(self, P: np.ndarray) -> np.ndarray:
        D = self._raw_drive(P)
        if self.homeostasis:
            D = (D - self.mu) / self.sd
        D[:, self.silent] = -np.inf
        return D

    def kc_code(self, P: np.ndarray, batch: int = 4096) -> sparse.csr_matrix:
        """Binary KC activity. APL feedback inhibition lets only the k most driven KCs fire;
        with apl=False (lesion) every KC above its usual drive fires."""
        blocks = []
        for i in range(0, len(P), batch):
            D = self.drive(P[i:i + batch])
            n = len(D)
            if self.apl:
                idx = np.argpartition(-D, self.k - 1, axis=1)[:, : self.k]
                blocks.append(sparse.csr_matrix((np.ones(n * self.k, np.float32), idx.ravel(),
                                                 np.arange(0, n * self.k + 1, self.k)), shape=(n, self.wiring.n_kc)))
            else:
                thr = 0.0 if self.homeostasis else D.mean(1, keepdims=True)
                blocks.append(sparse.csr_matrix(D > thr, dtype=np.float32))
        return sparse.vstack(blocks).tocsr()

    # learning ----------------------------------------------------------------------------------
    def fit(self, P: np.ndarray, y) -> "FlyMushroomBody":
        if self.homeostasis and self.mu is None:
            self.develop(P)
        self.n0 = self.n1 = None
        self.N0 = self.N1 = 0
        return self.partial_fit(P, y)

    def partial_fit(self, P: np.ndarray, y) -> "FlyMushroomBody":
        """Online dopamine learning: count which KCs fired for each class."""
        if self.homeostasis and self.mu is None:
            raise RuntimeError("call develop() on unlabelled prompts before partial_fit()")
        if self.n0 is None:
            self.n0, self.n1 = np.zeros(self.wiring.n_kc), np.zeros(self.wiring.n_kc)
        H, y = self.kc_code(P), np.asarray(y)
        self.n1 += np.asarray(H[y == 1].sum(0)).ravel()
        self.n0 += np.asarray(H[y == 0].sum(0)).ravel()
        self.N1 += int((y == 1).sum())
        self.N0 += int((y == 0).sum())
        return self

    def weights(self, beta: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(w_approach, w_avoid): KC->MBON synapse strengths in [0, 1]."""
        b = self.beta if beta is None else beta
        return np.exp(-b * self.n1 / max(self.N1, 1)), np.exp(-b * self.n0 / max(self.N0, 1))

    # output ------------------------------------------------------------------------------------
    def mbon(self, H: sparse.csr_matrix, beta: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(approach, avoid) MBON activity = mean synapse strength over the active KCs."""
        w_app, w_av = self.weights(beta)
        n_active = np.maximum(np.diff(H.indptr), 1)
        return H @ w_app / n_active, H @ w_av / n_active

    def _score(self, H: sparse.csr_matrix, beta: float | None = None) -> np.ndarray:
        approach, avoid = self.mbon(H, beta)
        return avoid - approach

    def score(self, P: np.ndarray) -> np.ndarray:
        """avoid - approach; above theta means quarantine (injection)."""
        return self._score(self.kc_code(P))

    def calibrate(self, P_val: np.ndarray, y_val, betas=BETAS) -> "FlyMushroomBody":
        """Choose the dopamine dose beta (validation AUC), the threshold theta (validation accuracy)
        and a Platt mapping from margin to probability. Uses validation only."""
        H, y = self.kc_code(P_val), np.asarray(y_val)
        self.beta = float(max(betas, key=lambda b: roc_auc_score(y, self._score(H, b))))
        s = self._score(H)
        self.theta = best_threshold(y, s)
        self.sigma = float(s.std()) or 1.0
        lr = LogisticRegression().fit(((s - self.theta) / self.sigma)[:, None], y)
        self.platt = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    def _prob(self, margin):
        a, b = self.platt
        return 1 / (1 + np.exp(-(a * margin + b)))

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        return self._prob((self.score(P) - self.theta) / self.sigma)

    def predict(self, P: np.ndarray) -> np.ndarray:
        return (self.score(P) > self.theta).astype(int)

    def trace(self, p: np.ndarray) -> dict:
        """Everything the web demo needs to animate one prompt (about 2 KB)."""
        active = self.kc_code(p[None, :]).indices
        w_app, w_av = self.weights()
        approach, avoid = float(w_app[active].mean()), float(w_av[active].mean())
        score = avoid - approach
        margin = (score - self.theta) / self.sigma
        freq = (self.n0 + self.n1) / max(self.N0 + self.N1, 1)
        return {
            "pn": np.round(p, 3).tolist(),
            "kc_active": active.tolist(),
            "kc_vote": np.round(w_av[active] - w_app[active], 3).tolist(),   # > 0 votes quarantine
            "approach": round(approach, 4),
            "avoid": round(avoid, 4),
            "theta": round(self.theta, 4),
            "score": round(score, 4),
            "margin": round(float(margin), 3),
            "p": round(float(self._prob(margin)), 4),
            "decision": "injection" if score > self.theta else "benign",
            # share of active KCs that almost never fired in training: an unfamiliar smell
            "novelty": round(float(np.mean(freq[active] < self.sparsity / 10)), 3),
        }


class FlyGuard:
    """Encoder + antennal lobe + mushroom body: text in, decision and circuit trace out."""

    def __init__(self, encoder, lobe: AntennalLobe, mb: FlyMushroomBody):
        self.encoder, self.lobe, self.mb = encoder, lobe, mb

    def rates(self, texts) -> np.ndarray:
        return self.lobe.transform(self.encoder.encode(list(texts)))

    def predict_proba(self, texts) -> np.ndarray:
        return self.mb.predict_proba(self.rates(texts))

    def classify(self, text: str) -> dict:
        return self.mb.trace(self.rates([text])[0])

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        mb, lobe = self.mb, self.lobe
        np.savez_compressed(path / "lobe.npz", mean=lobe.mean, components=lobe.components, channel=lobe.channel)
        mb.wiring.save(path / "wiring.npz")
        empty = np.zeros(0, np.float32)
        np.savez_compressed(path / "mushroom_body.npz", n0=mb.n0, n1=mb.n1,
                            mu=empty if mb.mu is None else mb.mu, sd=empty if mb.sd is None else mb.sd)
        meta = {
            "encoder": self.encoder.name, "n_glom": lobe.n_glom, "lobe_seed": lobe.seed,
            "wiring": mb.wiring.source, "n_kc": mb.wiring.n_kc, "sparsity": mb.sparsity,
            "homeostasis": mb.homeostasis, "apl": mb.apl, "N0": mb.N0, "N1": mb.N1,
            "beta": mb.beta, "theta": mb.theta, "sigma": mb.sigma, "platt": list(mb.platt),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, encoder) -> "FlyGuard":
        meta = json.loads((path / "meta.json").read_text())
        z = np.load(path / "lobe.npz")
        lobe = AntennalLobe(meta["n_glom"], meta["lobe_seed"])
        lobe.mean, lobe.components, lobe.channel = z["mean"], z["components"], z["channel"]
        mb = FlyMushroomBody(Wiring.load(path / "wiring.npz"), meta["sparsity"], meta["homeostasis"], meta["apl"])
        z = np.load(path / "mushroom_body.npz")
        mb.n0, mb.n1 = z["n0"], z["n1"]
        if meta["homeostasis"]:
            mb.mu, mb.sd = z["mu"], z["sd"]
        mb.N0, mb.N1 = meta["N0"], meta["N1"]
        mb.beta, mb.theta, mb.sigma, mb.platt = meta["beta"], meta["theta"], meta["sigma"], tuple(meta["platt"])
        return cls(encoder, lobe, mb)
