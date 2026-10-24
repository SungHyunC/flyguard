"""3-D layout of the circuit for the web brain panel.

Real FlyWire wiring comes with real neuron positions (written by `python -m flyguard.connectome`);
random wiring gets a schematic mushroom-body layout so the page works before any download.
Axes: x = lateral (right hemisphere positive), y = dorsal, z = anterior. FlyWire coordinates are in
micrometres, schematic ones in arbitrary units; the page rescales either.
"""
from __future__ import annotations

import json

import numpy as np

from .connectome import Wiring
from .paths import ARTIFACTS

FLYWIRE_LAYOUT = ARTIFACTS / "flywire" / "layout.json"


def _round(a) -> list:
    return np.round(np.asarray(a, dtype=float), 2).tolist()


def _directions(n: int, rng) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def schematic(n_kc: int, n_glom: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_dust = 6000
    side = np.where(rng.random(n_dust) < 0.5, -1.0, 1.0)
    shell = _directions(n_dust, rng) * [1.5, 1.0, 0.75] * (0.96 + 0.08 * rng.random((n_dust, 1)))
    dust = shell + np.c_[side * 1.3, np.zeros(n_dust), np.zeros(n_dust)]
    golden = np.pi * (3 - np.sqrt(5))
    i = np.arange(n_glom) + 0.5
    y = 1 - 2 * i / n_glom
    r = np.sqrt(1 - y ** 2)
    glom = np.array([0.55, -0.55, 0.55]) + 0.28 * np.c_[r * np.cos(golden * i), y, r * np.sin(golden * i)]
    cap = _directions(n_kc, rng)
    cap[:, 1] = np.abs(cap[:, 1])
    calyx = np.array([1.45, 0.55, -0.45])
    kc = calyx + cap * (0.3 + 0.1 * rng.random((n_kc, 1)))
    heel, vertical, medial = [0.95, -0.05, 0.25], [0.95, 0.8, 0.45], [0.25, -0.05, 0.5]
    return {
        "units": "schematic",
        "dust": _round(dust), "kc": _round(kc), "glom": _round(glom),
        "mbon": {"approach": heel, "avoid": medial},
        "apl": [1.15, 0.35, 0.0],
        "outline": [[calyx.tolist(), heel], [heel, vertical], [heel, medial]],
    }


def layout(wiring: Wiring) -> dict:
    if wiring.source.startswith("flywire") and FLYWIRE_LAYOUT.exists():
        lay = json.loads(FLYWIRE_LAYOUT.read_text())
    else:
        lay = schematic(wiring.n_kc, wiring.n_glom)
    coo = wiring.W.tocoo()
    lay.update(source=wiring.source, glom_names=wiring.glom_names, kc_types=wiring.kc_types,
               edges=[coo.row.tolist(), coo.col.tolist()])
    return lay
