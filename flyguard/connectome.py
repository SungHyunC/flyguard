"""Projection-neuron -> Kenyon-cell wiring (the mushroom body calyx).

A Wiring is a non-negative (n_kc, n_glom) matrix: row i says how strongly Kenyon cell i samples each
antennal-lobe glomerulus. Sources:
  - Wiring.random     each KC samples `claws` glomeruli uniformly (PN->KC wiring is close to random,
                      Caron et al. 2013)
  - wiring.shuffled() degree-preserving edge swaps; the control for the real connectome
  - Wiring.load(...)  the real right-hemisphere FlyWire wiring built by `python -m flyguard.connectome`

    python -m flyguard.connectome          # download FlyWire files (~130 MB, once) and build
                                           # artifacts/flywire/wiring.npz + layout.json

Data (public, no login): FlyWire v783 connectivity as shipped with Shiu et al. 2024
(github.com/philshiu/Drosophila_brain_model) and the FlyWire cell-type annotations of Schlegel et al.
2024 (github.com/flyconnectome/flywire_annotations).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .paths import ARTIFACTS, CONNECTOME

FILES = {
    "Connectivity_783.parquet": "https://github.com/philshiu/Drosophila_brain_model/raw/main/Connectivity_783.parquet",
    "Supplemental_file1_neuron_annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}
FLYWIRE = ARTIFACTS / "flywire"
VOXEL_UM = np.array([0.004, 0.004, 0.04])     # FlyWire voxel size (4 x 4 x 40 nm) in micrometres
# MBONs whose activation drives approach vs avoidance (Aso et al. 2014b): punishment depresses
# KC synapses onto MBON-γ1pedc>α/β (MBON11); reward depresses those onto MBON-γ5β'2a (MBON01)
# and MBON-β'2mp (MBON03, MBON04).
APPROACH_MBONS = ["MBON11"]
AVOID_MBONS = ["MBON01", "MBON03", "MBON04"]


@dataclass
class Wiring:
    W: sparse.csr_matrix                  # (n_kc, n_glom), non-negative
    glom_names: list[str]
    source: str
    kc_types: list[str] = field(default_factory=list)

    @property
    def n_kc(self) -> int:
        return self.W.shape[0]

    @property
    def n_glom(self) -> int:
        return self.W.shape[1]

    @classmethod
    def random(cls, n_kc: int = 2000, n_glom: int = 50, claws: int = 7, seed: int = 0) -> "Wiring":
        rng = np.random.default_rng(seed)
        cols = np.concatenate([rng.choice(n_glom, claws, replace=False) for _ in range(n_kc)])
        rows = np.repeat(np.arange(n_kc), claws)
        W = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_kc, n_glom))
        return cls(W, [f"g{i:02d}" for i in range(n_glom)], f"random(claws={claws}, seed={seed})")

    def shuffled(self, seed: int = 0, swaps_per_edge: int = 10) -> "Wiring":
        """Swap the glomerulus ends of two random edges (a-b, c-d -> a-d, c-b) whenever that creates
        no duplicate edge. Each KC keeps its number of inputs and their synapse weights; each
        glomerulus keeps its number of KC partners. Only who-connects-to-whom is randomised."""
        coo = self.W.tocoo()
        rows, cols, vals = coo.row.tolist(), coo.col.tolist(), coo.data.copy()
        edges = set(zip(rows, cols))
        rng = np.random.default_rng(seed)
        for i, j in rng.integers(0, len(rows), size=(swaps_per_edge * len(rows), 2)).tolist():
            a, b, c, d = rows[i], cols[i], rows[j], cols[j]
            if a == c or b == d or (a, d) in edges or (c, b) in edges:
                continue
            edges -= {(a, b), (c, d)}
            edges |= {(a, d), (c, b)}
            cols[i], cols[j] = d, b
        W = sparse.csr_matrix((vals, (rows, cols)), shape=self.W.shape)
        return Wiring(W, self.glom_names, f"shuffled({self.source}, seed={seed})", self.kc_types)

    def transformed(self, weight: str = "count") -> "Wiring":
        """Synapse-count transform: 'count' (as is), 'log1p', or 'binary'."""
        W = self.W.copy().astype(np.float32)
        if weight == "log1p":
            W.data = np.log1p(W.data)
        elif weight == "binary":
            W.data = np.ones_like(W.data)
        elif weight != "count":
            raise ValueError(weight)
        return Wiring(W, self.glom_names, f"{self.source} [{weight}]", self.kc_types)

    def stats(self) -> dict:
        inputs = np.diff(self.W.indptr)
        partners = np.bincount(self.W.indices, minlength=self.n_glom)
        return {
            "n_kc": self.n_kc, "n_glom": self.n_glom, "edges": int(self.W.nnz),
            "glomeruli_per_kc": {"mean": round(float(inputs.mean()), 2), "min": int(inputs.min()), "max": int(inputs.max())},
            "kcs_per_glomerulus": {"mean": round(float(partners.mean()), 1), "min": int(partners.min()), "max": int(partners.max())},
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(path, data=self.W.data, indices=self.W.indices, indptr=self.W.indptr,
                            shape=np.array(self.W.shape), glom_names=np.array(self.glom_names),
                            source=np.array(self.source), kc_types=np.array(self.kc_types))

    @classmethod
    def load(cls, path: Path) -> "Wiring":
        z = np.load(path, allow_pickle=False)
        W = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return cls(W, z["glom_names"].tolist(), str(z["source"]), z["kc_types"].tolist())


# --- building the real wiring from FlyWire --------------------------------------------------------

def download() -> None:
    CONNECTOME.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = CONNECTOME / name
        if not target.exists():
            print(f"downloading {name} ...", flush=True)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)


def _xyz(df: pd.DataFrame) -> np.ndarray:
    """Soma position (or a point on the neuron when the soma is unknown) in micrometres, with
    y flipped so dorsal is up and z flipped so the default view looks at the brain from the front."""
    soma = df[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=float)
    pos = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    p = np.where(np.isnan(soma).any(axis=1, keepdims=True) | (soma == 0).all(axis=1, keepdims=True), pos, soma)
    return p * VOXEL_UM * np.array([1, -1, -1])


def build(side: str = "right", min_syn: int = 5, n_dust: int = 20000, seed: int = 0) -> Wiring:
    """Projection neuron -> Kenyon cell wiring of one hemisphere, summed per glomerulus."""
    ann = pd.read_csv(CONNECTOME / "Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z",
                               "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side"])
    here = ann[ann.side == side]
    pns = here[(here.cell_class == "ALPN") & (here.cell_sub_class == "uniglomerular")].copy()
    pns["glom"] = pns.hemibrain_type.str.split("_").str[0]
    kcs = here[here.cell_class == "Kenyon_Cell"]

    conn = pd.read_parquet(CONNECTOME / "Connectivity_783.parquet",
                           columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    e = conn[conn.Presynaptic_ID.isin(pns.root_id) & conn.Postsynaptic_ID.isin(kcs.root_id)
             & (conn.Connectivity >= min_syn)]
    e = e.merge(pns[["root_id", "glom"]], left_on="Presynaptic_ID", right_on="root_id")
    per_glom = e.groupby(["Postsynaptic_ID", "glom"]).Connectivity.sum().reset_index()

    # Kenyon cells with olfactory input, grouped by type (γ, αβ, α'β') for a readable layout.
    kc_info = kcs.set_index("root_id").loc[per_glom.Postsynaptic_ID.unique()]
    kc_info = kc_info.assign(t=kc_info.cell_type.fillna(kc_info.hemibrain_type).fillna("KC")).sort_values(["t"])
    kc_index = {rid: i for i, rid in enumerate(kc_info.index)}
    glom_names = sorted(per_glom.glom.unique())
    glom_index = {g: j for j, g in enumerate(glom_names)}
    W = sparse.csr_matrix((per_glom.Connectivity.to_numpy(np.float32),
                           (per_glom.Postsynaptic_ID.map(kc_index), per_glom.glom.map(glom_index))),
                          shape=(len(kc_index), len(glom_names)))
    wiring = Wiring(W, glom_names, f"flywire_783 ({side}, >={min_syn} synapses)", kc_info.t.tolist())

    rng = np.random.default_rng(seed)
    group_mean = lambda df: _xyz(df).mean(0).round(1).tolist()
    mbons = here[here.cell_class == "MBON"]
    approach = mbons[mbons.cell_type.isin(APPROACH_MBONS)]
    avoid = mbons[mbons.cell_type.isin(AVOID_MBONS)]
    dans = here[here.cell_class == "DAN"]
    dust = ann.iloc[rng.permutation(len(ann))[:n_dust]]
    layout = {
        "units": "µm",
        "dust": _xyz(dust).round(1).tolist(),
        "kc": _xyz(kc_info.reset_index()).round(1).tolist(),
        "glom": [group_mean(pns[pns.glom == g]) for g in glom_names],
        "mbon": {"approach": group_mean(approach if len(approach) else mbons),
                 "avoid": group_mean(avoid if len(avoid) else mbons)},
        "apl": group_mean(here[here.hemibrain_type == "APL"]),
        "extra": {
            "MBON": _xyz(mbons).round(1).tolist(),
            "PAM": _xyz(dans[dans.cell_type.fillna("").str.startswith("PAM")]).round(1).tolist(),
            "PPL1": _xyz(dans[dans.cell_type.fillna("").str.startswith("PPL1")]).round(1).tolist(),
        },
    }
    FLYWIRE.mkdir(parents=True, exist_ok=True)
    wiring.save(FLYWIRE / "wiring.npz")
    (FLYWIRE / "layout.json").write_text(json.dumps(layout))

    synapses = np.asarray(W.sum(1)).ravel()
    print(f"{side} hemisphere: {len(pns)} uniglomerular PNs in {len(glom_names)} glomeruli, "
          f"{len(kcs)} KCs ({len(kc_index)} with PN input >= {min_syn} synapses)")
    print(f"PN synapses per KC: median {np.median(synapses):.0f}; types: {kc_info.t.value_counts().to_dict()}")
    print(json.dumps(wiring.stats()))
    return wiring


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--min-syn", type=int, default=5)
    args = ap.parse_args()
    download()
    build(args.side, args.min_syn)
