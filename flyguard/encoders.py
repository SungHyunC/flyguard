"""Text encoders: the fly's olfactory receptor neurons.

Every encoder maps a list of texts to an (n, d) float32 array. Dataset embeddings are cached in
artifacts/cache/, keyed by encoder name and a hash of the texts, so experiments stay fast and the
server and the reports see exactly the same numbers.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .paths import ARTIFACTS, CACHE, ensure_dirs

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class MiniLMEncoder:
    """Frozen sentence embedding (384-d, L2-normalised)."""
    name = "minilm"

    def __init__(self, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MINILM, device=device or _device())

    def encode(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=256, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)


class CharNgramEncoder:
    """No-transformer nose: character 3-5 gram TF-IDF compressed by SVD. Fit on train only."""
    name = "charngram"

    def __init__(self, dim: int = 50):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        self.svd = TruncatedSVD(dim, random_state=0)

    def fit(self, texts) -> "CharNgramEncoder":
        self.svd.fit(self.vec.fit_transform(list(texts)))
        return self

    def encode(self, texts) -> np.ndarray:
        Z = self.svd.transform(self.vec.transform(list(texts)))
        return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(np.float32)


class FineTunedEncoder:
    """Track A model from artifacts/encoder: its own P(injection), and a mean-pooled embedding."""
    name = "finetuned"

    def __init__(self, path=ARTIFACTS / "encoder", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.meta = json.loads((path / "meta.json").read_text())
        self.device = torch.device(device or _device())
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.threshold = self.meta["threshold"]
        self.label = self.meta["base_model"].split("/")[-1]

    def predict_proba(self, texts) -> np.ndarray:
        from .train_encoder import predict_proba
        return predict_proba(self.model, self.tok, list(texts), self.device, max_len=self.meta["max_len"])

    def encode(self, texts, batch_size: int = 128) -> np.ndarray:
        import torch
        texts, out = list(texts), []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                               max_length=self.meta["max_len"], return_tensors="pt").to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def cached(encoder, split: str, texts) -> np.ndarray:
    """encoder.encode(texts), cached on disk for dataset splits."""
    ensure_dirs()
    texts = list(texts)
    key = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"{encoder.name}_{split}_{key}.npy"
    if path.exists():
        return np.load(path)
    E = encoder.encode(texts)
    np.save(path, E)
    return E
