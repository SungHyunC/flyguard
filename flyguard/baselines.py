"""Non-biological reference models.

- TF-IDF (word 1-2 grams + character 3-5 grams) + logistic regression: the strong classical baseline.
- Shallow-feature probe: length, punctuation, case and digits only. If it scores well on train
  but not on validation, the training set contains a shortcut the other models could also exploit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def tfidf_model(C: float = 4.0):
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    )
    return make_pipeline(features, LogisticRegression(C=C, max_iter=3000))


def shallow_features(texts) -> np.ndarray:
    s = pd.Series(list(texts), dtype=str)
    n = s.str.len().clip(lower=1)
    return np.column_stack([
        np.log1p(s.str.len()),
        np.log1p(s.str.split().str.len()),
        s.str.count(r"[^\w\s]") / n,
        s.str.count(r"[A-Z]") / n,
        s.str.count(r"\d") / n,
        s.str.contains("?", regex=False).astype(float),
    ]).astype(np.float32)


def shallow_model():
    return make_pipeline(FunctionTransformer(shallow_features), StandardScaler(), LogisticRegression(max_iter=1000))
