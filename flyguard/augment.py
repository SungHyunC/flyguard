"""Training-data augmentation that breaks the "unusual characters mean injection" shortcut.

In S-Labs train the 22 prompts with non-ASCII characters are all obfuscation attacks, so a fine-tuned model
can learn that script alone predicts the label. `add_greetings` rewrites the same share of benign AND
injection prompts with a harmless greeting in another script, so script no longer predicts the label.
The phrases of the character test in flyguard.shortcuts (" 감사합니다.", " merci beaucoup") are not among
the greetings, so that test stays a held-out check. Used by flyguard.intervene and
`python -m flyguard.train_encoder --augment-scripts 0.2`.
"""
from __future__ import annotations

import numpy as np

GREETINGS = ["안녕하세요!", "좋은 하루 되세요.", "ありがとうございます。", "よろしくお願いします。", "谢谢你。", "你好！",
             "Спасибо!", "Добрый день.", "¡Hola, buenos días!", "😊", "👍", "🙏"]


def add_greetings(texts, labels, share: float = 0.2, seed: int = 0) -> list[str]:
    """Same prompts, same labels; `share` of each class gets a greeting before or after it."""
    rng = np.random.default_rng(seed)
    texts, labels = list(texts), np.asarray(labels)
    for cls in (0, 1):
        idx = np.flatnonzero(labels == cls)
        for i in rng.choice(idx, int(share * len(idx)), replace=False):
            g = GREETINGS[rng.integers(len(GREETINGS))]
            texts[i] = f"{g} {texts[i]}" if rng.random() < 0.5 else f"{texts[i]} {g}"
    return texts
