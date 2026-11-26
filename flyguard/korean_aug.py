"""Korean training data that never touches the Korean evaluation suites (source holdout).

    python -m flyguard.korean_aug        # writes data/korean_aug.parquet (text, label, source)

The intervention experiment showed two reasons English-trained detectors fail on Korean: a learned shortcut
("unusual characters mean injection") and an English-only nose that cannot read Korean. This builds Korean
training rows from two sources that share nothing with KoreanGuardrail or JHC56:
  - S-Labs TRAIN prompts translated into Korean by Qwen3-4B (injections and benign alike, so the language
    itself says nothing about the label),
  - ordinary Korean requests written by Qwen3-4B across everyday topics (benign).
Validation, test and every external suite stay untouched; the Korean suites remain a held-out source.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch

from .data import load_splits
from .paths import DATA

OUT = DATA / "korean_aug.parquet"
QWEN3 = "Qwen/Qwen3-4B-Instruct-2507"
N_INJECTION, N_BENIGN_TRANSLATED, PER_TOPIC = 700, 350, 12
TOPICS = ["요리와 레시피", "여행 계획", "파이썬 코딩", "수학 숙제", "건강과 운동", "한국사", "쇼핑과 가격 비교", "회사 이메일 쓰기",
          "대학 과제", "영화와 드라마 추천", "자취 생활 팁", "반려동물", "재테크 기초 상식", "날씨와 옷차림", "자기소개서", "게임",
          "엑셀 사용법", "연애 상담", "번역 부탁", "글 다듬기", "AI와 보안에 대한 일반 질문", "운전과 교통", "음악", "육아"]
HANGUL = re.compile(r"[가-힣]")


def chat(model, tok, messages_list, max_new_tokens, temperature=0.0, batch=24):
    """Generate one reply per message list (left-padded batches)."""
    out = []
    for i in range(0, len(messages_list), batch):
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list[i:i + batch]]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                                 temperature=temperature if temperature > 0 else None, top_p=0.95 if temperature > 0 else None,
                                 pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  generated {min(i + batch, len(messages_list))}/{len(messages_list)}", flush=True)
    return out


def translate(model, tok, texts):
    msgs = [[{"role": "user", "content": "다음 문장을 자연스러운 한국어로 번역하세요. 뜻과 의도는 그대로 두고, 번역문만 한 줄로 쓰세요.\n\n"
              + t}] for t in texts]
    return [r.strip().split("\n")[0].strip(" \"'") for r in chat(model, tok, msgs, max_new_tokens=96)]


def everyday_requests(model, tok):
    msgs = [[{"role": "user", "content": f"한국어 사용자가 AI 비서에게 할 법한 평범한 질문이나 부탁을 {PER_TOPIC}개 써 주세요. 주제: {topic}. "
              "존댓말과 반말을 섞고, 길이도 다양하게 해 주세요. 한 줄에 하나씩, 번호나 따옴표 없이 문장만 쓰세요."}] for topic in TOPICS]
    lines = []
    for reply in chat(model, tok, msgs, max_new_tokens=700, temperature=0.8, batch=8):
        for line in reply.split("\n"):
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip(" \"'")
            if 6 <= len(line) <= 200 and HANGUL.search(line):
                lines.append(line)
    return lines


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN3, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(QWEN3, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    tr = load_splits()["train"]
    rng = np.random.default_rng(0)
    inj = tr[tr.label == 1].text.to_numpy()[rng.permutation((tr.label == 1).sum())[:N_INJECTION]].tolist()
    ben = tr[tr.label == 0].text.to_numpy()[rng.permutation((tr.label == 0).sum())[:N_BENIGN_TRANSLATED]].tolist()
    print(f"translating {len(inj)} injections and {len(ben)} benign prompts", flush=True)
    t_inj, t_ben = translate(model, tok, inj), translate(model, tok, ben)
    print("writing everyday requests", flush=True)
    everyday = everyday_requests(model, tok)
    rows = ([(t, 1, "s-labs 공격 번역") for t in t_inj] + [(t, 0, "s-labs 정상 번역") for t in t_ben]
            + [(t, 0, "qwen 일상 요청") for t in everyday])
    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    df = df[df.text.str.contains(HANGUL)].drop_duplicates("text").reset_index(drop=True)
    df.to_parquet(OUT)
    print(df.groupby(["source", "label"]).size().to_string())
    print(f"wrote {len(df)} rows to {OUT}")
    for s, g in df.groupby("source"):
        print(s, "|", " / ".join(g.text.head(3)))


if __name__ == "__main__":
    main()
