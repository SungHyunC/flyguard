"""External evaluation suites: never trained on, used only to measure generalization.

    python -m flyguard.external        # download once (about 70 MB) and print a summary

Every row has text, label (1 = injection or jailbreak, 0 = benign), suite, group, kind (direct /
indirect), lang, and suite-specific detail columns.

- Horizon-Labs/prompt-injection-eval-suite: 15 configs. `slabs_test` is left out because it is our own
  test split. Its label definition matches ours: overriding or extracting instructions (jailbreak
  personas included) is 1; a plainly harmful request with no override attempt is 0.
- kimchunsik03/KoreanGuardrail (CC-BY-4.0): Korean injections, prompt leaks and hard negatives, each
  also rewritten with Hangul obfuscation (jamo decomposition, initial consonants, tensing, broken
  spacing, zero-width characters) at two strengths. Rows with technique "clean" are the controls.
- JHC56/korean-prompt-injection-registers (BSD-2): the same 221 items in English and 7 Korean registers
  (polite, very polite, casual, Gen-Z slang, emoticon, angry, threatening), plus the published
  predictions of Korean LLMs on the same rows.
"""
from __future__ import annotations

import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

from .paths import EXTERNAL

HORIZON_REPO = "Horizon-Labs/prompt-injection-eval-suite"
HORIZON = {                       # config -> what it tests
    "notinject": "over_defense", "xstest": "over_defense", "orbench_hard": "over_defense",
    "deepset_test": "direct", "neuralchemy_test": "direct", "synthetic_direct_test": "direct",
    "jackhhao_test": "jailbreak", "simsonsun_jailbreaks": "jailbreak",
    "boundary_pairs_test": "agentic", "agentic5k_test": "agentic",
    "bipia": "indirect", "piarena": "indirect", "llmail_phase2": "indirect", "synthetic_docs_test": "indirect",
    "synthetic_v2_test": "mixed",
}
KOGUARD_REPO = "kimchunsik03/KoreanGuardrail"
JHC56_RAW = "https://raw.githubusercontent.com/JHC56/korean-prompt-injection-registers/HEAD/"
JHC56_LLMS = {                    # result file stem -> display name
    "exaone_2.4b": "EXAONE 2.4B", "exaone_7.8b": "EXAONE 7.8B", "hcx_1.5b": "HyperCLOVA X 1.5B",
    "kanana_2.1b": "Kanana 2.1B", "kanana_8b": "Kanana 8B",
}
JHC56_FILES = (["data/dataset.jsonl", "results/kanana_safeguard_2.1b.jsonl"]
               + [f"results/{m}_{p}.jsonl" for m in JHC56_LLMS for p in ("baseline", "definition")])


def download() -> None:
    for config in HORIZON:
        if not (EXTERNAL / "horizon" / "data" / f"{config}.parquet").exists():
            hf_hub_download(HORIZON_REPO, f"data/{config}.parquet", repo_type="dataset", local_dir=EXTERNAL / "horizon")
    if not (EXTERNAL / "koreanguardrail" / "benchmark.jsonl").exists():
        hf_hub_download(KOGUARD_REPO, "benchmark.jsonl", repo_type="dataset", local_dir=EXTERNAL / "koreanguardrail")
    for f in JHC56_FILES:
        target = EXTERNAL / "jhc56" / f
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(JHC56_RAW + f, target)


def load_suites() -> pd.DataFrame:
    download()
    frames = []
    for config, group in HORIZON.items():
        df = pd.read_parquet(EXTERNAL / "horizon" / "data" / f"{config}.parquet")
        frames.append(pd.DataFrame({
            "text": df["text"].astype(str), "label": df["label"].astype(int), "suite": config, "group": group,
            "kind": df["kind"] if "kind" in df else "direct", "lang": df["lang"] if "lang" in df else "en",
        }))
    kg = pd.read_json(EXTERNAL / "koreanguardrail" / "benchmark.jsonl", lines=True)
    frames.append(pd.DataFrame({
        "text": kg["text"], "label": (kg["label"] == "attack").astype(int), "suite": "koreanguardrail",
        "group": "korean", "kind": "direct", "lang": "ko", "technique": kg["technique"],
        "intensity": kg["intensity"], "category": kg["category"], "item": kg["seed_id"],
    }))
    jr = pd.read_json(EXTERNAL / "jhc56" / "data" / "dataset.jsonl", lines=True)
    frames.append(pd.DataFrame({
        "text": jr["text"], "label": jr["label"].astype(int), "suite": "jhc56_registers", "group": "korean",
        "kind": "direct", "lang": jr["lang"], "register": jr["register"], "item": jr["id"],
        "technique": jr["technique"],
    }))
    return pd.concat(frames, ignore_index=True)


JHC56_STYLE = {"en": "english", "존댓말": "polite", "반말": "casual", "젠지": "genz"}   # older result files


def jhc56_llm_predictions() -> pd.DataFrame:
    """Published predictions of Korean LLMs on the register dataset: model, prompt, item, register, label, pred.
    Unparsed answers (pred missing) stay missing; they count as "not an injection" in the report."""
    download()
    rows = []
    for stem, name in JHC56_LLMS.items():
        for prompt in ("baseline", "definition"):
            df = pd.read_json(EXTERNAL / "jhc56" / "results" / f"{stem}_{prompt}.jsonl", lines=True)
            if "register" not in df:
                df["register"] = df["style"].map(lambda s: JHC56_STYLE.get(s, s))
            rows.append(df.assign(model=name, prompt=prompt))
    df = pd.read_json(EXTERNAL / "jhc56" / "results" / "kanana_safeguard_2.1b.jsonl", lines=True)
    rows.append(df.assign(model="Kanana Safeguard 2.1B", prompt="guard"))
    out = pd.concat(rows, ignore_index=True).rename(columns={"id": "item"})
    return out[["model", "prompt", "item", "register", "label", "pred"]]


if __name__ == "__main__":
    s = load_suites()
    print(s.groupby(["group", "suite"]).agg(n=("label", "size"), positives=("label", "sum"),
                                            median_chars=("text", lambda t: int(t.str.len().median()))).to_string())
    print("languages:", s.lang.value_counts().head(8).to_dict())
    print(jhc56_llm_predictions().groupby(["model", "prompt"]).size().to_string())
