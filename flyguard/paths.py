"""Project paths. Everything lives under the project root so the folder can be moved as a whole."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CONNECTOME = DATA / "connectome"
ARTIFACTS = ROOT / "artifacts"
DEMO = ARTIFACTS / "demo"           # what the web demo loads (flyguard.build)
CACHE = ARTIFACTS / "cache"
REPORTS = ROOT / "reports"
WEB = ROOT / "web"


def ensure_dirs() -> None:
    for p in (RAW, CONNECTOME, ARTIFACTS, CACHE, REPORTS):
        p.mkdir(parents=True, exist_ok=True)
