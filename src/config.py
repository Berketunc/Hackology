from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
CHECKPOINT = "facebook/esm2_t33_650M_UR50D"
HIDDEN_SIZE = 1280
SEED = 0
