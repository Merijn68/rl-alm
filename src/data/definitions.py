from pathlib import Path

ROOT_DIR = Path(__file__).parents[2].absolute()
FIGURES_PATH = ROOT_DIR / "reports/figures"
MODEL_PATH = ROOT_DIR / "models"
DATA_RAW = ROOT_DIR / "data/raw"
DATA_DEBUG = ROOT_DIR / "data/debug"
DEBUG_MODE = True
