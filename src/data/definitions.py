from pathlib import Path
import sys

# This can be tricky to find the root directory of the project
ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
FIGURES_PATH = ROOT_DIR / "reports/figures"
MODEL_PATH = ROOT_DIR / "models"
DATA_RAW = ROOT_DIR / "data/raw"
DATA_DEBUG = ROOT_DIR / "data/debug"
TENSORBOARD_LOGS = ROOT_DIR / "tensorboard_logs"
DEBUG_MODE = True
# this depened on the location of ffmpeg on your machine
FFMPEG_PATH = r"C:\Program Files\ffmpeg-6.0-essentials_build\bin"
