from pathlib import Path
import sys

# This can be tricky to find the root directory of the project
ROOT_DIR = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_DIR))
FIGURES_PATH = ROOT_DIR / "reports/figures"
VIDEO_PATH = ROOT_DIR / "reports/videos"
MODEL_PATH = ROOT_DIR / "models"
DATA_RAW = ROOT_DIR / "data/raw"
DATA_DEBUG = ROOT_DIR / "data/debug"
DATA_MODEL_PATH = ROOT_DIR / "data/model"
TENSORBOARD_LOGS = ROOT_DIR / "tensorboard_logs"
DEBUG_MODE = True
# this depened on the location of ffmpeg on your machine
FFMPEG_PATH = r"C:\Program Files\ffmpeg-6.0-essentials_build\bin"

READ_DATA_FROM_ECB = True  # set to True to read data from ECB instead of local file
