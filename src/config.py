from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

OUTPUT_MODELS = BASE_DIR / "outputs" / "models"
OUTPUT_REPORTS = BASE_DIR / "outputs" / "reports"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUTPUT_METRICS = BASE_DIR / "outputs" / "metrics"

SSCD1_PATH = DATA_RAW / "SSCD1.csv"
SSCD2_PATH = DATA_RAW / "SSCD2.csv"

TEXT_COLUMN = "message"
LABEL_COLUMN = "label"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
USE_TFIDF = True
