import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

PCAP_DIR = os.path.join(BASE_DIR, "pcaps")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Files
CIC_MERGED_FILE = os.path.join(DATA_PROCESSED, "cic_merged.csv")
NBAIOT_FILE = os.path.join(DATA_RAW, "nbaiot.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
