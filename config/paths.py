import os

# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Paths
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR,
    "best_multimodal_model.pt"
)