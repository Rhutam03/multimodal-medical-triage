import os
import torch

# Root of the project
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# -------- Paths --------
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_WEIGHTS_PATH = os.path.join(
    WEIGHTS_DIR,
    "best_multimodal_model.pt"
)

# -------- Runtime --------
DEVICE = torch.device("cpu")  # keep CPU for now

# -------- Model settings --------
NUM_CLASSES = 3
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 128
