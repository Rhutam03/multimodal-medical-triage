import os
import torch
from models.fusion_model import MultimodalTriageModel

device = torch.device("cpu")

_model = None

def load_model():
    global _model

    if _model is not None:
        return _model

    # 🔹 Absolute path based on THIS file location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    WEIGHTS_PATH = os.path.join(
        BASE_DIR,
        "weights",
        "best_multimodal_model.pt"
    )

    print("Loading weights from:", WEIGHTS_PATH)
    print("File exists:", os.path.exists(WEIGHTS_PATH))

    model = MultimodalTriageModel(num_classes=3)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    return _model
