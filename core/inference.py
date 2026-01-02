import torch

from core.config import (
    MODEL_WEIGHTS_PATH,
    device,
    NUM_CLASSES
)

from models.fusion_model import MultimodalTriageModel
from preprocessing.image_preprocess import preprocess_image
from preprocessing.text_preprocess import preprocess_text

# -------------------------
# Load model ONCE at startup
# -------------------------

model = MultimodalTriageModel(num_classes=NUM_CLASSES)

state_dict = torch.load(
    MODEL_WEIGHTS_PATH,
    map_location=device
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -------------------------
# Inference function
# -------------------------

def predict(image_path: str, text: str) -> int:
    # preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # preprocess text
    input_ids, attention_mask = preprocess_text(text)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # inference
    with torch.no_grad():
        outputs = model(
            image_tensor.unsqueeze(0),
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0)
        )

    # predicted class
    triage_level = outputs.argmax(dim=1).item()

    return triage_level

