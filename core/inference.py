import torch

from core.config import (
    MODEL_WEIGHTS_PATH,
    DEVICE,
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
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# -------------------------
# Inference function
# -------------------------

def predict(image_path: str, text: str) -> int:
    """
    Runs multimodal inference and returns class index
    """

    # Preprocess inputs
    image_tensor = preprocess_image(image_path).to(DEVICE)
    input_ids, attention_mask = preprocess_text(text)

    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            image_tensor,
            input_ids,
            attention_mask
        )

    # Class prediction
    pred = outputs.argmax(dim=1).item()
    return pred
