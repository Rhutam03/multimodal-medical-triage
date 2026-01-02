import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.output_dim = 768

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS token embedding
        return outputs.last_hidden_state[:, 0, :]
