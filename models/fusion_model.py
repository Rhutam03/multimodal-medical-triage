import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder

class MultimodalTriageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        fusion_dim = (
            self.image_encoder.output_dim +
            self.text_encoder.output_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_emb = self.image_encoder(image)
        txt_emb = self.text_encoder(input_ids, attention_mask)

        fused  = torch.cat([img_emb, txt_emb], dim=1)
        fused = nn.functional.normalize(fused)
        fused = nn.functional.relu(fused)
        fused = nn.functional.dropout(fused, p=0.3, training=self.training)

        return self.classifier(fused)
