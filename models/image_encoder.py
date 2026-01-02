import torch.nn as nn 
from torchvision.models import resnet18  

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(weights="IMAGENET1K_V1")

        # Remove final classification layer
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.output_dim = 512

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
