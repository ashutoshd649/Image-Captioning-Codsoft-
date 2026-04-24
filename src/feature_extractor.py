"""
src/feature_extractor.py
CNN encoder — wraps ResNet50 or VGG16 from torchvision.
Removes the classification head and returns image feature vectors.
"""

import torch
import torch.nn as nn
from torchvision import models
import config


class CNNEncoder(nn.Module):
    """
    Extracts a fixed-size feature vector from an image.

    Args:
        model_name  : "resnet50" or "vgg16"
        feature_dim : output embedding size (projected via linear layer)
        fine_tune   : if True, unfreeze CNN weights during training
    """

    def __init__(self,
                 model_name: str  = config.ENCODER_MODEL,
                 feature_dim: int = config.EMBED_DIM,
                 fine_tune: bool  = config.FINE_TUNE_ENCODER):
        super().__init__()
        self.model_name  = model_name
        self.feature_dim = feature_dim

        if model_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = base.fc.in_features          # 2048
            self.backbone = nn.Sequential(*list(base.children())[:-1])  # drop fc
        elif model_name == "vgg16":
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            in_features = 512 * 7 * 7                  # flatten of last pool
            self.backbone = base.features
            self.avgpool  = base.avgpool
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.project = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Freeze or unfreeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images : (B, 3, 224, 224)
        Returns:
            features : (B, feature_dim)
        """
        if self.model_name == "resnet50":
            feats = self.backbone(images)          # (B, 2048, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, 2048)
        else:  # vgg16
            feats = self.backbone(images)
            feats = self.avgpool(feats)
            feats = feats.view(feats.size(0), -1)  # (B, 25088)

        return self.project(feats)                 # (B, feature_dim)

    def fine_tune(self, enable: bool = True):
        """Toggle backbone fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = enable
