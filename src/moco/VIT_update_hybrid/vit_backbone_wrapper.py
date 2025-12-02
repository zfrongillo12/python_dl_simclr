import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Linear Classifier for ViTMoCo
# -------------------------------

# Alternative Structure for testing ViTMoCo backbone
"""
class LinearClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Pass through patch embedding
        x = self.encoder['patch_embed'](x)
        # Pass through transformer encoder
        x = self.encoder['transformer'](x)
        # Mean pooling over tokens (assumes shape B x N x embed_dim)
        x = x.mean(dim=1)
        # Linear classification
        x = self.fc(x)
        return x
"""

# ==============================
# ViT Backbone Wrapper - extracts features from ViTMoCo encoder to be used for linear evaluation
# Testing - ViTBackbone Wrapper
# ==============================
class ViTBackbone(nn.Module):
    def __init__(self, encoder_q):
        super().__init__()
        self.patch_embed = encoder_q['patch_embed']
        self.transformer = encoder_q['transformer']

    def forward(self, x):
        x = self.patch_embed(x) # (B, N, D)
        x = self.transformer(x) # (B, N, D)

        # Use CLS token = first token
        cls = x[:, 0]

        return cls

# ==============================
# Fine Tuning - ViTBackbone Wrapper
# ==============================
class FT_ViTBackbone(nn.Module):
    def __init__(self, encoder_q, embed_dim):
        super().__init__()
        self.patch_embed = encoder_q.patch_embed
        self.transformer = encoder_q.transformer
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.patch_embed(x)        # (B, N, D)
        x = self.transformer(x)        # (B, N, D)
        return x[:, 0]                 # CLS token
    
# -------------------------------
# Fine Tuning - Model
# -------------------------------
class FT_ViT_FinetuneModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        cls = self.backbone(x)
        return self.classifier(cls)