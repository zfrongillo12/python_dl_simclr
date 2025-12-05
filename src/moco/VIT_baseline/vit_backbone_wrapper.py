import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# ViT Backbone Wrapper - extracts features from ViTMoCo encoder to be used for linear evaluation
# Testing - ViTBackbone Wrapper
# ==============================
class ViTBackbone(nn.Module):
    def __init__(self, vit_moco_model):
        super().__init__()
        self.patch_embed = vit_moco_model.patch_embed
        self.pos_encoding = vit_moco_model.pos_encoding
        self.transformer = vit_moco_model.transformer
        self.embed_dim = vit_moco_model.embed_dim

    @torch.no_grad()
    def forward(self, x):
        # Patch embedding
        tokens, _ = self.patch_embed(x)   # (B, N, C)

        # Positional encoding + CLS
        tokens = self.pos_encoding(tokens)  # (B, N+1, C)

        # Pass through transformer
        out = self.transformer(tokens)      # (B, N+1, C)

        # return CLS token only
        cls = out[:, 0]   # (B, C)
        return cls

# ==============================
# Fine Tuning - ViTBackbone Wrapper
# ==============================
class FT_ViTBackbone(nn.Module):
    def __init__(self, encoder_q):
        super().__init__()
        self.vit = encoder_q     # torchvision ViT-B/16
        self.embed_dim = encoder_q.hidden_dim

    def forward(self, x):
        # Forward through ViT (outputs CLS token embedding)
        return self.vit(x)       # returns (B, embed_dim)

    
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