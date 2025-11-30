import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

import copy

class MoCo(nn.Module):
    """
    Build a MoCo encoder: query encoder & momentum-updated key encoder
    MoCo with Vision Transformer backbone (ViT-B/16)
    """
    def __init__(self, dim=128, K=65536, m=0.999, T=0.2, pretrained=False, device='cuda'):
        super().__init__()
        self.device = device

        # ----------------------------------------------------
        # 1. Query Encoder (Vision Transformer backbone)
        # ----------------------------------------------------
        if pretrained:
            print("Using pretrained ImageNet weights for ViT-B/16 (encoder_q)")
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            self.encoder_q = vit_b_16(weights=weights)
        else:
            print("Not using pretrained weights — ViT initialized randomly")
            self.encoder_q = vit_b_16(weights=None)

        # Remove classifier head
        feat_dim = self.encoder_q.heads.head.in_features
        self.encoder_q.heads = nn.Identity()

        # MLP for projection
        self.mlp_q = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        # ----------------------------------------------------
        # 2. Key Encoder
        # ----------------------------------------------------
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.mlp_k = copy.deepcopy(self.mlp_q)

        # Freeze key encoder
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.mlp_k.parameters():
            p.requires_grad = False

        # ----------------------------------------------------
        # 3. Queue
        # ----------------------------------------------------
        self.K = K
        self.m = m
        self.T = T

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Move to device
        self.encoder_q = self.encoder_q.to(device)
        self.encoder_k = self.encoder_k.to(device)
        self.mlp_q = self.mlp_q.to(device)
        self.mlp_k = self.mlp_k.to(device)
        self.queue = self.queue.to(device)
        self.queue_ptr = self.queue_ptr.to(device)

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update for key encoder parameters
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """
        Update FIFO queue with new keys
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        # Compute query features
        q = self.encoder_q(im_q)
        q = self.mlp_q(q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.mlp_k(k)
            k = F.normalize(k, dim=1)

        # Positive logits: q*k
        pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)

        # Negative logits: q*queue
        neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())

        logits = torch.cat([pos, neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

        return logits, labels, k
