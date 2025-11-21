import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import copy

"""
References:
  * Original MoCo paper: https://arxiv.org/abs/1911.05722
* Examples in Pytorch:
  * https://www.analyticsvidhya.com/blog/2020/08/moco-v2-in-pytorch/ 
  * https://github.com/facebookresearch/moco/blob/main/main_moco.py
  * https://github.com/facebookresearch/moco/blob/main/moco/builder.py
"""

class MoCo(nn.Module):
    """
    Build a MoCo encoder: query encoder & momentum-updated key encoder
    """
    def __init__(self, dim=128, K=65536, m=0.999, T=0.2, pretrained=False, device='cuda'):
        super().__init__()
        self.device = device
        self.pretrained = pretrained

        # ----------------------------------------------------
        # 1. Query Encoder (ResNet50 backbone)
        # ----------------------------------------------------
        if pretrained:
            # Use V2 (default) pretrained weights for ResNet50
            print("Using ImageNet pretrained weights for ResNet50: For encoder_q")
            self.encoder_q = resnet50(weights="IMAGENET1K_V2")
        else:
            print("Not using pretrained weights for ResNet50: Encoder initialized randomly")
            self.encoder_q = resnet50(weights=None)

        # Projection head (MLP)
        # !! Should remove final FC layer from ResNet50 !!
        feat_dim = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Identity()  # remove FC head

        # MLP for projection
        self.mlp_q = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        # ----------------------------------------------------
        # 2. Key Encoder (initialized as a copy but momentum-updated)
        # ----------------------------------------------------
        # Key encoder initialization - should be by deeopcopy instead of instantiate new instance
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.encoder_k.fc = nn.Identity()

        self.mlp_k = copy.deepcopy(self.mlp_q)

        # Freeze encoder
        # Do not use backprop on key encoder
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.mlp_k.parameters():
            p.requires_grad = False

        # ----------------------------------------------------
        # 3. MoCo queue - Important to save the queue of past keys
        # ----------------------------------------------------
        # Create the queue
        self.K = K
        self.m = m
        self.T = T

        self.register_buffer("queue", torch.randn(dim, K))
        # Stored keys correspond to the negative keys
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Move to device
        self.encoder_q = self.encoder_q.to(self.device)
        self.encoder_k = self.encoder_k.to(self.device)
        self.mlp_q = self.mlp_q.to(self.device)
        self.mlp_k = self.mlp_k.to(self.device)
        self.queue = self.queue.to(self.device)
        self.queue_ptr = self.queue_ptr.to(self.device)
    
    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update for key encoder parameters
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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
