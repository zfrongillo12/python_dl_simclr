import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy

# ----------------------------------------------------
# Conv Stem -> Patch Embedding
# ----------------------------------------------------
class ConvPatchEmbed(nn.Module):
    """
    Small conv stem to produce patch-like embeddings.
    Input: (B,1,H,W) for CXR
    Output: (B, N_patches, embed_dim) and (H_grid, W_grid)
    """
    def __init__(self, in_chans=3, embed_dim=384):
        super().__init__()
         # Experiment: 2 conv layers with stride to reduce spatial dims
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim
        # learned positional embeddings will be added in ViTHybrid

    def forward(self, x):
        # x: (B,1,H,W)
        x = self.proj(x) # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

# ----------------------------------------------------
# Simple ViT Transformer Encoder
# ----------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, num_layers=12, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio)
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.blocks(x) # (B, num_patches, embed_dim)
        return x

# ----------------------------------------------------
# MoCo ViT Hybrid
# ----------------------------------------------------
class ViTMoCo(nn.Module):
    """
    Build a MoCo encoder: query encoder & momentum-updated key encoder
    MoCo with Vision Transformer backbone (ViT) hybrid

    Support 3-channel or 1-channel input (e.g., CXR)

    Update: Use ConvPatchEmbed + SimpleViT instead of torchvision ViT
    1. ConvPatchEmbed: small conv stem to produce patch-like embeddings
    """
    def __init__(self, proj_dim=128, K=65536, m=0.999, T=0.2,
                 embed_dim=384, in_chans=3, device='cuda'):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.embed_dim = embed_dim

        # -----------------------------------------------------------------------------
        # 1. Query / Online Encoder (Vision Transformer backbone)
        # -----------------------------------------------------------------------------
        # Update to use the ConvPatchEmbed (instead of flattened) + SimpleViT
        self.encoder_q = nn.ModuleDict({
            'patch_embed': ConvPatchEmbed(in_chans=in_chans, embed_dim=embed_dim),
            'transformer': TransformerEncoder(embed_dim=embed_dim)
        })
        self.encoder_q_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # -----------------------------------------------------------------------------
        # 2. Key Encoder (momentum-updated)
        # -----------------------------------------------------------------------------
        # MLP for projection
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.encoder_k_proj = copy.deepcopy(self.encoder_q_proj)

        # -----------------------------------------------------------------------------
        # Queue for contrastive learning
        # -----------------------------------------------------------------------------
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.to(device)

    # -------------------
    # Momentum update
    # -------------------
    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update for key encoder parameters
        
        m * k + (1-m) * q
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.encoder_q_proj.parameters(), self.encoder_k_proj.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # -------------------
    # Queue update
    # -------------------
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update FIFO queue with new keys
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # FIFO queue update
        if self.K % batch_size != 0:
            raise ValueError(f"Queue size ({self.K}) must be divisible by batch size ({batch_size})")
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    # -------------------
    # Forward pass for MoCo
    # -------------------
    def forward(self, im_q, im_k=None):
        # Query features
        q = self.encoder_q['patch_embed'](im_q)
        q = self.encoder_q['transformer'](q)
        q = q.mean(dim=1)  # global avg pool
        q = self.encoder_q_proj(q)
        q = F.normalize(q, dim=1)

        if im_k is None:
            return q  # for evaluation

        # Compute key features - key encoder (no grad)
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.encoder_k['patch_embed'](im_k)
            k = self.encoder_k['transformer'](k)
            k = k.mean(dim=1)
            k = self.encoder_k_proj(k)
            k = F.normalize(k, dim=1)

        # Contrastive logits
        
        # Positive logits: q*k - (B,1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # Nx1
        # Negative logits: q*queue - (B,K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # NxK
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        
        # Labels: positive key indicators (first column)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        # enqueue and dequeue
        self._dequeue_and_enqueue(k)

        return logits, labels
