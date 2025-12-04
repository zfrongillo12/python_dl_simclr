import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from copy import deepcopy


# ----------------------------------------------------
# Conv Stem -> Patch Embedding (Reduced tokens)
# ----------------------------------------------------
class ConvPatchEmbed(nn.Module):
    """
    Small conv stem to produce patch-like (ViT) embeddings.
    Input: (B,3 input channels,H,W) for CXR
     * 3 convs -> total stride = 8 (224 -> 28)


    Outputs: (B, N_patches, embed_dim)
    """
    def __init__(self, in_chans=3, embed_dim=384):
        super().__init__()
        # Experiment (Update on 12/3 from 2 layers): 3 conv layers with stride to reduce spatial dims
        # Refine conv stem; from 3136 tokens to 784 tokens for 224x224 input  
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.embed_dim = embed_dim
        # learned positional embeddings will be added in ViTHybrid

    def forward(self, x):
        # x: (B,1,H,W)
        x = self.proj(x) # (B, C, H/8, W/8)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x, (H, W) # return spatial info


# ----------------------------------------------------
# Transformer Encoder
# ----------------------------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=384, depth=12, heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x) # (B, N, D)


# ----------------------------------------------------
# Learnable Positional Embedding + CLS Token
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1) # prepend CLS
        return x + self.pos_embed # add positions


# ----------------------------------------------------
# Projection Head (2-layer MLP)
# ----------------------------------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# ----------------------------------------------------
# MoCo ViT Hybrid
# ----------------------------------------------------
class ViTMoCo(nn.Module):
    """
    MoCo v2-like ViT encoder:
      * Conv stem
      * Positional embeddings + CLS
      * Transformer encoder (SimpleViT)
      * Projection head
      * Momentum encoder with queue
    
    Build a MoCo encoder: query encoder & momentum-updated key encoder
    MoCo with Vision Transformer backbone (ViT) hybrid

    Support 3-channel or 1-channel input (e.g., CXR)

    Update: Use ConvPatchEmbed + SimpleViT instead of torchvision ViT
     * ConvPatchEmbed: small conv stem to produce patch-like embeddings
    """
    def __init__(self, proj_dim=128, K=65536, m=0.999, T=0.2,
                 embed_dim=384, in_chans=3, device='cuda', in_shape=224):
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
        self.patch_embed = ConvPatchEmbed(in_chans, embed_dim)

        # Dummy forward pass to infer token count
        dummy = torch.zeros(1, in_chans, in_shape, in_shape)
        tokens, (H, W) = self.patch_embed(dummy)
        num_patches = tokens.shape[1]

        # Add positional encoding + CLS token
        self.pos_encoding = PositionalEncoding(embed_dim, num_patches)
        self.transformer = SimpleTransformer(embed_dim)

        self.proj_head = ProjectionMLP(embed_dim, proj_dim)

        # -----------------------------------------------------------------------------
        # 2. Key Encoder (momentum-updated)
        # -----------------------------------------------------------------------------
        # MLP for projection
        self.patch_embed_k = copy.deepcopy(self.patch_embed)
        self.pos_encoding_k = copy.deepcopy(self.pos_encoding)
        self.transformer_k = copy.deepcopy(self.transformer)
        self.proj_head_k = copy.deepcopy(self.proj_head)

        # -----------------------------------------------------------------------------
        # Queue for contrastive learning
        # -----------------------------------------------------------------------------
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.to(device)

    # ------------------------------------------------
    # Momentum update for key encoder
    # ------------------------------------------------
    @torch.no_grad()
    def momentum_update(self):
        """
        Momentum update for key encoder parameters
        
        m * k + (1-m) * q
        """
        def _update(q, k):
            for p_q, p_k in zip(q.parameters(), k.parameters()):
                p_k.data = p_k.data * self.m + p_q.data * (1 - self.m)

        _update(self.patch_embed, self.patch_embed_k)
        # Positional embeddings
        _update(self.pos_encoding, self.pos_encoding_k)
        _update(self.transformer, self.transformer_k)
        _update(self.proj_head, self.proj_head_k)

    # ------------------------------------------------
    # Queue update
    # ------------------------------------------------
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Update FIFO queue with new keys
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.K % batch_size == 0, "K must be divisible by batch size"

        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    # ------------------------------------------------
    # Encode function (online or momentum)
    # ------------------------------------------------
    def _encode(self, x, patch_embed, pos_enc, transformer, proj_head):
        tokens, _ = patch_embed(x)
        x = pos_enc(tokens)
        x = transformer(x)
        cls = x[:, 0] # use CLS token
        z = proj_head(cls)
        return F.normalize(z, dim=1)

    # ------------------------------------------------
    # Forward for MoCo pretraining
    # ------------------------------------------------
    def forward(self, im_q, im_k=None):
        # Online encoder (query)
        q = self._encode(
            im_q, self.patch_embed, self.pos_encoding,
            self.transformer, self.proj_head
        )

        if im_k is None:
            return q # for evaluation

        # Momentum encoder (key)
        with torch.no_grad():
            self.momentum_update()
            k = self._encode(
                im_k, self.patch_embed_k, self.pos_encoding_k,
                self.transformer_k, self.proj_head_k
            )

        # --------------------------
        # Contrastive logits
        # --------------------------
        # Positive logits: q*k - (B,1)
        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)     # (N,1)
        # Negative logits: q*queue - (B,K)
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.clone()) # (N,K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # Labels: positive key indicators (first column)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels
