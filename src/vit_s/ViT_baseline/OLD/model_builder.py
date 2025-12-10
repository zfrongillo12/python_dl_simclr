import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy

class MoCo(nn.Module):
    """
    Build a MoCo encoder: query encoder & momentum-updated key encoder
    MoCo with Vision Transformer backbone (timm ViT, e.g., ViT-S/16)
    """
    def __init__(self, vit_name="vit_small_patch16_224",
                 dim=128, K=65536, m=0.999, T=0.2,
                 pretrained=False, device='cuda'):
        super().__init__()
        self.device = device

        # ----------------------------------------------------
        # 1. Query Encoder (Vision Transformer backbone)
        # ----------------------------------------------------
        print(f"Using timm ViT backbone: {vit_name}")

        # Create timm ViT
        self.encoder_q = timm.create_model(
            vit_name,
            pretrained=pretrained,
            num_classes=0  # removes classifier head
        )

        # Extract CLS embedding dimension
        feat_dim = self.encoder_q.num_features

        # Projection MLP (standard MoCo v2)
        self.mlp_q = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
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
        self.queue = F.normalize(self.queue, dim=0)
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
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
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

    def forward(self, im_q, im_k=None, return_features=False):
        # Compute query features (ViT forward_features → CLS)
        q = self.encoder_q.forward_features(im_q)
        q = self.mlp_q(q)
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self.encoder_k.forward_features(im_k)
            k = self.mlp_k(k)
            k = F.normalize(k, dim=1)

        # Positive logits: q*k
        pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1)

        # Negative logits: q*queue
        neg = torch.einsum('nc,ck->nk', q, self.queue.clone().detach())

        logits = torch.cat([pos, neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)

        return logits, labels, q, k
    
    # Writing method to fetch trainable parameters (query encoder and the mlp head)
    # ------------------------------------------------
    def get_trainable_parameters(self):
        return list(self.encoder_q.parameters()) + list(self.mlp_q.parameters())

    # Saving the the encoder_q and the mlp_q for downstream use
    # ------------------------------------------------
    def get_encoder_state(self):
        return {
            "encoder_q": self.encoder_q.state_dict(),
            "mlp_q": self.mlp_q.state_dict(),
            "embed_dim": self.encoder_q.num_features
        }

