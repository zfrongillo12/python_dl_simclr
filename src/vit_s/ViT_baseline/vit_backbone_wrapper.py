class ViTBackboneTimm(nn.Module):
    def __init__(self, vit_moco_model):
        super().__init__()
        self.encoder = vit_moco_model.encoder_q
        self.embed_dim = self.encoder.num_features

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        print("Timm forward_features shape:", feats.shape)

        if feats.dim() == 3:
            print("Returning CLS:", feats[:, 0].shape)
            return feats[:, 0]

        print("Returning pooled vector:", feats.shape)
        return feats
