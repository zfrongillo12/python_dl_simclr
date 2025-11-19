import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# Creating the projection head for MoCo (following the pattern set in MoCo v2 MLP pattern)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x


# Creating the MoCo model wrapper with query encoder, key encoder, momentum update, and queue
class MoCoModel(nn.Module):
    def __init__(self, embedding_dim, queue_size, momentum, temperature):
        super().__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Creating the query encoder backbone
        resnet_query = resnet50(weights="IMAGENET1K_V1")
        feature_dimension = resnet_query.fc.in_features
        resnet_query.fc = ProjectionHead(feature_dimension, embedding_dim)
        self.encoder_query = resnet_query

        # Creating the key encoder backbone as a momentum copy
        resnet_key = resnet50(weights="IMAGENET1K_V1")
        resnet_key.fc = ProjectionHead(feature_dimension, embedding_dim)
        self.encoder_key = resnet_key

        # Copying parameters from the query encoder to the key encoder
        for key_param, query_param in zip(self.encoder_key.parameters(), self.encoder_query.parameters()):
            key_param.data.copy_(query_param.data)
            key_param.requires_grad = False

        # Creating the queue for negative keys
        self.register_buffer("queue", torch.randn(embedding_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)

        # Creating the pointer for inserting new keys into the queue
        self.register_buffer("queue_pointer", torch.zeros(1, dtype=torch.long))

    # Momentum update for the key encoder
    def update_key_encoder(self):
        for key_param, query_param in zip(self.encoder_key.parameters(), self.encoder_query.parameters()):
            key_param.data = key_param.data * self.momentum + query_param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def enqueue_dequeue(self, keys):
        # Inserting new keys into the queue in a circular manner
        batch_size = keys.shape[0]
        pointer = int(self.queue_pointer.item())

        end_pointer = pointer + batch_size

        if end_pointer <= self.queue_size:
            self.queue[:, pointer:end_pointer] = keys.T
        else:
            first_len = self.queue_size - pointer
            second_len = batch_size - first_len
            self.queue[:, pointer:] = keys[:first_len].T
            self.queue[:, :second_len] = keys[first_len:].T

        new_pointer = (pointer + batch_size) % self.queue_size
        self.queue_pointer[0] = new_pointer

    def forward(self, query_images, key_images):
        # Forward pass for the query encoder
        query_features = self.encoder_query(query_images)
        query_features = F.normalize(query_features, dim=1)

        # Momentum update for the key encoder
        with torch.no_grad():
            self.update_key_encoder()
            key_features = self.encoder_key(key_images)
            key_features = F.normalize(key_features, dim=1)

        # Computing positive logits
        positive_logits = torch.einsum("nc,nc->n", [query_features, key_features]).unsqueeze(1)

        # Computing negative logits against the queue
        negative_logits = torch.einsum("nc,ck->nk", [query_features, self.queue.clone().detach()])

        # Combining positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.temperature

        # Creating labels for contrastive loss
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Updating queue with new key features
        self.enqueue_dequeue(key_features)

        return logits, labels
