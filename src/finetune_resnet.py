import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

from classification_dataset import get_train_val_loaders

from PIL import Image
import pandas as pd
import os
from tqdm import tqdm


# ------- CONFIG -------
TRAIN_CSV = 'final_project_updated_names_train.csv'
VAL_CSV = 'final_project_updated_names_val.csv'
ROOT_DIR = '/path/to/dataset'
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 3  # e.g., Pneumonia vs. No Finding vs. Other
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PRETRAINED_ENCODER = 'moco_resnet50_encoder.pth'
LR = 1e-4
N_EPOCHS = 40
# ----------------------

def run_finetune_training():
    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(backbone.parameters(), lr=LR)

    for epoch in range(N_EPOCHS):
        backbone.train()
        loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch}')
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = backbone(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({'loss': loss.item()})

        # optionally validate
        backbone.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = backbone(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f'Epoch {epoch}: val_acc = {correct/total:.4f}')

    # Save finetuned model
    torch.save({'model_state': backbone.state_dict()}, 'finetuned_model.pth')
    print('Saved finetuned_model.pth')

if __name__ == "__main__":
    # Data loaders
    train_loader, val_loader = get_train_val_loaders(
        TRAIN_CSV,
        VAL_CSV,
        ROOT_DIR,
        BATCH_SIZE,
        NUM_WORKERS
    )

    # build model and load pretrained encoder
    try:
        from torchvision.models import ResNet50_Weights
        backbone = resnet50(weights=None)
    except Exception:
        backbone = resnet50(pretrained=False)

    # Modify final layer for NUM_CLASSES
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, NUM_CLASSES)
    backbone.to(DEVICE)

    # load pretrained encoder weights (state dict with 'encoder_q_state' or raw state dict)
    ckpt = torch.load(PRETRAINED_ENCODER, map_location=DEVICE)
    if 'encoder_q_state' in ckpt:
        state = ckpt['encoder_q_state']
    elif 'model_state' in ckpt:
        # If saved full model, try to extract encoder
        state = {k.replace('encoder_q.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('encoder_q')}
    else:
        state = ckpt

    # Load state from backbone
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print('Loaded pretrained encoder. missing keys:', missing, 'unexpected:', unexpected)

    run_finetune_training()