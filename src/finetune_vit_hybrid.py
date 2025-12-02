import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

from finetune.classification_dataset import get_classification_data_loader

from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import argparse
import pickle

from copy import deepcopy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ViT Hybrid
from moco.VIT_update_hybrid.model_builder import ViTMoCo as MoCo_ViT_Hybrid

# ViT Wrappers
from moco.VIT_update_hybrid.vit_backbone_wrapper import FT_ViTBackbone, FT_ViT_FinetuneModel

# Utility functions
def print_and_log(message, log_file=None):
    # Print to console with datetime
    printedatetime = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{printedatetime}] {message}')

    # Log to file if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'[{printedatetime}] {message}\n')
    return

def write_cm_to_file(cm, file_path, log_file=None, dataset_title='Pneumonia Classification'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: Finetune ResNet {dataset_title}')
    plt.savefig(file_path)
    plt.close()
    if log_file:
        print_and_log(f"Saved confusion matrix to {file_path}", log_file)
    else:
        print(f"Saved confusion matrix to {file_path}")
    return


# Training function
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, n_epochs):
    # Set model to train mode
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Loop on the training data - using tqdm for progress bar
    loop = tqdm(loader, desc=f'Epoch {epoch}/{n_epochs - 1}')

    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # optional: gradient clipping if needed
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        # Update progress bar
        loop.set_postfix(loss=loss.item(), accuracy=correct/total)

    # Return average loss and accuracy
    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# Validation function
def validate(model, loader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss = criterion(logits, y)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            # live display of running stats
            loop.set_postfix({
                "loss": f"{val_loss / (total/len(y)):.4f}",
                "acc": f"{correct / total:.4f}"
            })

    # Return average loss and accuracy
    avg_loss = val_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def run_testing(model, test_loader, device, criterion, dt, log_file,
                artifact_root='./', dataset_title='Pneumonia Classification'):
    # Compute model accuracy using validate()
    test_avg_loss, test_acc = validate(model, test_loader, device, criterion)
    print_and_log(f"Test Accuracy: {test_acc:.4f}", log_file)

    # Collect all predictions & labels for sklearn metrics
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds)
    print_and_log("Classification Report:\n" + report, log_file)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print_and_log("Confusion Matrix:\n" + str(cm), log_file)

    # Save confusion matrix figure
    file_path = os.path.join(artifact_root, f'confusion_matrix_{dt}.png')
    write_cm_to_file(cm, file_path, log_file, dataset_title)

    return all_labels, all_preds

# Finetuning training function
def run_finetune_training(model, train_loader, val_loader, device, lr, n_epochs, log_file=None, 
                          n_epochs_stop=5, artifact_root='./', subtitle="",
                          gradient_clip_value=True, max_norm=1):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])

    # Accumulate training stats
    train_stats = []
    best_val_acc = 0.0
    last_val_acc = 0.0
    n_epochs_no_improve = 0
    best_model = None

    print_and_log("Beginning finetuning training...", log_file)

    # Training loop
    for epoch in range(n_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, n_epochs)
        val_loss, val_acc = validate(model, val_loader, device, criterion)

        print_and_log(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}", log_file)
        print_and_log(f"           train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", log_file)

        # Track stats
        train_stats.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)
            n_epochs_no_improve = 0
            print_and_log(f"New best model found at epoch {epoch} with val_acc={val_acc:.4f}", log_file)
            model_path = os.path.join(artifact_root, f'{subtitle}_finetuning_epoch_{epoch}_model.pth')
            torch.save({'model_state': best_model.state_dict()}, model_path)

        # Early stopping check
        if val_acc <= last_val_acc:
            n_epochs_no_improve += 1
            if n_epochs_no_improve >= n_epochs_stop:
                print_and_log(f"No improvement for {n_epochs_no_improve} epochs. Early stopping at epoch {epoch}.", log_file)
                break

    print_and_log("Finetuning complete.", log_file)

    # Save finetuned model
    model_path = os.path.join(artifact_root, f'{subtitle}_finetuned_model.pth')
    torch.save({'model_state': best_model.state_dict()}, model_path)
    print_and_log(f'Saved {model_path}', log_file)
    return best_model, train_stats


# Helper Functions for fine tuning
def freeze_vit_layers(backbone, fraction=0.7):
    transformer = backbone.transformer

    print("[freeze] Inspecting transformer:", type(transformer))

    # Case 1: torchvision/timm ViT — transformer.blocks is a list of blocks
    if hasattr(transformer, "blocks"):
        maybe_blocks = transformer.blocks

        # If blocks is actually a TransformerEncoder, check inside it
        if isinstance(maybe_blocks, nn.TransformerEncoder):
            print("[freeze] Found .blocks but it is a TransformerEncoder; using .blocks.layers")
            blocks = maybe_blocks.layers  # ModuleList
        else:
            print("[freeze] Using .blocks directly")
            blocks = maybe_blocks

    # Case 2: PyTorch TransformerEncoder — uses .layers
    elif hasattr(transformer, "layers"):
        print("[freeze] Using .layers")
        blocks = transformer.layers

    else:
        raise AttributeError("Transformer has neither .blocks nor .layers")

    # Now blocks MUST be a list/ModuleList
    num_blocks = len(blocks)
    cutoff = int(num_blocks * fraction)

    print(f"[freeze] Total blocks: {num_blocks}, freezing bottom: {cutoff}")

    # Freeze patch embed
    for p in backbone.patch_embed.parameters():
        p.requires_grad = False

    # Freeze layers
    for i, blk in enumerate(blocks):
        if i < cutoff:
            for p in blk.parameters():
                p.requires_grad = False

    print("[freeze] Freeze complete.")

# ======================= Main Function ==========================
def main(args):
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'finetune_ViT_hybrid_training_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # ----------------------------------------------------
    # Dataset Loaders
    # ----------------------------------------------------
    # Load train and validation loaders
    train_loader = get_classification_data_loader(
        data_split_type='train',
        CSV_PATH=args.train_csv,
        ROOT_DIR=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col
    )
    
    val_loader = get_classification_data_loader(
        data_split_type='val',
        CSV_PATH=args.val_csv,
        ROOT_DIR=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col
    )

    # ----------------------------------------------------
    # Model Setup 
    # ----------------------------------------------------
    # build model and load pretrained encoder
    model = MoCo_ViT_Hybrid(proj_dim=128, K=65536, m=0.99, T=0.2, embed_dim=192, device=device)
    model.to(device)

    # Load MoCo model from checkpoint
    model_checkpoint_path = os.path.join(args.artifact_root, args.pretrained_encoder)
    print_and_log(f"Loading pretrained MoCo model from {model_checkpoint_path}...", log_file=log_file)

    # Load in the saved MoCo model
    ckpt = torch.load(model_checkpoint_path, map_location=device)

    # -----------------------------------------------
    # Backbone Setup
    # -----------------------------------------------

    # For MoCo: Only use the encoder for the backbone
    if 'encoder_q_state' in ckpt:
        print_and_log("Detected 'encoder_q_state' in checkpoint.", log_file)
        state = ckpt['encoder_q_state']
    elif 'model_state' in ckpt:
        print_and_log("Detected 'model_state' in checkpoint. Extracting encoder_q weights.", log_file)
        # If saved full model, try to extract encoder
        state = {k.replace('encoder_q.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('encoder_q')}
    else:
        state = ckpt

    # Load state from backbone
    missing, unexpected = model.load_state_dict(state, strict=False)
    print_and_log(f'Loaded pretrained encoder. missing keys: {missing}, unexpected: {unexpected}', log_file)

    # Build backbone
    print_and_log("Building ViT Hybrid backbone for finetuning...", log_file)
    # Embedding dim *must* match pretrained model
    backbone = FT_ViTBackbone(model.encoder_q, embed_dim=192).to(device)

    # Freeze portion of backbone layers for finetuning
    print_and_log("Freezing 70% of ViT layers in backbone for finetuning...", log_file)
    freeze_vit_layers(backbone, fraction=0.7)

    # Build final model
    print_and_log("Building finetune model...", log_file)
    model = FT_ViT_FinetuneModel(backbone, num_classes=args.num_classes).to(device)

    # ----------------------------------------------------
    # Finetuning
    # ----------------------------------------------------
    # Run finetuning
    print_and_log("Starting finetuning...", log_file)
    model, train_stats = run_finetune_training(model, train_loader, val_loader, device, args.lr, args.n_epochs, log_file=log_file, artifact_root=args.artifact_root, subtitle=args.subtitle)
    print_and_log("Finetuning complete.", log_file)

    # Save training stats to pickle
    stats_path = os.path.join(args.artifact_root, 'hybrid_finetune_training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(train_stats, f)

    # ----------------------------------------------------
    # Run Testing evaluation 
    # ----------------------------------------------------
    # Get test dataset loader
    test_loader = get_classification_data_loader(
        data_split_type='test',
        CSV_PATH=args.test_csv,
        ROOT_DIR=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col
    )

    # Run testing
    print_and_log("Starting testing evaluation...", log_file)
    run_testing(backbone=backbone,
                test_loader=test_loader,
                device=device,
                criterion=nn.CrossEntropyLoss(),
                dt=dt, 
                log_file=log_file,
                artifact_root=args.artifact_root,
                dataset_title=args.dataset_title)
    
    return

# ================================================================================
# Entry point / Argument parsing
# ================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ViT Hybrid on custom dataset")
    # Should be configured
    parser.add_argument('--train_csv', type=str, default='final_project_updated_names_train.csv', help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='final_project_updated_names_val.csv', help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='final_project_updated_names_test.csv', help='Path to test CSV')
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory of images')
    parser.add_argument('--pretrained_encoder', type=str, default='vit_hybrid_moco_encoder.pth', help='Path to pretrained encoder weights')
    parser.add_argument('--artifact_root', type=str, default='./finetune_artifacts', help='Directory to save artifacts')
    
    # Dataset
    parser.add_argument('--label_col', type=str, default='Pneumonia', help='Name of the label column in the dataset')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--dataset_title', type=str, default='Pneumonia Classification', help='Title of the dataset for logging')
    parser.add_argument('--subtitle', type=str, default='vit_hybrid', help='Subtitle for saved models and logs')

    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    
    args = parser.parse_args()

    main(args)