import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

from finetune.classification_dataset import get_train_val_loaders, get_test_loader

from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import argparse
import pickle

def print_and_log(message, log_file=None):
    # Print to console with datetime
    printedatetime = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{printedatetime}] {message}')

    # Log to file if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'[{printedatetime}] {message}\n')
    return


def run_validation(backbone, val_loader, device, criterion):
    backbone.eval()
    correct = 0
    total = 0
    val_loss_sum = 0.0
    val_batches = 0

    # Disable gradient calculation for validation
    with torch.no_grad():
        # Loop over validation data
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = backbone(images)
            preds = torch.argmax(logits, dim=1)
            loss = criterion(logits, labels)
            val_loss_sum += loss.item()
            val_batches += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy and average loss
    val_acc = correct / total if total > 0 else 0.0
    val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

    return val_acc, val_loss

def run_finetune_training(backbone, train_loader, val_loader, device, lr, n_epochs, log_file=None, weight_decay=1e-5):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": backbone.layer3.parameters(), "lr": 5e-6},
        {"params": backbone.layer4.parameters(), "lr": 5e-6},
        {"params": backbone.fc.parameters(), "lr": 1e-3},
    ], weight_decay=weight_decay)

    # Accumulate training stats
    train_stats = []

    # Training loop
    for epoch in range(n_epochs):
        backbone.train()

        # --- Run training ---
        train_correct = 0
        train_total = 0
        loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch}')
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            logits = backbone(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix({'loss': loss.item()})

            # Training accuracy calculation
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- Run validation ---
        val_acc, val_loss = run_validation(backbone, val_loader, device, criterion)

        # Update training stats
        train_stats.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'train_loss': loss.item(), 'val_loss': val_loss})
        print_and_log(f'Epoch {epoch}: train_acc = {train_acc:.4f} | val_acc = {val_acc:.4f}', log_file)
        print_and_log(f'Epoch {epoch}: train_loss = {loss.item():.4f} | val_loss = {val_loss:.4f}', log_file)

        # Save intermediate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'finetuned_model_epoch_{epoch+1}.pth'
            torch.save({'model_state': backbone.state_dict()}, checkpoint_path)
            print_and_log(f'Saved checkpoint: {checkpoint_path}', log_file)

    print_and_log("Finetuning complete.", log_file)

    # Save finetuned model
    torch.save({'model_state': backbone.state_dict()}, 'finetuned_model.pth')
    print_and_log('Saved finetuned_model.pth', log_file)
    return backbone, train_stats

# ======================= Main Function ==========================
def main(args):
    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'training_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # Load train and validation loaders
    train_loader, val_loader = get_train_val_loaders(
        args.train_csv,
        args.val_csv,
        args.root_dir,
        args.batch_size,
        args.num_workers
    )

    # build model and load pretrained encoder
    try:
        from torchvision.models import ResNet50_Weights
        backbone = resnet50(weights=None)
    except Exception:
        backbone = resnet50(pretrained=False)

    # Modify final layer for NUM_CLASSES
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, args.num_classes)
    backbone.to(args.device)

    # Load pretrained encoder weights (state dict with 'encoder_q_state' or raw state dict)
    ckpt = torch.load(args.pretrained_encoder, map_location=args.device)
    print_and_log(f"Loading pretrained encoder from: {args.pretrained_encoder}", log_file)
    if 'encoder_q_state' in ckpt:
        state = ckpt['encoder_q_state']
    elif 'model_state' in ckpt:
        # If saved full model, try to extract encoder
        state = {k.replace('encoder_q.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('encoder_q')}
    else:
        state = ckpt

    # Load state from backbone
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print_and_log(f'Loaded pretrained encoder. missing keys: {missing}, unexpected: {unexpected}', log_file)

    # Freeze backbone layers - 70/80% of layers frozen
    for name, param in backbone.named_parameters():
        if not ("layer3" in name or "layer4" in name or "fc" in name):
            param.requires_grad = False

    # Run finetuning
    print_and_log("Starting finetuning...", log_file)
    backbone, train_stats = run_finetune_training(backbone, train_loader, val_loader, args.device, args.lr, args.n_epochs, log_file=log_file)
    print_and_log("Finetuning complete.", log_file)

    # Save training stats to pickle
    stats_path = os.path.join(args.artifact_root, 'finetune_training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(train_stats, f)

    # Run testing evaluation
    test_loader = get_test_loader(
        TEST_CSV=args.test_csv,
        ROOT_DIR=args.root_dir,
        BATCH_SIZE=args.batch_size,
        NUM_WORKERS=args.num_workers
    )

    # Evaluate on test set
    test_acc, _ = run_validation(backbone, test_loader, args.device, nn.CrossEntropyLoss())
    print_and_log(f'Test Accuracy: {test_acc:.4f}', log_file)

# ======================= Arg Parse ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune ResNet on custom dataset")
    # Should be configured
    parser.add_argument('--train_csv', type=str, default='final_project_updated_names_train.csv', help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, default='final_project_updated_names_val.csv', help='Path to validation CSV')
    parser.add_argument('--test_csv', type=str, default='final_project_updated_names_test.csv', help='Path to test CSV')
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory of images')
    parser.add_argument('--pretrained_encoder', type=str, default='moco_resnet50_encoder.pth', help='Path to pretrained encoder weights')
    parser.add_argument('--artifact_root', type=str, default='./finetune_artifacts', help='Directory to save artifacts')

    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    
    args = parser.parse_args()

    main(args)