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

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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


def run_testing(backbone, test_loader, device, criterion, dt, log_file, artifact_root='./', dataset_title='Pneumonia Classification'):
    # Run base accuracy eval
    test_acc, _ = run_validation(backbone, test_loader, device, criterion)
    print_and_log(f'Test Accuracy: {test_acc:.4f}', log_file)

    # Run sklearn classification report
    backbone.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)
            logits = backbone(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds)
    print_and_log("Classification Report:\n" + report, log_file)

    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print_and_log("Confusion Matrix:\n" + str(cm), log_file)

    # Write confusion matrix seaborn heatmap to file
    file_path = os.path.join(artifact_root, f'confusion_matrix_{dt}.png')
    write_cm_to_file(cm, file_path, log_file, dataset_title=dataset_title)

    return

def run_finetune_training(backbone, train_loader, val_loader, device, lr, n_epochs, log_file=None, weight_decay=1e-5, n_epochs_stop=4, artifact_root='./', subtitle=""):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {"params": backbone.layer3.parameters(), "lr": 5e-4},
        {"params": backbone.layer4.parameters(), "lr": 5e-4},
        {"params": backbone.fc.parameters(),    "lr": 1e-2},
    ], momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Accumulate training stats
    train_stats = []

    # Training loop
    for epoch in range(n_epochs):
        backbone.train()

        # --- Run training ---
        epoch_loss = 0.0
        num_batches = 0

        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch}')
        
        for images, labels in loop:
            # Transfer to device
            images = images.to(device)
            labels = labels.to(device)

            logits = backbone(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

            # Training accuracy calculation
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = epoch_loss / num_batches
        train_acc = train_correct / train_total

        # --- Run validation ---
        val_acc, val_loss = run_validation(backbone, val_loader, device, criterion)

        # Update training stats
        train_stats.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        })
        print_and_log(f'Epoch {epoch}: train_acc = {train_acc:.4f} | val_acc = {val_acc:.4f}', log_file)
        print_and_log(f'Epoch {epoch}: train_loss = {avg_train_loss:.4f} | val_loss = {val_loss:.4f}', log_file)

        # --- Early stopping ---
        if len(train_stats) > n_epochs_stop:
            recent = [s['val_acc'] for s in train_stats[-(n_epochs_stop+1):]]
            if all(recent[i] <= recent[i-1] for i in range(1, len(recent))):
                break
        
        # Step the scheduler
        scheduler.step()

        # Save intermediate model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(artifact_root, f'{subtitle}_finetuned_model_epoch_{epoch+1}.pth')
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
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print_and_log(f'Loaded pretrained encoder. missing keys: {missing}, unexpected: {unexpected}', log_file)

    # Freeze backbone layers - 70/80% of layers frozen
    for name, param in backbone.named_parameters():
        if not ("layer3" in name or "layer4" in name or "fc" in name):
            param.requires_grad = False

    # ----------------------------------------------------
    # Finetuning
    # ----------------------------------------------------
    # Run finetuning
    print_and_log("Starting finetuning...", log_file)
    backbone, train_stats = run_finetune_training(backbone, train_loader, val_loader, args.device, args.lr, args.n_epochs, log_file=log_file, artifact_root=args.artifact_root, subtitle=args.subtitle)
    print_and_log("Finetuning complete.", log_file)

    # Save training stats to pickle
    stats_path = os.path.join(args.artifact_root, 'finetune_training_stats.pkl')
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
                device=args.device,
                criterion=nn.CrossEntropyLoss(),
                dt=dt, 
                log_file=log_file,
                artifact_root=args.artifact_root,
                dataset_title=args.dataset_title)
    
    return

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
    
    # Dataset
    parser.add_argument('--label_col', type=str, default='Pneumonia', help='Name of the label column in the dataset')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument('--dataset_title', type=str, default='Pneumonia Classification', help='Title of the dataset for logging')
    parser.add_argument('--subtitle', type=str, default='resnet50', help='Subtitle for saved models and logs')

    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs')
    
    args = parser.parse_args()

    main(args)