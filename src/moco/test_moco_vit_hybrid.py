import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import print_and_log
from tqdm import tqdm
import argparse

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ViT Hybrid
from VIT_update_hybrid.model_builder import ViTMoCo as MoCo_ViT_Hybrid
from dataset_loader import get_moco_medical_loader

from VIT_update_hybrid.vit_backbone_wrapper import ViTBackbone

from classification_dataset import get_classification_data_loader

from utils import set_seed, print_and_log, save_stats

# Evaluate function for linear evaluation
def evaluate(backbone, linear_head, data_loader, device='cuda'):
    backbone.eval()
    linear_head.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            features = backbone(images)
            logits = linear_head(features)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def write_cm_to_file(cm, file_path, log_file=None, dataset_title='Pneumonia Classification'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: Finetune ViT Hybrid {dataset_title}')
    plt.savefig(file_path)
    plt.close()
    if log_file:
        print_and_log(f"Saved confusion matrix to {file_path}", log_file)
    else:
        print(f"Saved confusion matrix to {file_path}")
    return

def run_testing(backbone, linear_head, test_loader, device, dt, log_file, artifact_root='./', dataset_title='Pneumonia Classification'):
    # Run sklearn classification report
    backbone.eval()
    linear_head.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = backbone(images)
            logits = linear_head(features)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    report = classification_report(all_labels, all_preds)
    print_and_log("Classification Report:\n" + report, log_file=log_file)

    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print_and_log("Confusion Matrix:\n" + str(cm), log_file=log_file)

    # Write confusion matrix seaborn heatmap to file
    file_path = os.path.join(artifact_root, f'confusion_matrix_{dt}.png')
    write_cm_to_file(cm, file_path, log_file=log_file, dataset_title=dataset_title)
    return


def run_linear_evaluation(backbone, linear_head, train_loader, test_loader, epochs=20, device='cuda', log_file=None):
    # For linear evaluation, only train the linear head
    # Setup criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        linear_head.parameters(), 
        lr=0.1, momentum=0.9, weight_decay=0.0001
    )

    # Set backbone to eval mode - frozen already
    backbone.eval()

    # Save testing stats
    n_epochs_no_improve = 0
    test_stats = {}

    print_and_log(f"Starting linear evaluation for {epochs} epochs...", log_file=log_file)
    for epoch in range(epochs):
        linear_head.train()

        # Loop on the training data - using tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs - 1}')

        running_loss = 0.0

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass- backbone is frozen
            with torch.no_grad():
                features = backbone(images)
            
            logits = linear_head(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss (minibatch)
            running_loss += loss.item()

            # Update progress bar
            loop.set_postfix({'loss': loss.item()})

        # Get training accuracy
        train_acc = evaluate(backbone, linear_head, train_loader, device=device)

        # evaluate on test dataset after each epoch
        acc = evaluate(backbone, linear_head, test_loader, device=device)
        avg_running_loss = running_loss / len(train_loader)
        print_and_log(f"Epoch {epoch+1}/{epochs}, Avg. Running Loss: {avg_running_loss:.6f}', Train Acc:{train_acc:.2f}% Test Acc: {acc:.2f}%", log_file=log_file)

        # Early stopping if no improvement over 5 previous epochs
        if epoch > 0 and acc <= test_stats[epoch - 1]:
            n_epochs_no_improve += 1
            if n_epochs_no_improve >= 5:
                print_and_log("No improvement in test accuracy for 5 consecutive epochs. Stopping early.", log_file=log_file)
                break
        else:
            n_epochs_no_improve = 0

        # Save test accuracy for this epoch
        test_stats[epoch] = acc
        
    return test_stats, backbone


# ================================================================================
# Test MoCo backbone
# ================================================================================
def test_moco_backbone(model, train_loader, test_loader, linear_n_epochs=20, device='cuda', num_classes=2, log_file=None, artifact_root='./'):
    print("Extracting encoder_q from ViTMoCo model...")

    print_and_log("Model Keys: " + str(model.state_dict().keys()), log_file=log_file)

    # Load backbone using ViTBackbone wrapper
    backbone = ViTBackbone(model.encoder_q).to(device)   # ModuleDict
    num_features = model.embed_dim  # 384 or 192

    # Freeze backbone for linear eval
    for param in backbone.parameters():
        param.requires_grad = False

    classifier_head = torch.nn.Linear(num_features, num_classes)

    # Move to device
    backbone.to(device)
    classifier_head.to(device)

    # Run linear evaluation
    test_stats, backbone = run_linear_evaluation(
        backbone,
        classifier_head,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=linear_n_epochs,
        device=device,
        log_file=log_file
    )

    # confusion matrix, report, etc.
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    run_testing(backbone, classifier_head, test_loader, device, dt, log_file, artifact_root=artifact_root)
    
    return test_stats


def run_moco_testing(model, train_loader, test_loader, linear_n_epochs=20, device='cuda', num_classes=2, log_file="./artifacts/testing_log.txt", artifact_root='./'):
    print_and_log("Starting MoCo backbone testing...", log_file=log_file)
    test_stats = test_moco_backbone(model, train_loader, test_loader, linear_n_epochs, device=device, num_classes=num_classes, log_file=log_file, artifact_root=artifact_root)
    print_and_log("MoCo backbone testing complete!", log_file=log_file)

    return test_stats


# ================================================================================
# Main function to run testing only
# ================================================================================
def main(args):
    set_seed(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    test_log_file = os.path.join(args.artifact_root, f'moco_backbone_testing_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", test_log_file)

    # Get test loader
    print_and_log("Starting MoCo backbone testing...", log_file=test_log_file)

    # ----------------------------------------------------
    # Model Setup 
    # ----------------------------------------------------
    # build model and load pretrained encoder
    model = MoCo_ViT_Hybrid(proj_dim=128, K=65536, m=0.99, T=0.2, embed_dim=192, device=device)
    model.to(device)

    # Load MoCo model from checkpoint
    model_checkpoint_path = os.path.join(args.artifact_root, args.model_checkpoint)
    print_and_log(f"Loading pretrained MoCo model from {model_checkpoint_path}...", log_file=test_log_file)

    # Load in the saved MoCo model
    ckpt = torch.load(model_checkpoint_path, map_location=device)

    # For MoCo: Only use the encoder for the backbone
    if 'encoder_q_state' in ckpt:
        print_and_log("Detected 'encoder_q_state' in checkpoint.", test_log_file)
        state = ckpt['encoder_q_state']
    elif 'model_state' in ckpt:
        print_and_log("Detected 'model_state' in checkpoint. Extracting encoder_q weights.", test_log_file)
        # If saved full model, try to extract encoder
        state = {k.replace('encoder_q.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('encoder_q')}
    else:
        state = ckpt
    
    # Load state from backbone
    missing, unexpected = model.load_state_dict(state, strict=False)
    print_and_log(f'Loaded pretrained encoder. missing keys: {missing}, unexpected: {unexpected}', log_file=test_log_file)

    # Get training and testing loaders - for linear evaluation
    # Labeled dataset for linear evaluation
    linear_train_loader = get_classification_data_loader(
        data_split_type='train',
        CSV_PATH=args.linear_train_csv_path,
        ROOT_DIR=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col
    )
    linear_test_loader = get_classification_data_loader(
        data_split_type='test',
        CSV_PATH=args.linear_test_csv_path,
        ROOT_DIR=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_col=args.label_col
    )

    # Run testing
    test_stats = run_moco_testing(
        model,
        linear_train_loader,
        linear_test_loader,
        device=device,
        linear_n_epochs=args.linear_n_epochs,
        num_classes=args.test_num_classes,
        log_file=test_log_file,
        artifact_root=args.artifact_root
    )
    print_and_log("MoCo backbone testing complete!!", log_file=test_log_file)

    # Save testing stats
    save_stats(test_stats, args.artifact_root + '/test_stats.json', 'testing', log_file=test_log_file)
    return

# ================================================================================
# Entry point / Argument parsing
# ================================================================================
if __name__ == "__main__":
    # Argument parser for CLI configuration
    parser = argparse.ArgumentParser(description="MoCo Medical Encoder Training")
    # Should be set
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory for images')
    parser.add_argument('--artifact_root', type=str, default='./artifacts/', help='Directory for checkpoints')
    parser.add_argument('--model_checkpoint', type=str, default='moco_resnet50_encoder.pth', help='Model checkpoint filename to load; expected in the artifact_root directory')

    # For testing
    parser.add_argument('--test_num_classes', type=int, default=2, help='Number of classes for testing classification')
    parser.add_argument('--linear_n_epochs', type=int, default=30, help='Number of epochs for test linear classification training')
    parser.add_argument('--linear_train_csv_path', type=str, default='linear_train.csv', help='Train CSV file with image paths')
    parser.add_argument('--linear_test_csv_path', type=str, default='linear_test.csv', help='Test CSV file with image paths')
    
    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--label_col', type=str, default='Pneumonia', help='Label column name in CSV for classification dataset')

    # Hyperparameters that can be tuned
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    main(args)