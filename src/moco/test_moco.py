import os
import torch
from torch.utils.data import DataLoader
from Git_Project.src.moco.resnet50_baseline.model_builder import MoCo

from utils import print_and_log
from tqdm import tqdm

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.title(f'Confusion Matrix: Finetune ResNet {dataset_title}')
    plt.savefig(file_path)
    plt.close()
    if log_file:
        print_and_log(f"Saved confusion matrix to {file_path}", log_file)
    else:
        print(f"Saved confusion matrix to {file_path}")
    return

def run_testing(backbone, test_loader, device, dt, log_file, artifact_root='./', dataset_title='Pneumonia Classification'):
    # Run sklearn classification report
    backbone.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = backbone(images)
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
        for images, labels in train_loader:
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

        # Get training accuracy
        train_acc = evaluate(backbone, linear_head, train_loader, device=device)

        # evaluate on test dataset after each epoch
        acc = evaluate(backbone, linear_head, test_loader, device=device)
        print_and_log(f"Epoch {epoch+1}/{epochs}, Train Acc:{train_acc:.2f}% Test Acc: {acc:.2f}%", log_file=log_file)

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
    # 1) Extract the backbone encoder_q
    if hasattr(model, 'encoder_q'):
        print("Extracting encoder_q from MoCo model for linear evaluation...")
        backbone = model.encoder_q
    else:
        backbone = model

    # Remove the projection head (MLP)
    num_features = 2048
    backbone.fc = torch.nn.Identity()

    # 2) Freeze the pretrained backbone for linear evaluation
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 3) Create a linear classifier on top of the frozen backbone
    # Based on the number of classes in the dataset (e.g., 2 for binary classification) - Input Argument
    classifier_head = torch.nn.Linear(num_features, num_classes)

    # Move to device
    backbone.to(device)
    classifier_head.to(device)

    # 4) Run linear evaluation
    # Train only the linear layer on the labeled dataset
    # Measure the accuracy on the test set after training
    test_stats, backbone = run_linear_evaluation(backbone, classifier_head, train_loader=train_loader, test_loader=test_loader, epochs=linear_n_epochs, device=device, log_file=log_file)

    # 5) Save confusion matrix and classification report
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    run_testing(backbone, test_loader, device, dt, log_file, artifact_root=artifact_root, dataset_title='Pneumonia Classification')

    return test_stats


def run_moco_testing(model, train_loader, test_loader, linear_n_epochs=20, device='cuda', num_classes=2, log_file="./artifacts/testing_log.txt", artifact_root='./'):
    print_and_log("Starting MoCo backbone testing...", log_file=log_file)
    test_stats = test_moco_backbone(model, train_loader, test_loader, linear_n_epochs, device=device, num_classes=num_classes, log_file=log_file, artifact_root=artifact_root)
    print_and_log("MoCo backbone testing complete!", log_file=log_file)

    return test_stats