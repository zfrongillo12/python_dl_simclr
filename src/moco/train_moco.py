import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import os
import json

# ResNet50
from resnet50_baseline.model_builder import MoCo
from dataset_loader import get_moco_medical_loader

# ViT
from VIT_baseline.model_builder import MoCo as MoCo_ViT
from VIT_baseline.dataset_loader import get_moco_medical_loader as get_moco_medical_loader_vit

from classification_dataset import get_classification_data_loader

from utils import set_seed, save_state, print_and_log
from test_moco import run_moco_testing


def run_moco_training(model,
                      n_epochs,
                      train_loader,
                      optimizer,
                      warmup_epochs,
                      scheduler_cos, 
                      artifact_root="./artifacts/",
                      base_lr=0.03,
                      log_file="./artifacts/training_log.txt"):
    # Start training log
    print_and_log(f"Starting MoCo training for {n_epochs} epochs", log_file=log_file)

    # Accumulate training stats for pretraining
    train_stats = []

    for epoch in range(n_epochs):
        # Set model to train mode
        model.train()
        # Loop on the training data - using tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs - 1}')

        # Training loop - for each batch
        for im_q, im_k in loop:
            im_q = im_q.to(model.device, non_blocking=True)
            im_k = im_k.to(model.device, non_blocking=True)

            logits, labels, keys = model(im_q, im_k)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the queue
            model.dequeue_and_enqueue(keys)

            # Update progress bar
            loop.set_postfix({'loss': loss.item()})

            # Print loss with datetime
            # print_and_log(f'Epoch {epoch}, Loss: {loss.item():.4f}', log_file=log_file)

        # Scheduler step: simple warmup handling
        if epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale
        else:
            scheduler_cos.step()

        # Print loss with datetime
        print_and_log(f'Epoch complete {epoch}, Loss: {loss.item():.4f}', log_file=log_file)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_checkpoint_path = f'{artifact_root}/moco_checkpoint_epoch_{epoch+1}.pth'
            print_and_log(f'Saving checkpoint at epoch {epoch+1}; {model_checkpoint_path}', log_file=log_file)
            os.makedirs(artifact_root, exist_ok=True)
            save_state(model_checkpoint_path, model, optimizer, epoch)

        # End of epoch: save training stats
        train_stats.append({'epoch': epoch, 'loss': loss.item()})

    return model, train_stats

def save_stats(stats, path, stat_type, log_file=None):
    with open(path, 'w') as f:
        json.dump(stats, f)
    print_and_log(f"Saved {stat_type} stats to {path}", log_file=log_file)
    return

# ================================================================================
# Main function to parse arguments and run training + testing
# ================================================================================
def main(args):
    set_seed(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'moco_training_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # ---------------------------------------
    # Get training loader
    # ---------------------------------------
    print_and_log(f"Creating MoCo medical image Train DataLoader... : from {args.train_csv_path}", log_file=log_file)
    # Unlabeled dataset for MoCo pretraining
    if args.model_type == 'VIT':
        train_loader = get_moco_medical_loader_vit(
            data_split_type='train',
            csv_path=args.train_csv_path,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        train_loader = get_moco_medical_loader(
            data_split_type='train',
            csv_path=args.train_csv_path,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # ---------------------------------------
    # Model Setup - MoCo
    # ---------------------------------------
    if args.model_type == 'VIT':
        print_and_log("Using ViT backbone for MoCo", log_file=log_file)
        model = MoCo_ViT(dim=args.dim, K=args.K, m=args.m, T=args.T, pretrained=args.pretrained_imagenet, device=device)
        model.to(device)

        base_lr = 1e-4 # Starting point
        effective_lr = base_lr * (args.batch_size ** 0.5 / 256 ** 0.5)

        optimizer = torch.optim.AdamW(
            list(model.encoder_q.parameters()) + list(model.mlp_q.parameters()),
            lr=effective_lr,
            weight_decay=0.05,     # ViT default WD
            betas=(0.9, 0.999)     # standard
        )

        warmup_epochs = 30
        # Warmup (linear) for the first 25 epochs
        # Cosine annealing scheduler
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.n_epochs - warmup_epochs)
        )
    else:
        print_and_log("Using ResNet50 backbone for MoCo", log_file=log_file)
        # Use ResNet50 backbone is instantiated by MoCo
        # by default, no pretraining should be applied
        model = MoCo(dim=args.dim, K=args.K, m=args.m, T=args.T, pretrained=args.pretrained_imagenet, device=device)
        model.to(device)

        # LR scaled to batch size (Scaling Rule for MoCo): lr_base * (batch_size / 256)
        # LR Scale -> Base is 256; adjusted for smaller batch sizes
        base_lr = args.base_lr * (args.batch_size / 256)
        optimizer = optim.SGD(
            list(model.encoder_q.parameters()) + list(model.mlp_q.parameters()),
            lr=base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        # Warmup (linear) for the first 25 epochs
        # Cosine annealing scheduler
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.n_epochs - args.warmup_epochs)
        )

    # ---------------------------------------
    # Training Loop
    # ---------------------------------------

    # Run encoder training
    model, train_stats = run_moco_training(
        model,
        n_epochs=args.n_epochs,
        train_loader=train_loader,
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler_cos=scheduler_cos,
        artifact_root=args.artifact_root,
        base_lr=args.base_lr,
        log_file=log_file,
    )
    print_and_log('MoCo training complete!!', log_file=log_file)

    # Save the trained encoder (query backbone + mlp)
    # After training, save only 
    #   1) the encoder_q (used as the backbone for transfer learning)
    #   2) mlp_q (for further pre-training; but is generally not used for transfer learning)

    # Create file name for saving model - use args.out_model_name
    model_output_path = os.path.join(args.artifact_root, args.out_model_name)

    # Save encoder_q and mlp_q state dicts
    torch.save({
        'encoder_q_state': model.encoder_q.state_dict(),
        'mlp_q_state': model.mlp_q.state_dict()
    }, model_output_path)
    print_and_log(f"Saved pretrained encoder to {model_output_path}", log_file=log_file)

    # Save training stats
    save_stats(train_stats, args.artifact_root + '/pretrain_stats_training.json', 'training', log_file=log_file)

    # ---------------------------------------
    # Run Testing
    # ---------------------------------------
    # Get test loader
    print_and_log("Starting MoCo backbone testing...", log_file=log_file)

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
    test_log_file = os.path.join(args.artifact_root, f'moco_testing_log_{dt}.txt')
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
    print_and_log("MoCo backbone testing complete!!", log_file=log_file)

    # Save testing stats
    save_stats(test_stats, args.artifact_root + '/test_stats.json', 'testing', log_file=log_file)

    return


# ================================================================================
# Entry point / Argument parsing
# ================================================================================
if __name__ == "__main__":
    # Argument parser for CLI configuration
    parser = argparse.ArgumentParser(description="MoCo Medical Encoder Training")
    # Should be set
    parser.add_argument('--train_csv_path', type=str, default='train.csv', help='Train CSV file with image paths')
    parser.add_argument('--test_csv_path', type=str, default='test.csv', help='Test CSV file with image paths')
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory for images')
    parser.add_argument('--artifact_root', type=str, default='./artifacts/', help='Directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='ResNet50', help='Model type for MoCo (ResNet50 or VIT)')

    # For testing
    parser.add_argument('--test_num_classes', type=int, default=2, help='Number of classes for testing classification')
    parser.add_argument('--linear_n_epochs', type=int, default=30, help='Number of epochs for test linear classification training')
    parser.add_argument('--linear_train_csv_path', type=str, default='linear_train.csv', help='Train CSV file with image paths')
    parser.add_argument('--linear_test_csv_path', type=str, default='linear_test.csv', help='Test CSV file with image paths')
    
    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs for backbone training')
    parser.add_argument('--out_model_name', type=str, default='moco_resnet50_encoder.pth', help='Output filename for encoder')
    parser.add_argument('--label_col', type=str, default='Pneumonia', help='Label column name in CSV for classification dataset')

    # Hyperparameters that can be tuned
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--K', type=int, default=65536, help='Queue size')
    parser.add_argument('--m', type=float, default=0.99, help='Momentum for updating key encoder')
    parser.add_argument('--T', type=float, default=0.2, help='Softmax temperature')
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pretrained_imagenet', type=bool, default=False, help='Use ImageNet pretrained weights')
    parser.add_argument('--base_lr', type=float, default=0.03, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=25, help='Number of warmup epochs')

    args = parser.parse_args()

    main(args)