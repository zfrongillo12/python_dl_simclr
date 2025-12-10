import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import os
import json

# ViT Hybrid
from VIT_update_hybrid.model_builder import ViTMoCo as MoCo_ViT_Hybrid
# from dataset_loader import get_moco_medical_loader

from VIT_baseline.dataset_loader import get_moco_medical_loader as get_moco_medical_loader_vit

from test_moco_vit_hybrid import run_moco_testing

from utils import set_seed, save_state, print_and_log, save_stats

# ================================================================================
# For ViT Hybrid [Case 2], train function is slightly modified to accommodate different model structure
# ================================================================================
def run_moco_training_vit_hybrid(model,
                      n_epochs,
                      train_loader,
                      optimizer,
                      warmup_epochs,
                      scheduler_cos,
                      q_params,
                      artifact_root="./artifacts/",
                      base_lr=0.03,
                      log_file="./artifacts/training_log.txt"):
    # Start training log
    print_and_log(f"Starting MoCo ViT Hybrid training for {n_epochs} epochs...", log_file=log_file)

    # Accumulate training stats for pretraining
    train_stats = []

    best_loss = 100 # initialize best loss to a large value

    for epoch in range(n_epochs):
        # Set model to train mode
        model.train()
        # Loop on the training data - using tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs - 1}')

        running_loss = 0.0

        # Training loop - for each batch (loop represents one batch from train_loader)
        for im_q, im_k in loop:
            im_q = im_q.to(model.device, non_blocking=True)
            im_k = im_k.to(model.device, non_blocking=True)

            # Forward pass
            logits, labels = model(im_q, im_k)
            loss = F.cross_entropy(logits, labels) # compute scalar loss

            optimizer.zero_grad()
            loss.backward()

            # optional: gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(q_params, max_norm=3.0)
            optimizer.step()

            # Update loss (minibatch)
            running_loss += loss.item()

            # Queue is updated internally in model forward for ViT Hybrid

            # Update progress bar
            loop.set_postfix({'loss': loss.item()})

        # Scheduler step: simple warmup handling
        if epoch < warmup_epochs:
            lr_scale = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_scale
        else:
            scheduler_cos.step()

        # Print loss with datetime
        avg_running_loss = running_loss / len(train_loader)
        print_and_log(f'Epoch complete {epoch}, Avg. Running Loss: {avg_running_loss:.6f}', log_file=log_file)

        # Save checkpoint every time the loss improves, or for the first epoch, or at regular intervals
        if avg_running_loss < best_loss or epoch == 0 or (epoch + 1) % 5 == 0:
            if epoch != 0:
                print_and_log(f'Loss improved from {best_loss:.6f} to {avg_running_loss:.6f}. Saving checkpoint.', log_file=log_file)
            elif epoch == 0:
                print_and_log(f'Saving initial checkpoint at epoch {epoch+1}; Avg. Running Loss: {avg_running_loss:.6f}; Last Loss: {loss.item():.6f}', log_file=log_file)
            else:
                print_and_log(f'Saving periodic checkpoint at epoch {epoch+1}; Avg. Running Loss: {avg_running_loss:.6f}; Last Loss: {loss.item():.6f}', log_file=log_file)

            # Update Loss
            best_loss = avg_running_loss

            # Save checkpoint
            model_checkpoint_path = f'{artifact_root}/vit_hybrid_moco_checkpoint_epoch_{epoch+1}.pth'
            print_and_log(f'Saving checkpoint at epoch {epoch+1}; {model_checkpoint_path}', log_file=log_file)
            os.makedirs(artifact_root, exist_ok=True)
            save_state(model_checkpoint_path, model, optimizer, epoch)

        # End of epoch: save training stats
        train_stats.append({'epoch': epoch, 'loss': avg_running_loss})

    return model, train_stats


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
    print_and_log(f"Creating MoCo medical image Train DataLoader (Hybrid ViT)... : from {args.train_csv_path}", log_file=log_file)
    # Unlabeled dataset for MoCo pretraining
    train_loader = get_moco_medical_loader_vit(
        data_split_type='train',
        csv_path=args.train_csv_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ---------------------------------------
    # Model Setup - MoCo
    # ---------------------------------------
    q_params = None

    # ---------------------------------------
    # Case 2: ViT Hybrid Backbone - UPDATED
    # ---------------------------------------
    print_and_log("Using ViT Hybrid backbone for MoCo", log_file=log_file)
    print_and_log("Note: No option to use pretrained weights; ignoring if set", log_file=log_file)
    model = MoCo_ViT_Hybrid(proj_dim=args.dim, K=args.K, m=args.m, T=args.T, embed_dim=args.embedding_dim, device=device)
    model.to(device)

    # Log total parameters for model for comparison
    # ---------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    print_and_log(f"Total model parameters: {total_params:,}", log_file=log_file)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_and_log(f"Trainable model parameters: {trainable_params:,}", log_file=log_file)

    model_size_mb = total_params * 4 / (1024**2)
    print_and_log(f"Model size: {model_size_mb:.2f} MB", log_file=log_file)
    # ---------------------------------------

    base_lr = 1e-4 # Starting point
    # optimizer: update only the query encoder (q_patch, q_vit, q_proj)
    q_params = (
        list(model.patch_embed.parameters()) +
        list(model.pos_encoding.parameters()) +
        list(model.transformer.parameters()) +
        list(model.proj_head.parameters())
    )

    # Original MoCo used SGD; but AdamW is more common for ViT
    optimizer = torch.optim.Adam(q_params, lr=1e-4, weight_decay=1e-5)
    effective_lr = base_lr * (args.batch_size ** 0.5 / 256 ** 0.5)

    # cosine LR scheduler
    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)  # T_max = epochs

    # ---------------------------------------
    # Training Loop
    # ---------------------------------------

    # Run encoder training
    # Ensure q_params is populated
    if q_params is None:
        raise ValueError("q_params is None for ViT Hybrid model; cannot proceed with training.")

    # Run ViT Hybrid specific training function
    model, train_stats = run_moco_training_vit_hybrid(
        model,
        n_epochs=args.n_epochs,
        train_loader=train_loader,
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler_cos=scheduler_cos,
        q_params=q_params,
        artifact_root=args.artifact_root,
        base_lr=base_lr,
        log_file=log_file,
    )

    print_and_log('MoCo training complete!!', log_file=log_file)

    # Save the trained encoder (query backbone + mlp)
    # After training, save only 
    #   1) the encoder_q (used as the backbone for transfer learning)
    #   2) encoder_q_proj (for further pre-training; but is generally not used for transfer learning)

    # Create file name for saving model - use args.out_model_name
    model_output_path = os.path.join(args.artifact_root, args.out_model_name)

    # Save encoder_q and mlp_q state dicts
    torch.save({
        'patch_embed': model.patch_embed.state_dict(),
        'pos_encoding': model.pos_encoding.state_dict(),
        'transformer': model.transformer.state_dict(),
        'proj_head': model.proj_head.state_dict(),
        'embedding_dim': model.embed_dim
    }, model_output_path)
    print_and_log(f"Saved pretrained encoder to {model_output_path}", log_file=log_file)

    # Save training stats
    save_stats(train_stats, args.artifact_root + '/pretrain_stats_training.json', 'training', log_file=log_file)

    # Testing will be completed in a separate script
    #if args.run_test:
    #    run_moco_testing(model, train_loader, test_loader, linear_n_epochs=20, device='cuda', num_classes=2, log_file="./artifacts/testing_log.txt", artifact_root='./')

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
    parser.add_argument('--run_test', type=bool, default=False, help='Whether to run testing after training')

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
    
    # For ViT Hybrid - [Case 2]
    parser.add_argument('--embedding_dim', type=int, default=384, help='Embedding dimension for ViT Hybrid backbone')

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