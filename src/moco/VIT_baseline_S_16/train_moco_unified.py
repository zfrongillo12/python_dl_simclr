# =====================================================================
# Unified MoCo Training Script (Hybrid ViT, ViT-S/16 (timm), ResNet-50)
# Preserves structure and style from original ViT-Hybrid script
# =====================================================================

import torch
import torch.nn.functional as F

import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import os
import json
import datetime

# =====================================================================
# Model Imports
# IMPORTANT: EDIT THESE THREE IMPORTS BEFORE RUNNING
# =====================================================================

# 1. Hybrid ViT
# Example: from VIT_update_hybrid.model_builder import ViTMoCo as MoCo_ViT_Hybrid
from VIT_hybrid.model_builder import ViTMoCo as MoCo_ViT_Hybrid

# 2. ViT-S/16 (timm)
# Example: from timm_moco.vit_small_moco import MoCoViTSmall
from ViT_baseline.model_builder import MoCo as MoCoViTSmall

# 3. ResNet-50 MoCo
# Example: from resnet_moco.resnet50_moco import MoCoResNet50
from Resnet_50.model_builder import MoCo as MoCoResNet50

# Dataset loader (same for all models now)
from dataset_loader import get_moco_medical_loader

# Utilities
from utils import set_seed, save_state, print_and_log, save_stats, compute_diagnostics


# ================================================================================
# Unified MoCo Training Loop (extended for FP32/FP16/BF16 and diagnostics)
# ================================================================================

def run_moco_training(model,
                      n_epochs,
                      train_loader,
                      optimizer,
                      warmup_epochs,
                      scheduler_cos,
                      q_params,
                      artifact_root="./artifacts/",
                      base_lr=0.03,
                      log_file="./artifacts/training_log.txt",
                      max_moco_steps=None,
                      precision="fp32"):
    # Start training log
    print_and_log(f"Starting Unified MoCo training for {n_epochs} epochs...", log_file=log_file)
    print_and_log(f"Precision mode = {precision}", log_file=log_file)

    # Accumulate training stats for pretraining
    train_stats = []

    best_loss = float("inf")
    global_step = 0

    # AMP precision selection
    if precision == "bf16":
        dtype_autocast = torch.bfloat16
    elif precision == "fp16":
        dtype_autocast = torch.float16
    else:
        # fallback for FB32
        dtype_autocast = None
    scaler = torch.amp.GradScaler(device='cuda', enabled=(precision == "fp16"))

    for epoch in range(n_epochs):
        # Set model to train mode
        model.train()

        # Loop on the training data - using tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs - 1}')

        running_loss = 0.0
        running_diag = {"pos_cos": 0.0, "q_norm": 0.0, "k_norm": 0.0}
        diag_count = 0

        # Training loop - for each batch
        for im_q, im_k in loop:

            # ---- EARLY STEP-LIMIT PATCH ----
            if max_moco_steps is not None and global_step >= max_moco_steps:
                print_and_log(f"Stopping MoCo early at step {global_step}", log_file=log_file)
                return model, train_stats
            # --------------------------------
            global_step += 1

            im_q = im_q.to(model.device, non_blocking=True)
            im_k = im_k.to(model.device, non_blocking=True)

            optimizer.zero_grad()

            # ========================== AMP BLOCK ==========================
            if dtype_autocast is None:
                # FP32 path
                logits, labels, q_vec, k_vec = model(im_q, im_k, return_features=True)
                loss = F.cross_entropy(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_params, max_norm=3.0)
                optimizer.step()

            else:
                # FP16 or BF16
                with autocast(dtype=dtype_autocast):
                    logits, labels, q_vec, k_vec = model(im_q, im_k, return_features=True)
                    loss = F.cross_entropy(logits, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(q_params, max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
            # =================================================================

            running_loss += loss.item()

            # ----------- DIAGNOSTICS --------------
            diag = compute_diagnostics(model, q_vec, k_vec)
            for k in running_diag:
                running_diag[k] += diag[k]
            diag_count += 1
            # --------------------------------------

            loop.set_postfix({'loss': loss.item(), "pos_cos": diag["pos_cos"]})

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

        avg_diag = {k: running_diag[k] / diag_count for k in running_diag}
        print_and_log(f"POS_COS={avg_diag['pos_cos']:.4f}", log_file=log_file)

        # Save checkpoint when loss improves or periodically
        if avg_running_loss < best_loss or epoch == 0 or (epoch + 1) % 5 == 0:
            if epoch != 0:
                print_and_log(f'Loss improved from {best_loss:.6f} to {avg_running_loss:.6f}. Saving checkpoint.', log_file=log_file)
            else:
                print_and_log(f'Saving initial checkpoint at epoch {epoch+1}; Avg. Running Loss: {avg_running_loss:.6f}', log_file=log_file)

            best_loss = avg_running_loss

            model_checkpoint_path = f'{artifact_root}/checkpoint_epoch_{epoch+1}.pth'
            os.makedirs(artifact_root, exist_ok=True)
            save_state(model_checkpoint_path, model, optimizer, epoch)
            print_and_log(f'Saving checkpoint at epoch {epoch+1}; {model_checkpoint_path}', log_file=log_file)

        # End of epoch: save training stats
        train_stats.append({'epoch': epoch,
                            'loss': avg_running_loss,
                            **avg_diag})

    return model, train_stats



# ================================================================================
# Unified Model Builder – this can now handle all 3 backbones (in a modular fashion)
# ================================================================================

def build_model(model_type, args, device):

    if model_type == "HYBRID":
        model = MoCo_ViT_Hybrid(
            proj_dim=args.dim,
            K=args.K,
            m=args.m,
            T=args.T,
            embed_dim=args.embedding_dim,
            device=device
        )

    elif model_type == "VIT_S16":
        model = MoCoViTSmall(
            dim=args.dim,
            K=args.K,
            m=args.m,
            T=args.T,
            device=device
        )

    elif model_type == "RESNET50":
        model = MoCoResNet50(
            dim=args.dim,
            K=args.K,
            m=args.m,
            T=args.T,
            device=device
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model



# ================================================================================
# Main function to parse arguments and run training
# Preserves original ordering + logging style
# ================================================================================

def main(args):
    set_seed(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'moco_training_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # ---------------------------------------
    # Get training loader
    # ---------------------------------------
    print_and_log(f"Creating MoCo medical image Train DataLoader... : from {args.train_csv_path}", log_file=log_file)

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
    print_and_log(f"Using backbone: {args.model_type}", log_file=log_file)
    model = build_model(args.model_type, args, device)
    model.to(device)

    # optimizer: update only the query encoder
    q_params = model.get_trainable_parameters()

    optimizer = torch.optim.Adam(q_params, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # ---------------------------------------
    # Training Loop
    # ---------------------------------------
    model, train_stats = run_moco_training(
        model,
        n_epochs=args.n_epochs,
        train_loader=train_loader,
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        scheduler_cos=scheduler_cos,
        q_params=q_params,
        artifact_root=args.artifact_root,
        base_lr=args.base_lr,
        log_file=log_file,
        max_moco_steps=args.max_moco_steps,
        precision=args.precision
    )

    print_and_log('MoCo training complete!!', log_file=log_file)

    # Save the trained encoder
    model_output_path = os.path.join(args.artifact_root, args.out_model_name)

    encoder_state = model.get_encoder_state()
    torch.save(encoder_state, model_output_path)

    print_and_log(f"Saved pretrained encoder to {model_output_path}", log_file=log_file)

    # Save training stats
    save_stats(train_stats, args.artifact_root + '/pretrain_stats_training.json', 'training', log_file=log_file)



# ================================================================================
# Entry point / Argument parsing
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified MoCo Medical Encoder Training")

    # Should be set
    parser.add_argument('--train_csv_path', type=str, required=True, help='Train CSV file with image paths')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory for images')
    parser.add_argument('--artifact_root', type=str, default='./artifacts/', help='Directory for checkpoints')
    parser.add_argument('--model_type', type=str, choices=["HYBRID", "VIT_S16", "RESNET50"], required=True)

    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs for backbone training')
    parser.add_argument('--out_model_name', type=str, default='moco_encoder.pth', help='Output filename for encoder')
    parser.add_argument('--max_moco_steps', type=int, default=None, help='Stop MoCo training after this many steps')

    parser.add_argument('--embedding_dim', type=int, default=512, help='Embedding dimension for ViT Hybrid backbone')

    # Hyperparameters
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--K', type=int, default=65536, help='Queue size')
    parser.add_argument('--m', type=float, default=0.99, help='Momentum for updating key encoder')
    parser.add_argument('--T', type=float, default=0.2, help='Softmax temperature')
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Adam weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')

    parser.add_argument("--precision", type=str,
                        choices=["fp32", "fp16", "bf16"],
                        default="fp32",
                        help="Training precision mode (FP32 / AMP FP16 / BF16)")

    args = parser.parse_args()
    main(args)
