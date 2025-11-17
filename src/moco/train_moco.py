import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model_builder import MoCo
from dataset_loader import get_moco_medical_loader
from utils import set_seed, save_state
from tqdm import tqdm
import argparse
import os
import json

def print_and_log(message, log_file=None):
    # Print to console with datetime
    printedatetime = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[{printedatetime}] {message}')

    # Log to file if log_file is provided
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f'[{printedatetime}] {message}\n')
    return

def run_moco_training(model, n_epochs, train_loader, optimizer, warmup_epochs, scheduler_cos, artifact_root="./artifacts/", base_lr=0.03, log_file="./artifacts/training_log.txt"):
    # Start training log
    print_and_log(f"Starting MoCo training for {n_epochs} epochs", log_file=log_file)

    # Accumulate training stats for pretrainig
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
            print_and_log(f'Epoch {epoch}, Loss: {loss.item():.4f}', log_file=log_file)

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
            print_and_log(f'Saving checkpoint at epoch {epoch+1}', log_file=log_file)
            os.makedirs(artifact_root, exist_ok=True)
            save_state(f'{artifact_root}/moco_checkpoint_epoch_{epoch+1}.pth', model, optimizer, epoch)
        
        # End of epoch: save training stats
        train_stats.append({'epoch': epoch, 'loss': loss.item()})

    return model, train_stats

def main(args):
    set_seed(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'training_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # Get training loader
    train_loader = get_moco_medical_loader(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Use pretrained imagenet weights (ResNet50 backbone is instantiated by MoCo) - by default True
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
    torch.save({
        'encoder_q_state': model.encoder_q.state_dict(),
        'mlp_q_state': model.mlp_q.state_dict()
    }, args.out_path)
    print_and_log('Saved pretrained encoder to', args.out_path)

    # Save training stats
    with open(args.artifact_root + '/train_stats.json', 'w') as f:
        json.dump(train_stats, f)
    print_and_log('Saved training stats to', args.artifact_root + '/train_stats.json')


if __name__ == "__main__":
    # Argument parser for CLI configuration
    parser = argparse.ArgumentParser(description="MoCo Medical Encoder Training")
    # Should be set
    parser.add_argument('--csv_path', type=str, default='train.csv', help='Train CSV file with image paths')
    parser.add_argument('--root_dir', type=str, default='/path/to/dataset', help='Root directory for images')
    parser.add_argument('--artifact_root', type=str, default='./artifacts/', help='Directory for checkpoints')
    
    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')

    # Hyperparameters that can be tuned
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--K', type=int, default=65536, help='Queue size')
    parser.add_argument('--m', type=float, default=0.99, help='Momentum for updating key encoder')
    parser.add_argument('--T', type=float, default=0.2, help='Softmax temperature')
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pretrained_imagenet', type=bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--out_path', type=str, default='moco_resnet50_encoder.pth', help='Output path for encoder')
    parser.add_argument('--base_lr', type=float, default=0.03, help='Base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='SGD weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=25, help='Number of warmup epochs')

    args = parser.parse_args()

    main(args)