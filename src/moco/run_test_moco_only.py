import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import os
import json

# ResNet50
from model_builder import MoCo
from dataset_loader import get_moco_medical_loader

# ViT
from VIT.model_builder import MoCo as MoCo_ViT
from VIT.dataset_loader import get_moco_medical_loader as get_moco_medical_loader_vit

from classification_dataset import get_classification_data_loader

from utils import set_seed, save_state, print_and_log
from test_moco import run_moco_testing

def save_stats(stats, path, stat_type, log_file=None):
    with open(path, 'w') as f:
        json.dump(stats, f)
    print_and_log(f"Saved {stat_type} stats to {path}", log_file=log_file)
    return

# ================================================================================
# Main function to parse arguments and run testing only
# ================================================================================
def main(args):
    set_seed(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Create log file
    dt = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.artifact_root, f'moco_testing_log_{dt}.txt')
    os.makedirs(args.artifact_root, exist_ok=True)
    print("Created log file: ", log_file)

    # ----------------------------------------------------
    # Model Setup 
    # ----------------------------------------------------
    # build model and load pretrained encoder
    model = MoCo(dim=2048, K=65536, m=0.999, T=0.07)
    model.to(device)

    # Load MoCo model from checkpoint
    print_and_log(f"Loading pretrained MoCo model from {args.pretrained_model_path}...", log_file=log_file)
    ckpt = torch.load(args.pretrained_model_path, map_location=device)
    
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
    print_and_log(f'Loaded pretrained encoder. missing keys: {missing}, unexpected: {unexpected}', log_file=log_file)


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
    parser.add_argument('--pretrained_model_path', type=str, help='Path to the pretrained MoCo model checkpoint for testing')

    # For testing
    parser.add_argument('--test_num_classes', type=int, default=2, help='Number of classes for testing classification')
    parser.add_argument('--linear_n_epochs', type=int, default=30, help='Number of epochs for test linear classification training')
    parser.add_argument('--linear_train_csv_path', type=str, default='linear_train.csv', help='Train CSV file with image paths')
    parser.add_argument('--linear_test_csv_path', type=str, default='linear_test.csv', help='Test CSV file with image paths')
    
    # Optional
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--out_model_name', type=str, default='moco_resnet50_encoder.pth', help='Output filename for encoder')
    parser.add_argument('--label_col', type=str, default='Pneumonia', help='Label column name in CSV for classification dataset')

    # Hyperparameters that can be tuned
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    main(args)