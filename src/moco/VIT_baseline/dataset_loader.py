import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class MoCoTwoCropsTransform:
    """
    Take one image -> return two randomly augmented images.
    This is used in MoCo for contrastive learning; produces the positive pairs (q,k) - query view and positive key
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class MedicalImageDataset(Dataset):
    """
    CSV file with one column: 'Path'
    Can contain absolute or relative file paths.
    """
    def __init__(self, csv_path, root_dir="", transform=None):
        print("Loading dataset from:", csv_path)
        self.df = pd.read_csv(csv_path)
        if 'Path' not in self.df.columns:
            raise ValueError("CSV must contain a 'Path' column")

        self.paths = self.df["Path"].tolist()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Combine root + file path
        img_path = os.path.join(self.root_dir, str(self.paths[idx]))

        # Load as PIL image with exception handling
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not load: {img_path}. Error: {e}")

        if self.transform:
            im_q, im_k = self.transform(img) # Two crops returned
            return im_q, im_k

        # Fallback: return normalized tensors
        base = transforms.ToTensor()
        return base(img), base(img)


def collate_fn(batch):
    """
    Custom collate function to handle batches of (im_q, im_k) tuples.
    """
    im_q = torch.stack([b[0] for b in batch], dim=0)
    im_k = torch.stack([b[1] for b in batch], dim=0)
    return im_q, im_k


def get_moco_medical_loader(csv_path, root_dir, batch_size=64, num_workers=4,data_split_type='train'):
    """
    Creates a MoCo DataLoader for medical images (ViT) compatible using a CSV file.
    """
    print(f"Creating MoCo DataLoader for {data_split_type} data...")
    print(f"CSV Path: {csv_path}")
    print(f"Image Root Dir: {root_dir}")

    # ==== MoCo (ViT) augmentations ====
    # Need to be less aggressive than ResNet50 augmentations
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),

        # Less aggressive for medical (and ViT-safe)
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.05, hue=0.02
        ),

        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0)),
        transforms.ToTensor(),

        # Normalization (ImageNet-compatible)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ==== Dataset ====
    img_root_dir = os.path.join(root_dir, data_split_type)
    dataset = MedicalImageDataset(
        csv_path=csv_path,
        root_dir=img_root_dir,
        transform=MoCoTwoCropsTransform(augmentation)
    )

    # Print size of dataset
    print(f"Training Dataset size: {len(dataset)} images")

    # Print size of one sample
    sample_q, sample_k = dataset[0]
    print(f"Data Sample q shape: {sample_q.shape}, Sample k shape: {sample_k.shape}")

    # Create DataLoader
    drop_last = (len(dataset) % batch_size != 0)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    return loader
