import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class MoCoTwoCropsTransform:
    """
    Take one image -> return two randomly augmented images.
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
        img_path = os.path.join(self.root_dir, self.paths[idx])

        # Load as PIL image
        img = Image.open(img_path).convert("RGB")  # medical images usually grayscale -> convert to RGB

        if self.transform:
            im_q, im_k = self.transform(img)  # Two crops returned
            return im_q, im_k

        return img, img


def collate_fn(batch):
    """
    Custom collate function to handle batches of (im_q, im_k) tuples.
    """
    im_q = torch.stack([b[0] for b in batch], dim=0)
    im_k = torch.stack([b[1] for b in batch], dim=0)
    return im_q, im_k


def get_moco_medical_loader(csv_path, root_dir, batch_size=64, num_workers=4):
    """
    Creates a MoCo DataLoader for medical images using a CSV file.
    """
    # MoCo v2-style augmentations (tuned for medical images: less color jitter optional)
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
        )], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MedicalImageDataset(csv_path=csv_path, root_dir=root_dir, transform=MoCoTwoCropsTransform(augmentation))

    drop_last = True if len(dataset) % batch_size != 0 else False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=drop_last, collate_fn=collate_fn)
    return loader
