import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ClassificationDataset(Dataset):
    """
    CSV with columns: Path, Pneumonia (0/1 or -1 etc.). Provide csv and root_dir
    """
    def __init__(self, csv_path, root_dir='', transform=None, label_col='Pneumonia'):
        self.df = pd.read_csv(csv_path)
        if 'Path' not in self.df.columns:
            raise ValueError("CSV must contain a 'Path' column")
        if label_col not in self.df.columns:
            raise ValueError(f"CSV must contain a label column named {label_col}")
        self.paths = self.df['Path'].tolist()
        # Cross Entropy requires non-negative labels (e.g., 0,1,2)
        # Original Labels(for Pneumonia): -1 (No Finding), 0 (Other), 1 (Pneumonia)
        # New Labels: 0 (No Finding), 1 (Other), 2 (Pneumonia)
        # Remap labels: -1 -> 0, 0 -> 1, 1 -> 2
        self.labels = [x + 1 for x in self.df[label_col].tolist()]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = os.path.join(self.root_dir, self.paths[idx])
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
    
def get_train_val_loaders(TRAIN_CSV, VAL_CSV, ROOT_DIR, BATCH_SIZE, NUM_WORKERS):
    # Transforms for fine tuning
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the train/val Classification Datasets
    train_ds = ClassificationDataset(TRAIN_CSV, root_dir=ROOT_DIR, transform=transform)
    val_ds = ClassificationDataset(VAL_CSV, root_dir=ROOT_DIR, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader
