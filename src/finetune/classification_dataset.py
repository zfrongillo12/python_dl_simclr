import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def check_label_values(df, label_col):
    """
    Check the label values in the dataframe column.
    
    If negative labels are found, re-map them to non-negative values suitable for Cross Entropy Loss.

    e.g.) For Stanford Pneumonia dataset:
        * Original Labels(for Pneumonia): -1 (No Finding), 0 (Other), 1 (Pneumonia)
        * New Labels: 3 (No Finding), 0 (Other), 1 (Pneumonia)
    """
    unique_labels = df[label_col].unique()
    print(f"Unique labels in column '{label_col}': {unique_labels}")

    # Check for negative labels
    if any(label < 0 for label in unique_labels):
        print("Warning: Negative labels found. Will re-map labels for Cross Entropy Loss.")
        
        # Remap labels: -1 -> 3, 0 -> 0, 1 -> 1
        df[label_col] = df[label_col].replace({-1: 3, 0: 0, 1: 1})
    
        # Sanity check
        print(f"POST-update: Unique labels in column '{label_col}': {unique_labels}")
    return df


class PneumoniaClassificationDataset(Dataset):
    """
    CSV with columns: Path, Pneumonia (0/1 or -1 etc.). Provide csv and root_dir
    to load images and labels for classification.

    Will re-map labels to be non-negative for Cross Entropy Loss.
    """
    def __init__(self, csv_path, root_dir='', transform=None, label_col='Pneumonia'):
        # Read in CSV to df
        self.df = pd.read_csv(csv_path)
        
        # Check required columns
        if 'Path' not in self.df.columns:
            raise ValueError("CSV must contain a 'Path' column")
        if label_col not in self.df.columns:
            raise ValueError(f"CSV must contain a label column named {label_col}")
        
        # Obtain image paths
        self.paths = self.df['Path'].tolist()

        # Obtain labels - re-map values if necessary
        # Cross Entropy requires non-negative labels (e.g., 0,1,2)
        self.df = check_label_values(self.df, label_col)
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

def get_train_val_loaders(TRAIN_CSV, VAL_CSV, ROOT_DIR, BATCH_SIZE, NUM_WORKERS, label_col='Pneumonia'):
    """
    Creates DataLoaders for training and validation classification datasets.

    Note: the ROOT_DIR is expected to have 'train' and 'val' subdirectories
    """

    # Transforms for fine tuning
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the train/val Classification Datasets
    print("Loading training and validation datasets...")
    print("Training CSV:", TRAIN_CSV)
    print(" * Images - Train Root Directory:", ROOT_DIR + "/train")
    print("Validation CSV:", VAL_CSV)
    print(" * Images - Val Root Directory:", ROOT_DIR + "/val")

    train_ds = PneumoniaClassificationDataset(TRAIN_CSV, root_dir=ROOT_DIR + "/train", transform=transform, label_col=label_col)
    val_ds = PneumoniaClassificationDataset(VAL_CSV, root_dir=ROOT_DIR + "/val", transform=transform, label_col=label_col)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader


def get_test_loader(TEST_CSV, ROOT_DIR, BATCH_SIZE, NUM_WORKERS, label_col='Pneumonia'):
    """
    Creates DataLoader for test classification dataset.

    Note: the ROOT_DIR is expected to have a 'test' subdirectory
    """

    # Transforms for fine tuning
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test Classification Dataset
    print("Loading test dataset...")
    print("Test CSV:", TEST_CSV)
    print(" * Images - Test Root Directory:", ROOT_DIR + "/test")

    test_ds = PneumoniaClassificationDataset(TEST_CSV, root_dir=ROOT_DIR + "/test", transform=transform, label_col=label_col)

    # Create DataLoader
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return test_loader