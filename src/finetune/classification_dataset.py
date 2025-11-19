import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Helper function to check and re-map label values
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

# ===============================================================================
# Labeled Dataset Definition
# ===============================================================================
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
        # Combine root + file path to obtain item
        p = os.path.join(self.root_dir, self.paths[idx])
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label

# ===============================================================================
# DataLoader Creation Functions
# ===============================================================================

def get_classification_data_loader(data_split_type, CSV_PATH, ROOT_DIR, batch_size, num_workers, label_col='Pneumonia'):
    """
    Creates DataLoader for classification dataset.
    * CSV_PATH: Path to the CSV file containing image paths and labels
    * ROOT_DIR: Root directory for images
    * data_split_type: 'train', 'val', or 'test' - used for logging purposes

    Note: the ROOT_DIR is expected to have a 'train' / 'val' / 'test' subdirectory
    """
    # Check data_split_type
    if data_split_type not in ['train', 'val', 'test']:
        raise ValueError("data_split_type must be one of: 'train', 'val', 'test'")

    # Transforms for fine tuning
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the Classification Dataset
    print(f"Loading {data_split_type} dataset...")
    print("CSV:", CSV_PATH)
    print(f" * Images - {data_split_type.capitalize()} Root Directory:", ROOT_DIR + f"/{data_split_type}")

    # Create Dataset
    dataset = PneumoniaClassificationDataset(CSV_PATH, root_dir=ROOT_DIR, transform=transform, label_col=label_col)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
