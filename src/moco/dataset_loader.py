import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_augmentations import moco_medical_transform

# Creating the CheXpert Dataset class for MoCo pretraining
class CheXpertMoCoDataset(Dataset):
    def __init__(self, csv_path, root_directory, use_augmentations=True):
        self.data = pd.read_csv(csv_path)
        self.root_directory = root_directory
        self.use_augmentations = use_augmentations

        # Initializing the augmentation pipeline
        if self.use_augmentations:
            self.transform = moco_medical_transform()
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Getting the full path of the image
        image_path = os.path.join(self.root_directory, row["Path"])

        # Loading image and converting it from grayscale to RGB (to match ResNet-50 input, WHICH EXPECTS RGB)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            # Creating two augmented views for MoCo (following SimCLR/MoCo conventions, we need two differently augmented versions)
            view_one = self.transform(image)
            view_two = self.transform(image)
        else:
            # If transform is off (should not be used for pretraining)
            view_one = image
            view_two = image

        # Creating the label (we are NOT using the label for MoCo pretraining, but storing it anyway)
        label = int(row["Pneumonia"])

        # Returning the two augmented versions of the image
        return view_one, view_two, label


# Creating the DataLoader for MoCo pretraining
def create_moco_dataloader(csv_path, root_directory, batch_size, num_workers=4, shuffle=True):
    dataset = CheXpertMoCoDataset(
        csv_path=csv_path,
        root_directory=root_directory,
        use_augmentations=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
