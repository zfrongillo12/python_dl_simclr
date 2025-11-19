import torchvision.transforms as transforms

# Setting a constant for the Resnet Image Size (224 is expected for ResNet50)
resnet_image_size = 224

# Performing the Data Augmentations that are necessary for Medical Data X-Rays (Resizing, Random Crop, Flipping, Rotation)
# Note: I am normalizing the Tensor, because the ImageNet backbone expects a normalized Tensor
def moco_medical_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(resnet_image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02
            )
        ], p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
