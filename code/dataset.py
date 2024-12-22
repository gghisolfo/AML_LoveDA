from torch.utils.data import DataLoader

# Percorsi al dataset
train_images_dir = "/path/to/LoveDA/Train/Images"
train_masks_dir = "/path/to/LoveDA/Train/Annotations"

# Dataset e DataLoader
train_dataset = LoveDADataset(images_dir=train_images_dir, masks_dir=train_masks_dir, transform=transform)
dataloader_train = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

for images, masks in dataloader_train:
    print(images.shape)  # Dovrebbe essere [batch_size, 3, H, W]
    print(masks.shape)   # Dovrebbe essere [batch_size, H, W]
    break
