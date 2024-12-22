import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class LoveDADataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Carica immagine e maschera
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Applica trasformazioni, se definite
        if self.transform:
            image = self.transform(image)
            mask = T.ToTensor()(mask)  # Converti maschera in tensore
        
        return image, mask.long()  # La maschera deve essere di tipo long
