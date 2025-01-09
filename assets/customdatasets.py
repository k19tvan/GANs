import os
import PIL
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        image = PIL.Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image