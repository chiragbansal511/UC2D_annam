from torch.utils.data import Dataset
from PIL import Image
import os
from src.config import label2idx

class SoilDataset(Dataset):
    def __init__(self, df, img_dir, labels=True, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            label = label2idx[self.df.iloc[idx]['label']]
            return image, label
        else:
            return image, img_id
